import numpy as np
import torch
import gym
import argparse
import os

from TD3 import TD3, utils
from diffusion import Diffusion


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean=0, std=1, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array((state-mean)/std))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="Hopper-v2")               # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--beta", default=1.0, type=float)          # Reward Shaper [..., 0.1, 1.0, 10, ...]
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}_beta{args.beta}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	# Target policy smoothing is scaled wrt the action scale
	kwargs["policy_noise"] = args.policy_noise * max_action
	kwargs["noise_clip"] = args.noise_clip * max_action
	kwargs["policy_freq"] = args.policy_freq
	policy = TD3.TD3(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	diffusion_dir = None
	if 'Walker2d' in args.env:
		diffusion_dir = ''
	elif 'Hopper' in args.env:
		diffusion_dir = 'results/hopper-expert-v2_0_n3_1776775902'
		max_state = [0.9935811,-0.19259559,-0.96671677,-0.87913865,-0.90737796,-0.06279139,-2.8442838,-3.7573864,-7.0393734,-10.0,-10.0]
		min_state = [1.6400596,0.058619823,0.00092868117,0.035054278,0.8937572,4.61973,2.966122,2.2793424,8.878098,7.2151656,10.0]
		mean_state = [1.3662088,-0.11727711,-0.55273896,-0.15372501,-0.009590324,2.7499044,0.022893708,-0.004142655,-0.07470939,-0.042105675,0.08156684]
		std_state = [0.16130634,0.043242376,0.1518521,0.21312758,0.5942794,0.65053856,1.5456787,0.77958363,2.120829,2.8511946,5.8828473]
	elif 'HalfCheetah' in args.env:
		diffusion_dir = ''

	else:
		print('Please, check the environment.')

	diffusion = Diffusion(state_dim, torch.FloatTensor(min_state), torch.FloatTensor(max_state),
						  torch.FloatTensor(mean_state), torch.FloatTensor(std_state),
                          1024, 4,
                          0.00003, 0.00001, 'cuda', True)
	diffusion.load_model(diffusion_dir)

	state, done = env.reset(), False
	episode_reward = 0
	episode_real_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array((state-mean_state)/(std_state)))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, real_reward, done, _ = env.step(action)
		# reward = -0.001*np.log(-args.beta*diffusion.get_reward(torch.FloatTensor(state), t=0.9, use_v=False))
		reward = np.exp(args.beta*diffusion.get_reward(torch.FloatTensor(state), t=0.9, use_v=False))
		# reward = np.exp(args.beta*diffusion.get_reward2(torch.FloatTensor(state), t=0.5))
		# reward = real_reward
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add((state-mean_state)/(std_state), action, (next_state-mean_state)/(std_state), reward, done_bool)

		state = next_state
		episode_reward += reward
		episode_real_reward += real_reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Diffusion Reward: {episode_reward:.3f} Reward: {episode_real_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_real_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed, mean_state, std_state))
			np.save(f"./results/{file_name}norm", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
