import numpy as np
import torch
import gym
import argparse
import os

from TD3 import TD3, utils
from diffusion import Diffusion


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
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
	parser.add_argument("--t_n", default=1, type=int)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
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

	state, done = env.reset(), False
	episode_reward = 0
	episode_real_reward = 0
	episode_timesteps = 0
	episode_num = 0

	diffusion_dir = None
	if 'Walker2d' in args.env:
		# diffusion_dir = 'results/walker2d-expert-v2|exp_1|diffusion-reward|T-5|0'
		diffusion_dir = 'results/walker2d-expert-v2|exp_s|diffusion-reward|h-1024|T-5|b-1024|lr-0.0003|lr_decay|top_n3|0'
		max_state = [ 1.4241449 ,  0.9316807 ,  0.08722941,  0.2225719 ,  1.3556396 ,
			0.24078076,  0.22475049,  1.3066268 ,  6.7584543 ,  2.3756075 ,
		10.        , 10.        , 10.        , 10.        , 10.        ,
		10.        , 10.        ]
		min_state = [  0.8089652 ,  -0.63499796,  -1.1461437 ,  -2.1976676 ,
			-1.1735957 ,  -0.4092272 ,  -1.5569555 ,  -1.2461717 ,
			-2.2380939 ,  -2.2622204 , -10.        , -10.        ,
		-10.        , -10.        , -10.        , -10.        ,
		-10.        ]
	elif 'Hopper' in args.env:
		diffusion_dir = '/home/jongchan/SDIRL/results/hopper-expert-v2_0_1770999261'#'/home/jongchan/SDIRL/results/hopper-expert-v2_0_1770894512/'
		# max_state = [1.6489217,0.115841776,0.006688197,0.04280751,0.8972518,5.362543,2.9994369,2.7899454,9.480361,7.9713435,10.0]
		# min_state = [0.70369244,-0.19259559,-1.0802795,-1.0333455,-0.91142553,-0.10721531,-2.9792914,-3.9308662,-7.243099,-10.0,-10.0]
		max_state = [1.6489217,0.15957971,0.0128377555,0.043391787,0.9100201,5.5477157,2.9994369,3.753021,10.0,8.021581,10.0]
		min_state = [0.70369244,-0.19535631,-1.0802795,-1.1179677,-0.9210384,-0.10902443,-5.5818257,-4.16065,-7.6354856,-10.0,-10.0]
	elif 'HalfCheetah' in args.env:
		diffusion_dir = 'results/halfcheetah-expert-v2|exp_1|diffusion-reward|T-5|0'
		max_state = [ 1.1942351, 16.129652 ,  1.1130722,  0.8880837,  0.8279613,
			0.8960234,  1.0304255,  0.6605752, 14.500948 ,  4.665963 ,
		11.089605 , 24.24637  , 31.478853 , 23.72161  , 30.337742 ,
		29.96998  , 25.469524 ]
		min_state = [ -0.5885626 ,  -0.73779875,  -0.6496971 ,  -0.98177534,
			-0.65364826,  -1.2188079 ,  -1.21969   ,  -0.6229152 ,
			-3.2509532 ,  -4.8543468 ,  -9.26473   , -26.298426  ,
		-32.988655  , -30.405226  , -28.336708  , -29.96498   ,
		-23.619951  ]
	else:
		print('Please, check the environment.')

	diffusion = Diffusion(state_dim, torch.FloatTensor(min_state), torch.FloatTensor(max_state), 0, 1, 
                          1024, 4,
                          0.00003, 0.00001, 'cuda', True)
	diffusion.load_model(diffusion_dir)

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, real_reward, done, _ = env.step(action)
		# reward = 5+diffusion.get_reward(torch.FloatTensor(state), t=0.8) # ddim?? score function?
		# reward = 0.01*np.log(-diffusion.get_reward(torch.FloatTensor(state), t=0.8))
		reward = 1*np.exp(diffusion.get_reward(torch.FloatTensor(state), t=0.9))
		# reward = real_reward
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

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
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}_exp1_261_09", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
