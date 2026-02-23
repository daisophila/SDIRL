import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import json
import csv
import time
import tqdm

import gym
import d4rl

from diffusion import Diffusion
import utils

from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'hopper-expert-v2', 'Environment and dataset for IRL')
flags.DEFINE_integer('seed', 0, 'Random seed')

flags.DEFINE_integer('top_n', 5, 'Select N trajectories of top Return')

flags.DEFINE_boolean('use_ema', True, 'Use Exponential moving average for model update')

# flags.DEFINE_string('model', 'MLP', 'Network architecture for learning')
flags.DEFINE_integer('hidden_dim', 1024, 'Hidden unit size')
flags.DEFINE_integer('num_hidden_layers', 4, 'Number of hidden layers')

flags.DEFINE_integer('epochs', 1000000, 'Total number of epoch for laerning')
flags.DEFINE_float('lr', 1e-4, 'Learning rate')
flags.DEFINE_float('weight_decay', 1e-2, 'Learning rate')
# flags.DEFINE_boolean('lr_sch', True, 'Use learning rate schedule')
flags.DEFINE_integer('batch_size', 1024, 'Batch size')

flags.DEFINE_string('device', 'cuda', 'Hardware accelerator for learning')

flags.DEFINE_string('save_path', './results/', 'Model save folder')



def main(argv):
    exp_dir = FLAGS.save_path + f'{FLAGS.env}_{FLAGS.seed}_{time.time():.0f}'
    os.makedirs(exp_dir)

    flags_dict = FLAGS.flag_values_dict()
    with open(f'{exp_dir}/config.json', 'w') as f:
        json.dump(flags_dict, f, indent=4)

    env = gym.make(FLAGS.env)

    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = utils.Trajectories(env.get_dataset(), FLAGS.top_n)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False)

    x_min = dataset.states.min(axis=0).values
    x_max = dataset.states.max(axis=0).values
    x_mean = dataset.states.mean(axis=0)
    x_std = dataset.states.std(axis=0)

    with open(f'{exp_dir}/x_min_max.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(np.array(x_min))
        writer.writerow(np.array(x_max))
    
    diffusion = Diffusion(state_dim, x_min, x_max, x_mean, x_std, 
                          FLAGS.hidden_dim, FLAGS.num_hidden_layers,
                          FLAGS.lr, FLAGS.weight_decay, FLAGS.device, FLAGS.use_ema)

    for epoch in range(FLAGS.epochs):
        # epoch_loss = 0.
        # batches = 0
        for batch in dataloader:
            info = diffusion.update(batch)
            # epoch_loss += info['loss']
            # batches += 1
        
        if (epoch+1)%10000==0:
            epoch_loss = 0.
            batches = 0
            for batch in dataloader:
                loss = diffusion.get_reward(batch['states'])
                epoch_loss -= loss
                batches += 1
            print(f'Epoch: {epoch+1}/{FLAGS.epochs}, Diffusion EMA loss: {(epoch_loss/batches):.4f}')

            with open(f'{exp_dir}/training_loss.csv', 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, round(epoch_loss/batches, 4)])
        
    diffusion.save_model(exp_dir)


if __name__ == '__main__':
    app.run(main)