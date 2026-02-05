import numpy
import torch

class Replay_Buffer(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
    
    def load_dataset(self, data):
        return data

    def sample(self):
        return {}