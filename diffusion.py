import numpy as np
import torch


class Diffusion(object):
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_layers):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
    
    def loss(self, batch):
        return