import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, x_dim, hidden_dim, hidden_layers):
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        self.activation = nn.ReLU()

        layers = []
        layers.append(nn.Linear(self.x_dim+1, self.hidden_dim))
        layers.append(self.activation)
        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self.activation)
        layers.append(nn.Linear(self.hidden_dim, self.x_dim))
        
        self.network = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    
    def foward(self, x, t):
        x = torch.cat([x, t], dim=1)
        return self.network(x)


class Diffusion(object):
    def __init__(self, x_dim, hidden_dim, hidden_layers, device):
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.device = device

        self.model = MLP(self.x_dim, self.hidden_dim, self.hidden_layers).to(self.device)
        self.optim = torch.optim.RAdam(self.model.params())

    
    def update(self, batch):
        loss = self.loss(batch['states'])

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def loss(self, x_1):
        eps = torch.randn_like(x_1, requires_grad=False)
        t = torch.rand_like(len(x_1), requires_grad=False)
        v = x_1 - eps
        x_t = t*x_1 + (1-t)*eps
        v_pred = (self.model(x_t, t)-x_t)/(1-t)
        return torch.mean((v_pred-v)**2)

    def get_reward(self, x_1):
        return 0