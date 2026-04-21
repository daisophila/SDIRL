import numpy as np
import torch
import torch.nn as nn

import copy


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        # 추론에 사용할 가중치를 담을 shadow 모델 생성
        self.shadow = copy.deepcopy(self.model)
        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self):
        # shadow_weight = decay * shadow_weight + (1 - decay) * current_weight
        for shadow_param, current_param in zip(self.shadow.parameters(), self.model.parameters()):
            shadow_param.data.copy_(
                self.decay * shadow_param.data + (1.0 - self.decay) * current_param.data
            )


class MLP(nn.Module):
    def __init__(self, x_dim, hidden_dim, hidden_layers):
        super().__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        self.activation = nn.SiLU() # nn.ReLU()

        layers = []
        layers.append(nn.Linear(self.x_dim+1, self.hidden_dim))
        layers.append(self.activation)
        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self.activation)
        layers.append(nn.Linear(self.hidden_dim, self.x_dim))
        
        self.network = nn.Sequential(*layers)

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)
        return self.network(x)


# class Transfomer(nn.Modeule):
#     def __init__(self, x_dim, hidden_dim, hidden_layers):
#         self.x_dim = x_dim
#         self.hidden_dim = hidden_dim
#         self.hidden_layers = hidden_layers


class Diffusion(object):
    def __init__(self, x_dim, x_min, x_max, x_mean, x_std, hidden_dim, hidden_layers, lr, weight_decay, device, use_ema=True):
        self.x_dim = x_dim
        self.x_min = x_min
        self.x_max = x_max
        self.x_mean = x_mean
        self.x_std = x_std
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.device = device

        self.model = MLP(self.x_dim, self.hidden_dim, self.hidden_layers).to(self.device)
        # self.model = Transfomer(self.x_dim, self.hidden_dim, self.hidden_layers).to(self.device)
        self.optim = torch.optim.RAdam(self.model.parameters(), lr=self.lr, 
                                        weight_decay=weight_decay, betas=(0.9, 0.999))
        # self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.use_ema = use_ema
        if self.use_ema:
            self.ema = EMA(self.model, decay=0.999)
        
    
    def update(self, batch):
        with torch.no_grad():
            x = 2*(batch['states'] - self.x_min)/(self.x_max - self.x_min + 1e-6) - 1
            # x = (batch['states']-self.x_mean)/(self.x_std+1e-6)
        
        loss = self.loss(x.to(self.device))
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        
        if self.use_ema:
            self.ema.update()

        return {'loss': loss.item()}


    def loss(self, x_1):
        eps = torch.randn_like(x_1, requires_grad=False).to(self.device)
        # t = torch.rand((len(x_1), 1), requires_grad=False).to(self.device)
        t = torch.randn((len(x_1), 1), requires_grad=False).to(self.device)
        t = torch.sigmoid(t * 0.8 - 0.8)
        
        x_t = t*x_1 + (1-t)*eps
        v = (x_1 - x_t)/(1-t)
        v_pred = (self.model(x_t, t)-x_t)/(1-t)
        return torch.mean((v_pred-v)**2)


    def get_reward(self, x_1, t=0.5, use_v=True):
        with torch.no_grad():
            x_1 = 2*(x_1 - self.x_min)/(self.x_max - self.x_min + 1e-6) - 1
            # x_1 = (x_1-self.x_mean)/(self.x_std+1e-6)
            x_1 = x_1.to(self.device)
        eps = torch.randn_like(x_1, requires_grad=False).to(self.device)
        t = torch.full((len(x_1), 1), t, requires_grad=False).to(self.device)
        x_t = t*x_1 + (1-t)*eps
        # using v loss
        if use_v:
            v = x_1 - eps
            # v_pred = (self.ema.shadow(x_t, t)-x_t)/(1-t)
            v_pred = (self.model(x_t, t)-x_t)/(1-t)
            return -torch.mean((v_pred-v)**2).item()
        # using x loss
        else:
            x_t = t*x_1 + (1-t)*eps
            # return -torch.mean((self.ema.shadow(x_t, t)-x_1)**2).item()
            return -torch.mean((self.model(x_t, t)-x_1)**2).item()

    
    def save_model(self, dir, name=None):
        if name is not None:
            torch.save(self.model.state_dict(), f'{dir}/diffusion_{name}.pth')
            torch.save(self.ema.shadow.state_dict(), f'{dir}/diffusion_ema_{name}.pth')
        else:
            torch.save(self.model.state_dict(), f'{dir}/diffusion.pth')
            torch.save(self.ema.shadow.state_dict(), f'{dir}/diffusion_ema.pth')

    def load_model(self, dir, name=None):
        if name is not None:
            self.model.load_state_dict(torch.load(f'{dir}/diffusion_{name}.pth'))
            self.ema.shadow.load_state_dict(torch.load(f'{dir}/diffusion_ema_{name}.pth'))
        else:
            self.model.load_state_dict(torch.load(f'{dir}/diffusion.pth'))
            self.ema.shadow.load_state_dict(torch.load(f'{dir}/diffusion_ema.pth'))

        