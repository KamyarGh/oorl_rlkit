import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp

class Model(nn.Module):
    def __init__(self, input_dim, num_layer_blocks=2, hid_dim=100, use_bn=True):
        super().__init__()

        self.mod_list = nn.ModuleList([nn.Linear(input_dim, hid_dim)])
        if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
        self.mod_list.append(nn.Tanh())

        for i in range(num_layer_blocks - 1):
            self.mod_list.append(nn.Linear(hid_dim, hid_dim))
            if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
            self.mod_list.append(nn.Tanh())
        
        self.mod_list.append(nn.Linear(hid_dim, 1))
        self.model = nn.Sequential(*self.mod_list)


    def forward(self, obs_batch, act_batch):
        input_batch = torch.cat([obs_batch, act_batch], dim=1)
        return self.model(input_batch)


class MlpGAILDisc(Mlp):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)

    def forward(self, obs_batch, act_batch):
        input_batch = torch.cat([obs_batch, act_batch], dim=1)
        return super().forward(input_batch)
