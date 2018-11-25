import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule

class Model(nn.Module):
    def __init__(self, input_dim, hid_dim=100):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, obs_batch, act_batch):
        input_batch = torch.cat([obs_batch, act_batch], dim=1)
        return self.model(input_batch)
