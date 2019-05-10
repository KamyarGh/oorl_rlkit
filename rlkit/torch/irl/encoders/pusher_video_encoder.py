import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.torch_meta_irl_algorithm import np_to_pytorch_batch
from rlkit.torch.distributions import ReparamMultivariateNormalDiag

class PusherVideoEncoder(PyTorchModule):
    def __init__(
        self,
        # z_dims, # z will take the form of a conv kernel
    ):
        self.save_init_params(locals())
        super().__init__()

        # self.z_dims = [-1] + z_dims

        CH = 32
        # kernel = (10, 10, 10) # depth, height, width
        kernel = (3, 5, 5) # depth, height, width
        stride = (2, 2, 2) # depth, height, width
        # self.conv = nn.Sequential(
        #     nn.Conv3d(3, CH, kernel, stride=stride, bias=True),
        #     nn.ReLU(),
        #     nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
        #     nn.ReLU(),
        #     nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
        #     nn.ReLU(),
        #     # nn.Conv3d(CH, CH, (1,1,1), (1,1,1), bias=True),
        # )
        self.all_convs = nn.ModuleList(
            [
                nn.Conv3d(3, CH, kernel, stride=stride, bias=True),
                nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
                nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
                nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
            ]
        )

        FLAT_SIZE = CH*1*5*5
        self.fc = nn.Sequential(
            nn.Linear(FLAT_SIZE, FLAT_SIZE),
            nn.ReLU(),
            nn.Linear(FLAT_SIZE, FLAT_SIZE),
            nn.ReLU(),
            nn.Linear(FLAT_SIZE, 3*8*5*5 + 16*8*5*5 + 16*8*5*5)
            # nn.Linear(FLAT_SIZE, 3*8*5*5 + 16*8*5*5 + 16*8*5*5 + 16*8*5*5)
        )
        # self.z1_fc = nn.Linear(FLAT_SIZE, 3*8*5*5)
        # self.z2_fc = nn.Linear(FLAT_SIZE, 16*8*5*5)
        # self.z3_fc = nn.Linear(FLAT_SIZE, 16*8*5*5)
        # self.z4_fc = nn.Linear(FLAT_SIZE, 16*8*5*5)
        # FLAT_SIZE = CH*13*13
        # OUT_SIZE = int(np.prod(z_dims))
        # self.fc = nn.Sequential(
        #     nn.Linear(FLAT_SIZE, OUT_SIZE),
        #     nn.ReLU(),
        #     nn.Linear(OUT_SIZE, OUT_SIZE),
        #     nn.ReLU(),
        #     nn.Linear(OUT_SIZE, OUT_SIZE)
        # )
    
    def forward(self, demo_batch):
        h = demo_batch
        for i, c in enumerate(self.all_convs):
            h = c(h)
            h = F.relu(h)
        output = h
        output = torch.sum(output, dim=2)
        output = output.view(
            output.size(0),
            output.size(1) * output.size(2) * output.size(3)
        )
        # z = self.fc(output).view(*self.z_dims)
        z = self.fc(output)
        return z


class PusherLastTimestepEncoder(PyTorchModule):
    def __init__(
        self,
        # z_dims, # z will take the form of a conv kernel
    ):
        self.save_init_params(locals())
        super().__init__()
        
        CH = 32
        k = 5
        s = 2
        p = 2
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, CH, k, stride=s, padding=p),
            nn.BatchNorm2d(CH),
            nn.ReLU(),
            nn.Conv2d(CH, CH, k, stride=s, padding=p),
            nn.BatchNorm2d(CH),
            nn.ReLU(),
            nn.Conv2d(CH, CH, k, stride=s, padding=p),
            nn.BatchNorm2d(CH),
            nn.ReLU(),
        )
        flat_dim = CH * 6 * 6
        self.fc_part = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

    
    def forward(self, demo_batch):
        conv_output = self.conv_part(demo_batch)
        conv_output = conv_output.view(conv_output.size(0), -1)
        fc_output = self.fc_part(conv_output)

        return fc_output


if __name__ == '__main__':
    p = PusherVideoEncoder()
    x = Variable(torch.rand(16, 3, 100, 125, 125))
    p.cuda()
    x = x.cuda()
    y = p(x)
    print(torch.max(y), torch.min(y), y.size())
