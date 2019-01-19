import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.torch_meta_irl_algorithm import np_to_pytorch_batch
from rlkit.torch.irl.encoders.aggregators import sum_aggregator_unmasked, tanh_sum_aggregator_unmasked
from rlkit.torch.distributions import ReparamMultivariateNormalDiag


class TrivialTrajEncoder(PyTorchModule):
    '''
        Assuming length of trajectories is 65 and we will take [::4] so 17 timesteps
        Dimensions are hard-coded for few shot fetch env
    '''
    def __init__(
        self,
    ):
        self.save_init_params(locals())
        super().__init__()

        # V0
        # self.conv_part = nn.Sequential(
        #     nn.Conv1d(26, 50, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(50),
        #     nn.ReLU(),
        #     nn.Conv1d(50, 50, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(50),
        #     nn.ReLU()
        # )
        # self.mlp_part = nn.Sequential(
        #     nn.Linear(150, 50)
        # )
        # self.output_size = 50

        # V1
        # self.conv_part = nn.Sequential(
        #     nn.Conv1d(26, 128, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        # )

        # V1 for subsample 8
        self.conv_part = nn.Sequential(
            nn.Conv1d(26, 128, 4, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True),
        )


    def forward(self, all_timesteps):
        # if traj len 16 comment the next line out
        # all_timesteps = all_timesteps[:,:,::4,:]

        all_timesteps = all_timesteps.permute(0,1,3,2).contiguous()

        N_tasks, N_trajs, dim, traj_len = all_timesteps.size(0), all_timesteps.size(1), all_timesteps.size(2), all_timesteps.size(3)

        # V0
        # all_timesteps = all_timesteps.view(N_tasks*N_trajs, dim, traj_len)
        # embeddings = self.conv_part(all_timesteps)
        # embeddings = embeddings.view(N_tasks*N_trajs, -1)
        # embeddings = self.mlp_part(embeddings)
        # embeddings = embeddings.view(N_tasks, N_trajs, -1)

        # V1
        all_timesteps = all_timesteps.view(N_tasks*N_trajs, dim, traj_len)
        embeddings = self.conv_part(all_timesteps)
        embeddings = embeddings.view(N_tasks, N_trajs, -1)

        return embeddings


class TrivialR2ZMap(PyTorchModule):
    def __init__(
        self,
        z_dim
    ):
        self.save_init_params(locals())
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mean_fc = nn.Linear(128, z_dim)
        self.log_sig_fc = nn.Linear(128, z_dim)


    def forward(self, r):
        trunk_output = self.trunk(r)
        mean = self.mean_fc(trunk_output)
        log_sig = self.log_sig_fc(trunk_output)
        return mean, log_sig


class TrivialNPEncoder(PyTorchModule):
    def __init__(
        self,
        agg_type,
        traj_encoder,
        r_to_z_map
    ):
        self.save_init_params(locals())
        super().__init__()

        self.r_to_z_map = r_to_z_map
        self.traj_encoder = traj_encoder

        if agg_type == 'sum':
            self.agg = sum_aggregator_unmasked
        elif agg_type == 'tanh_sum':
            self.agg = tanh_sum_aggregator_unmasked
        else:
            raise Exception('Not a valid aggregator!')
    

    def forward(self, context):
        '''
            For this first version of trivial encoder we are going
            to assume all tasks have the same number of trajs and
            all trajs have the same length
        '''
        # first convert the list of list of dicts to a tensor of dims
        # N_tasks x N_trajs x Len x Dim

        # obs = np.array([[d['observations'] for d in task_trajs] for task_trajs in context])
        # acts = np.array([[d['actions'] for d in task_trajs] for task_trajs in context])

        # if traj len 16 use this instead of the above two lines
        # obs = np.array([[d['observations'][:16,...] for d in task_trajs] for task_trajs in context])
        # acts = np.array([[d['actions'][:16,...] for d in task_trajs] for task_trajs in context])

        # if traj len 8 use this instead of the above two lines
        obs = np.array([[d['observations'][:8,...] for d in task_trajs] for task_trajs in context])
        acts = np.array([[d['actions'][:8,...] for d in task_trajs] for task_trajs in context])

        all_timesteps = np.concatenate([obs, acts], axis=-1)
        # print('\n'*20)
        # print(all_timesteps)
        all_timesteps = Variable(ptu.from_numpy(all_timesteps), requires_grad=False)

        traj_embeddings = self.traj_encoder(all_timesteps)

        r = self.agg(traj_embeddings)
        post_mean, post_log_sig_diag = self.r_to_z_map(r)

        return ReparamMultivariateNormalDiag(post_mean, post_log_sig_diag)

        # c_len = len(context)
        # return ReparamMultivariateNormalDiag(Variable(torch.zeros(c_len, 50), requires_grad=False), Variable(torch.ones(c_len, 50), requires_grad=False))
