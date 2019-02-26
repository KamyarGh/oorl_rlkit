import torch
import torch.optim as optim
from torch import nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_irl_algorithm import TorchIRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.core.train_util import linear_schedule

import numpy as np
from collections import OrderedDict

torch.set_printoptions(profile="full")


def concat_trajs(trajs):
    new_dict = {}
    for k in trajs[0].keys():
        if isinstance(trajs[0][k], dict):
            new_dict[k] = concat_trajs([t[k] for t in trajs])
        else:
            new_dict[k] = np.concatenate([t[k] for t in trajs], axis=0)
    return new_dict


class BC(TorchIRLAlgorithm):
    '''
        Maximum Likelihood Behaviour Cloning
    '''
    def __init__(
        self,
        env,
        policy,
        expert_replay_buffer,

        optim_batch_size=1024,
        policy_lr=1e-3,

        num_update_loops_per_train_call=1000,
        num_policy_updates_per_loop_iter=1,

        policy_mean_reg_weight=0.001,
        policy_std_reg_weight=0.001,

        plotter=None,
        render_eval_paths=False,
        eval_deterministic=True,
        **kwargs
    ):
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        
        super().__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            expert_replay_buffer=expert_replay_buffer,
            **kwargs
        )

        self.optim_batch_size = optim_batch_size
        self.policy_optimizer = optim.Adam(
            self.exploration_policy.parameters(),
            lr=policy_lr,
        )

        self.rewardf_eval_statistics = None
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight


    def get_batch(self, batch_size):
        buffer = self.expert_replay_buffer
        batch = buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch


    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)


    def _do_policy_training(self, epoch):
        exp_batch = self.get_batch(self.optim_batch_size)
        obs = exp_batch['observations']
        acts = exp_batch['actions']
        if self.wrap_absorbing:
            obs = torch.cat([obs, exp_batch['absorbing'][:, 0:1]], dim=-1)

        self.policy_optimizer.zero_grad()
        log_prob, mean, log_std = self.exploration_policy.get_log_prob(obs, acts, return_normal_params=True)
        # print(torch.max(log_prob))
        try:
            # print(torch.mean(log_prob))
            # if doing log prob
            loss = -1.0 * log_prob.mean()
            # if doing MSE
            # loss = ((self.exploration_policy(obs)[0] - acts)**2).mean()
        except Exception as e:
            print(log_prob)
            print(mean)
            print(log_std)
            raise e
        mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
        total_loss = loss + mean_reg_loss + std_reg_loss
        # total_loss = loss
        total_loss.backward()
        self.policy_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.rewardf_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.rewardf_eval_statistics = OrderedDict()
            np_log_prob = ptu.get_numpy(log_prob)
            self.rewardf_eval_statistics['Avg Log Like'] = np.mean(np_log_prob)
            self.rewardf_eval_statistics['Max Log Like'] = np.max(np_log_prob)
            self.rewardf_eval_statistics['Mean Reg Loss'] = np.mean(ptu.get_numpy(mean_reg_loss))
            self.rewardf_eval_statistics['Std Reg Loss'] = np.mean(ptu.get_numpy(std_reg_loss))
            self.rewardf_eval_statistics['Avg Mean'] = np.mean(ptu.get_numpy(mean))
            cov = ptu.get_numpy(torch.exp(2*log_std))
            self.rewardf_eval_statistics['Avg Cov'] = np.mean(cov)
            self.rewardf_eval_statistics['Min Cov'] = np.min(cov)


    @property
    def networks(self):
        return [self.exploration_policy]


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(policy=self.exploration_policy)
        return snapshot


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }
