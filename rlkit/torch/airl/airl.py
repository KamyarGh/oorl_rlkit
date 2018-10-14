from collections import OrderedDict

import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_irl_algorithm import TorchIRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic


class AIRL(TorchIRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            rewardf,

            policy_optimizer,
            expert_replay_buffer,

            rewardf_optim_batch_size=32,
            policy_optim_batch_size=1000,

            rewardf_lr=1e-3,
            rewardf_optimizer_class=optim.Adam,

            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
            **kwargs
    ):
        assert rewardf_lr != 1e-3, 'Just checking that this is being taken from the spec file'
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        super().__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            expert_replay_buffer=expert_replay_buffer,
            policy_optimizer=policy_optimizer,
            **kwargs
        )
        self.policy = policy
        # this is the f function in AIRL
        self.rewardf = rewardf

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.rewardf_optimizer = rewardf_optimizer_class(
            self.rewardf.parameters(),
            lr=rewardf_lr,
        )

        self.expert_replay_buffer = expert_replay_buffer
        self.rewardf_optim_batch_size = rewardf_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.rewardf_eval_statistics = None


    def get_expert_batch(self, batch_size):
        batch = self.expert_replay_buffer.random_batch(batch_size)
        return np_to_pytorch_batch(batch)
    

    def get_policy_batch(self, batch_size):
        batch = self.replay_buffer.random_batch(batch_size)
        return np_to_pytorch_batch(batch)


    def _do_reward_training(self):
        '''
            Train the discriminator
        '''
        expert_batch = self.get_expert_batch(self.rewardf_optim_batch_size)
        expert_obs = expert_batch['observations']
        expert_actions = expert_batch['actions']

        exp_f_for_exp_s_a = self.rewardf(expert_obs, expert_actions)
        policy_log_prob_for_exp_s_a = self.policy.get_log_prob(expert_obs, expert_actions).detach()

        policy_batch = self.get_policy_batch(self.rewardf_optim_batch_size)
        policy_obs = policy_batch['observations']
        policy_actions = policy_batch['actions']

        exp_f_for_policy_s_a = self.rewardf(policy_obs, policy_actions)
        policy_log_prob_for_policy_s_a = self.policy.get_log_prob(policy_obs, policy_actions).detach()

        """
        Discriminator Loss
        """
        exp_s_a_loss = exp_f_for_exp_s_a - log_sum_exp(exp_f_for_exp_s_a + policy_log_prob_for_exp_s_a, dim=1, keepdim=True)
        policy_s_a_loss = policy_log_prob_for_policy_s_a - log_sum_exp(exp_f_for_policy_s_a + policy_log_prob_for_policy_s_a, dim=1, keepdim=True)
        rewardf_loss = -1.0 * (exp_s_a_loss.mean() + policy_s_a_loss.mean())

        """
        Update networks
        """
        self.rewardf_optimizer.zero_grad()
        rewardf_loss.backward()
        self.rewardf_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.rewardf_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.rewardf_eval_statistics = OrderedDict()
            self.rewardf_eval_statistics['Disc Neg LogProb'] = np.mean(ptu.get_numpy(rewardf_loss))


    def _do_policy_training(self):
        policy_batch = self.get_policy_batch(self.policy_optim_batch_size)
        policy_batch['rewards'] = self.rewardf(policy_batch['observations'], policy_batch['actions'])
        self.policy_optimizer.train_step(policy_batch)


    @property
    def networks(self):
        return [self.rewardf] + self.policy_optimizer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(rewardf=self.rewardf)
        snapshot.update(self.policy_optimizer.get_snapshot())
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


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)