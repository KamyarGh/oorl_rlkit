from collections import OrderedDict

import numpy as np
import torch.optim as optim
import torch
from torch import nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_irl_algorithm import TorchIRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic

log_0p5 = np.log(0.5)

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

            use_grad_pen=True,
            grad_pen_weight=10,

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
        # this is the f function in AIRL
        self.rewardf = rewardf

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.rewardf_optimizer = rewardf_optimizer_class(
            self.rewardf.parameters(),
            lr=rewardf_lr,
        )

        self.rewardf_optim_batch_size = rewardf_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.rewardf_eval_statistics = None

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight


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
        self.rewardf_optimizer.zero_grad()

        expert_batch = self.get_expert_batch(self.rewardf_optim_batch_size)
        expert_obs = expert_batch['observations']
        expert_actions = expert_batch['actions']

        f_for_exp_s_a = self.rewardf(expert_obs, expert_actions)
        policy_log_prob_for_exp_s_a = self.exploration_policy.get_log_prob(expert_obs, expert_actions).detach()

        policy_batch = self.get_policy_batch(self.rewardf_optim_batch_size)
        policy_obs = policy_batch['observations']
        policy_actions = policy_batch['actions']
        # policy_obs = expert_obs
        # policy_actions = expert_actions

        f_for_policy_s_a = self.rewardf(policy_obs, policy_actions)
        policy_log_prob_for_policy_s_a = self.exploration_policy.get_log_prob(policy_obs, policy_actions).detach()

        abs_f_for_exp = torch.abs(f_for_exp_s_a.detach()).mean()
        abs_f_for_policy = torch.abs(f_for_policy_s_a.detach()).mean()
        std_f_for_exp = f_for_exp_s_a.detach().std()
        std_f_for_policy = f_for_policy_s_a.detach().std()
        # print(exp_f_for_exp_s_a.mean())

        """
        Discriminator Loss
        """
        exp_s_a_log_prob = f_for_exp_s_a - log_sum_exp(f_for_exp_s_a + policy_log_prob_for_exp_s_a, dim=1, keepdim=True)
        policy_s_a_log_prob = policy_log_prob_for_policy_s_a - log_sum_exp(f_for_policy_s_a + policy_log_prob_for_policy_s_a, dim=1, keepdim=True)
        rewardf_loss = -0.5 * (exp_s_a_log_prob + policy_s_a_log_prob).mean()

        exp_acc = (exp_s_a_log_prob > log_0p5).type(torch.FloatTensor).mean()
        policy_acc = (policy_s_a_log_prob > log_0p5).type(torch.FloatTensor).mean()
        total_acc = (exp_acc + policy_acc)/2.0

        if self.use_grad_pen:
            eps = Variable(torch.rand(self.rewardf_optim_batch_size, 1))
            if ptu.gpu_enabled(): eps = eps.cuda()
            
            interp_obs = eps*expert_obs + (1-eps)*policy_obs
            interp_obs.detach()
            interp_obs.requires_grad = True
            interp_actions = eps*expert_actions + (1-eps)*policy_actions
            interp_actions.detach()
            interp_actions.requires_grad = True

            f_for_interp = self.rewardf(interp_obs, interp_actions)
            policy_log_prob_for_interp = self.exploration_policy.get_log_prob(interp_obs, interp_actions).detach()
            log_prob_for_interp = f_for_interp - log_sum_exp(f_for_interp + policy_log_prob_for_interp, dim=1, keepdim=True)

            gradients = autograd.grad(
                outputs=log_prob_for_interp.sum(),
                inputs=[interp_obs, interp_actions],
                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                create_graph=True, retain_graph=True, only_inputs=True
            )
            total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()

            rewardf_loss = rewardf_loss + gradient_penalty * self.grad_pen_weight

        """
        Update networks
        """
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
            self.rewardf_eval_statistics['Disc Accuracy'] = np.mean(ptu.get_numpy(total_acc))
            self.rewardf_eval_statistics['Disc Expert Acc'] = np.mean(ptu.get_numpy(exp_acc))
            self.rewardf_eval_statistics['Disc Policy Acc'] = np.mean(ptu.get_numpy(policy_acc))
            self.rewardf_eval_statistics['f Abs'] = np.mean(ptu.get_numpy((abs_f_for_exp + abs_f_for_policy)/2.0))
            self.rewardf_eval_statistics['f Abs for Exp'] = np.mean(ptu.get_numpy(abs_f_for_exp))
            self.rewardf_eval_statistics['f Abs for Policy'] = np.mean(ptu.get_numpy(abs_f_for_policy))
            self.rewardf_eval_statistics['f Std for Exp'] = np.mean(ptu.get_numpy(std_f_for_exp))
            self.rewardf_eval_statistics['f Std for Policy'] = np.mean(ptu.get_numpy(std_f_for_policy))


    def _do_policy_training(self):
        policy_batch = self.get_policy_batch(self.policy_optim_batch_size)
        obs = policy_batch['observations']
        actions = policy_batch['actions']
        f_for_s_a = self.rewardf(obs, actions).detach()
        # policy_log_prob_for_s_a = self.exploration_policy.get_log_prob(obs, actions).detach()
        # policy_batch['rewards'] = f_for_s_a - policy_log_prob_for_s_a
        # --------
        policy_batch['rewards'] = f_for_s_a
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