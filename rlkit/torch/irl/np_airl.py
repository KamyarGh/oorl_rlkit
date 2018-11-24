from collections import OrderedDict

import numpy as np

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


class NeuralProcessAIRL(TorchIRLAlgorithm):
    '''
        Meta-AIRL
        Neural Process + AIRL/DAC

        assuming the context trajectories all have the same length and flat and everything nice
        context batch: N x L x (S+A)
        test batch: N x K x(S+A)
    '''
    def __init__(
            self,
            env,

            policy,
            encoder,
            discriminator,

            policy_optimizer,
            expert_replay_buffer,

            num_context_trajs=3,
            test_batch_size_per_task=5,
            num_tasks_used_per_update=4,

            disc_optim_batch_size=32,
            policy_optim_batch_size=1000,

            disc_lr=1e-3,
            disc_optimizer_class=optim.Adam,

            encoder_lr=1e-3,
            encoder_optimizer_class=optim.Adam,

            use_grad_pen=True,
            grad_pen_weight=10,

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
            policy_optimizer=policy_optimizer,
            **kwargs
        )

        self.encoder = encoder
        self.discriminator = discriminator
        self.rewardf_eval_statistics = None
        
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
        )
        self.encoder_optimizer = encoder_optimizer_class(
            self.encoder.parameters(),
            lr=encoder_lr,
        )

        self.disc_optim_batch_size = disc_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.num_tasks_used_per_update = num_tasks_used_per_update
        self.num_context_trajs = num_context_trajs
        self.test_batch_size_per_task = test_batch_size_per_task

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(self.disc_optim_batch_size, 1),
                torch.zeros(self.disc_optim_batch_size, 1)
            ],
            dim=0
        )
        self.bce_targets = Variable(self.bce_targets)
        if ptu.gpu_enabled():
            self.bce.cuda()
            self.bce_targets = self.bce_targets.cuda()
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight


    def get_expert_trajs(self, num_tasks, num_trajs_per_task, train_mode=True, task_params=None):
        '''
            train_mode=True means samples from train set of expert trajectories
            otherwise sample from test set of expert trajectories
        '''
        batch, task_params = self.expert_replay_buffer.sample_expert_trajs(
            num_tasks,
            num_trajs_per_task,
            train_mode=train_mode,
            task_params=task_params
        )
        if self.wrap_absorbing:
            raise NotImplementedError()
        return np_to_pytorch_batch(batch), task_params
    

    def get_expert_batch(self, num_tasks, batch_size_per_task, train_mode=True, task_params=None):
        '''
            train_mode=True means samples from train set of expert trajectories
            otherwise sample from test set of expert trajectories
        '''
        batch, task_params = self.expert_replay_buffer.sample_expert_random_batch(
            num_tasks,
            batch_size_per_task,
            train_mode=train_mode,
            task_params=task_params
        )
        if self.wrap_absorbing:
            raise NotImplementedError()
        return np_to_pytorch_batch(batch), task_params
    

    def get_policy_batch(self, batch_size):
        batch = self.replay_buffer.random_batch(batch_size)
        return np_to_pytorch_batch(batch)


    def _do_training(self):
        '''
        '''
        # train the disc
        # here we have to make a choice of how/when we update the encoder
        for i in range(self.num_disc_updates):
            context_batch, task_params = self.get_expert_trajs(
                self.num_tasks_used_per_update,
                self.num_context_trajs,
                train_mode=True
            )
            test_batch = self.get_expert_batch(
                self.num_tasks_used_per_update,
                self.test_batch_size_per_task,
                train_mode=True,
                task_params=task_params
            )
            policy_batch = self.get_policy_batch(
                self.num_tasks_used_per_update,
                self.test_batch_size_per_task,
                task_params=task_params
            )
            
            if i % self.freq_update_encoder != 0:
                post_dist = self.encoder(context_batch)
                z = post_dist.sample()
                z = z.detach()
            else:
                self.encoder_optimizer.zero_grad()
                post_dist = self.encoder(context_batch)
                z = post_dist.sample()

            self._update_disc(test_batch, policy_batch)
            if i % self.freq_update_encoder == 0:
                # in _update_disc the necessary gradients will have been computed
                self.encoder_optimizer.step()
        
        # train the policy
        for i in range(self.num_policy_updates):
            context_batch, task_params = self.get_expert_trajs(
                self.num_tasks_used_per_update,
                self.num_context_trajs,
                train_mode=True
            )
            policy_batch = self.get_policy_batch(
                self.num_tasks_used_per_update,
                self.test_batch_size_per_task,
                task_params=task_params
            )

            post_dist = self.encoder(context_batch).detach()
            z = post_dist.sample()
            _update_policy(policy_batch, z)

    
    def _update_disc(self, test_batch, policy_batch, z)
            self.disc_optimizer.zero_grad()

            test_obs = test_batch['observations']
            test_actions = test_batch['actions']
            test_obs = test_obs.view(-1, test_obs.size(-1))
            test_actions = test_actions.view(-1, test_actions.size(-1))

            policy_obs = policy_batch['observations']
            policy_actions = policy_batch['actions']
            policy_obs = policy_obs.view(-1, policy_obs.size(-1))
            policy_actions = policy_actions.view(-1, policy_actions.size(-1))

            obs = torch.cat([test_obs, policy_obs], dim=0)
            actions = torch.cat([test_actions, policy_actions], dim=0)

            disc_logits = self.discriminator(obs, actions, z)
            disc_preds = (disc_logits > 0).type(torch.FloatTensor)
            disc_loss = self.bce(disc_logits, self.bce_targets)
            accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

            if self.use_grad_pen:
                eps = Variable(torch.rand(self.disc_optim_batch_size, 1))
                if ptu.gpu_enabled(): eps = eps.cuda()
                
                interp_obs = eps*test_obs + (1-eps)*policy_obs
                interp_obs.detach()
                interp_obs.requires_grad = True
                interp_actions = eps*test_actions + (1-eps)*policy_actions
                interp_actions.detach()
                interp_actions.requires_grad = True
                gradients = autograd.grad(
                    outputs=self.discriminator(interp_obs, interp_actions, z).sum(),
                    inputs=[interp_obs, interp_actions],
                    # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True
                )
                total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
                gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()

                disc_loss = disc_loss + gradient_penalty * self.grad_pen_weight

            disc_loss.backward()
            self.disc_optimizer.step()

            """
            Save some statistics for eval
            """
            if self.rewardf_eval_statistics is None:
                """
                Eval should set this to None.
                This way, these statistics are only computed for one batch.
                """
                self.rewardf_eval_statistics = OrderedDict()
                self.rewardf_eval_statistics['Disc Loss'] = np.mean(ptu.get_numpy(disc_loss))
                self.rewardf_eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))
    

    def _update_policy(self, policy_batch, z):
        # If you compute log(D) - log(1-D) then you just get the logits
        policy_batch['rewards'] = self.discriminator(policy_batch['observations'], policy_batch['actions'], z)
        self.policy_optimizer.train_step(policy_batch, z)


    @property
    def networks(self):
        return [self.discriminator, self.encoder] + self.policy_optimizer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            disc=self.discriminator,
            encoder=self.encoder
        )
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
