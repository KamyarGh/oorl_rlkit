from collections import OrderedDict
from random import sample

import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_meta_irl_algorithm import TorchMetaIRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper
from rlkit.data_management.path_builder import PathBuilder

from gym.spaces import Dict


class NeuralProcessBC(TorchMetaIRLAlgorithm):
    '''
        Meta-BC using a neural process

        assuming the context trajectories all have the same length and flat and everything nice
    '''
    def __init__(
            self,
            env,

            # this is the main policy network that we wrap with
            # PostCondWrapperPolicy for get_exploration policy
            main_policy,

            train_context_expert_replay_buffer,
            train_test_expert_replay_buffer,
            test_context_expert_replay_buffer,
            test_test_expert_replay_buffer,

            np_encoder,

            num_policy_updates_per_epoch=1000,
            num_tasks_used_per_update=4,
            num_context_trajs_for_training=3,
            test_batch_size_per_task=5,

            # for each task, for each context, infer post, for each post sample, generate some eval trajs
            num_tasks_per_eval=10,
            num_diff_context_per_eval_task=2,
            num_context_trajs_for_eval=3,
            num_eval_trajs_per_post_sample=2,

            num_context_trajs_for_exploration=3,

            policy_lr=1e-3,
            policy_optimizer_class=optim.Adam,

            encoder_lr=1e-3,
            encoder_optimizer_class=optim.Adam,

            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            train_context_expert_replay_buffer=train_context_expert_replay_buffer,
            train_test_expert_replay_buffer=train_test_expert_replay_buffer,
            test_context_expert_replay_buffer=test_context_expert_replay_buffer,
            test_test_expert_replay_buffer=test_test_expert_replay_buffer,
            **kwargs
        )

        self.main_policy = main_policy
        self.encoder = np_encoder
        self.rewardf_eval_statistics = None

        self.policy_optimizer = policy_optimizer_class(
            self.main_policy.parameters(),
            lr=policy_lr,
        )
        self.encoder_optimizer = encoder_optimizer_class(
            self.encoder.parameters(),
            lr=encoder_lr,
        )

        self.num_policy_updates_per_epoch = num_policy_updates_per_epoch
        self.num_tasks_used_per_update = num_tasks_used_per_update
        self.num_context_trajs_for_training = num_context_trajs_for_training
        self.test_batch_size_per_task = test_batch_size_per_task

        self.num_tasks_per_eval = num_tasks_per_eval
        self.num_diff_context_per_eval_task = num_diff_context_per_eval_task
        self.num_context_trajs_for_eval = num_context_trajs_for_eval
        self.num_eval_trajs_per_post_sample = num_eval_trajs_per_post_sample

        self.num_context_trajs_for_exploration = num_context_trajs_for_exploration


    def get_exploration_policy(self, task_identifier):
        list_of_trajs = self.train_context_expert_replay_buffer.sample_trajs_from_task(
            task_identifier,
            self.num_context_trajs_for_exploration,
        )
        post_dist = self.encoder([list_of_trajs])
        # z = post_dist.sample()
        z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        return PostCondMLPPolicyWrapper(self.main_policy, z)
    

    def get_eval_policy(self, task_identifier, mode='meta_test'):
        if mode == 'meta_train':
            rb = self.train_context_expert_replay_buffer
        else:
            rb = self.test_context_expert_replay_buffer
        list_of_trajs = rb.sample_trajs_from_task(
            task_identifier,
            self.num_context_trajs_for_eval,
        )
        post_dist = self.encoder([list_of_trajs])
        # z = post_dist.sample()
        z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        return PostCondMLPPolicyWrapper(self.main_policy, z)


    def _do_training(self):
        '''
        '''
        # train the policy
        for i in range(self.num_policy_updates_per_epoch):
            self.encoder_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
            
            # context batch is a list of list of dicts
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.num_context_trajs_for_training,
                num_tasks=self.num_tasks_used_per_update
            )
            post_dist = self.encoder(context_batch)
            # z = post_dist.sample() # N_tasks x Dim
            z = post_dist.mean

            test_train_batch, _ = self.train_context_expert_replay_buffer.sample_random_batch(
                self.test_batch_size_per_task,
                task_identifiers_list=task_identifiers_list
            )
            test_test_batch, _ = self.train_test_expert_replay_buffer.sample_random_batch(
                self.test_batch_size_per_task,
                task_identifiers_list=task_identifiers_list
            )
            # test_batch is a list of dicts: each dict is a random batch from that task
            
            # convert it to a pytorch tensor
            # note that our objective says we should maximize likelihood of
            # BOTH the context_batch and the test_batch
            obs = np.concatenate([d['observations'] for d in test_train_batch]+[d['observations'] for d in test_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            obs = Variable(ptu.from_numpy(obs), requires_grad=False)
            acts = np.concatenate([d['actions'] for d in test_train_batch]+[d['actions'] for d in test_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            acts = Variable(ptu.from_numpy(acts), requires_grad=False)

            # repeat z to have the right size
            z = z.repeat(1, self.test_batch_size_per_task).view(
                self.num_tasks_used_per_update * self.test_batch_size_per_task,
                -1
            )
            z = z.repeat(2, 1)

            # get action predictions
            pred_acts = self.main_policy(torch.cat([obs, z], dim=-1))

            # loss and backprop
            loss = torch.sum((acts - pred_acts)**2, dim=1).mean()
            loss.backward()

            self.encoder_optimizer.step()
            self.policy_optimizer.step()
        
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Regr MSE Loss'] = np.mean(ptu.get_numpy(loss))


    def obtain_eval_samples(self, epoch, mode='meta_train'):
        self.training_mode(False)

        if mode == 'meta_train':
            params_samples = [self.train_task_params_sampler.sample() for _ in range(self.num_tasks_per_eval)]
        else:
            params_samples = [self.test_task_params_sampler.sample() for _ in range(self.num_tasks_per_eval)]
        all_eval_tasks_paths = []
        for task_params, obs_task_params in params_samples:
            cur_eval_task_paths = []
            self.env.reset(task_params=task_params, obs_task_params=obs_task_params)
            task_identifier = self.env.task_identifier

            for _ in range(self.num_diff_context_per_eval_task):
                eval_policy = self.get_eval_policy(task_identifier, mode=mode)

                for _ in range(self.num_eval_trajs_per_post_sample):
                    cur_eval_path_builder = PathBuilder()
                    observation = self.env.reset(task_params=task_params, obs_task_params=obs_task_params)
                    terminal = False

                    while (not terminal) and len(cur_eval_path_builder) < self.max_path_length:
                        if isinstance(self.obs_space, Dict):
                            if self.policy_uses_pixels:
                                agent_obs = observation['pixels']
                            else:
                                agent_obs = observation['obs']
                        else:
                            agent_obs = observation
                        action, agent_info = eval_policy.get_action(agent_obs)
                        
                        next_ob, raw_reward, terminal, env_info = (self.env.step(action))
                        if self.no_terminal:
                            terminal = False
                        
                        reward = raw_reward
                        terminal = np.array([terminal])
                        reward = np.array([reward])
                        cur_eval_path_builder.add_all(
                            observations=observation,
                            actions=action,
                            rewards=reward,
                            next_observations=next_ob,
                            terminals=terminal,
                            agent_infos=agent_info,
                            env_infos=env_info,
                            task_identifiers=task_identifier
                        )
                        observation = next_ob

                    if terminal and self.wrap_absorbing:
                        raise NotImplementedError("I think they used 0 actions for this")
                        cur_eval_path_builder.add_all(
                            observations=next_ob,
                            actions=action,
                            rewards=reward,
                            next_observations=next_ob,
                            terminals=terminal,
                            agent_infos=agent_info,
                            env_infos=env_info,
                            task_identifiers=task_identifier
                        )
                    
                    if len(cur_eval_path_builder) > 0:
                        cur_eval_task_paths.append(
                            cur_eval_path_builder.get_all_stacked()
                        )
            all_eval_tasks_paths.extend(cur_eval_task_paths)
        
        # flatten the list of lists
        return all_eval_tasks_paths
                

    @property
    def networks(self):
        return [
            self.encoder,
            self.main_policy
        ]


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            encoder=self.encoder,
            main_policy=self.main_policy
        )
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
