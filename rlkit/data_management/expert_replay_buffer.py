from collections import defaultdict
from random import sample

import numpy as np

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer

class ExpertReplayBuffer(EnvReplayBuffer):
    '''
        Expert replay buffer for non-meta-learning setup
        Allows for sampling state-action-state or full trajectories from
        expert demonstrations

        You can't put in more data than max_replay_buffer_size
    '''
    def __init__(self, *args, **kwargs):
        super(ExpertReplayBuffer, self).__init__(*args, **kwargs)
        self.traj_starts = []
    

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if self._top != self._max_replay_buffer_size:
            super(ExpertReplayBuffer, self).add_sample(
                observation, action, reward, terminal,
                next_observation, **kwargs
            )
        else:
            print('Not adding the samples to the expert buffer, max buffer size exceeded')


    def terminate_episode(self):
        self._terminals[self._top - 1] = 1.0

        # to avoid adding it twice
        if len(self.traj_starts) > 0:
            if self.traj_starts[-1] != self._top:
                self.traj_starts.append(self._top)
    

    def _advance(self):
        if self._terminals[self._top] == 1.0:
            if self._top + 1 != self._max_replay_buffer_size:
                if self.traj_starts[-1] != self._top + 1:
                    self.traj_starts.append(self._top)
        self._top += 1
        self._size += 1
    

    def sample_expert_random_batch(self, batch_size, return_next_obs, return_rew, return_terminals):
        '''
            Almost a copy of random_batch function
            Always returns obs and actions in dict
        '''
        dict_to_return = {}
        indices = np.random.randint(0, self._size, batch_size)
        if isinstance(self._observations, dict):
            if self.policy_uses_task_params and not self.concat_task_params_to_policy_obs:
                raise NotImplementedError()
            dict_to_return['observations'] = np.concatenate((self._observations['obs'][indices], self._observations['obs_task_params'][indices]), -1)
            if return_next_obs:
                dict_to_return['next_observations'] = np.concatenate((self._next_obs['obs'][indices], self._next_obs['obs_task_params'][indices]), -1)
        else:
            dict_to_return['observations'] = self._observations[indices]
            if return_next_obs:
                dict_to_return['next_observations'] = self._next_obs[indices]

        dict_to_return['actions'] = self._action_dim[indices]
        if return_rew: dict_to_return['rewards'] = self._rewards[indices]
        if return_terminals: dict_to_return['terminals'] = self._terminals[indices]
        
        return dict_to_return
    

    def sample_expert_trajs(self, num_trajs):
        '''
            I don't need this right now
        '''
        raise NotImplementedError()





class MetaExpertReplayBuffer(EnvReplayBuffer):
    '''
        Expert replay buffer for meta-learning setup
        Allows for sampling state-action-state or full trajectories from
        expert demonstrations

        You can't put in more data than max_replay_buffer_size

        The assumption is that task_params are always a flat numpy array
        so that we can
    '''
    def __init__(
                self, replay_buffer_size_per_task_params, *args, **kwargs
        ):
        raise NotImplementedError()
        self.task_params_to_replay_buffer = defaultdict(
            lambda: ExpertReplayBuffer(
                replay_buffer_size_per_task_params, observation_dim, action_dim,
                discrete_action_dim=discrete_action_dim, policy_uses_pixels=policy_uses_pixels
            )
        )
    

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        '''
        This is a slightly hacky assumption:
        Assuming that if you are using this class then you are doing meta-IRL and using
        the DMCS meta env. So you have obs_task_params as part of your observations
        '''
        task_params = observation['obs_task_params']
        task_params = tuple(task_params)
        self.task_params_to_replay_buffer[task_params].add_sample(
            observation, action, reward, terminal,
            next_observation, **kwargs
        )
        self.current_task_param = task_params

    
    def terminate_episode(self):
        '''
        Assuming serial execution
        '''
        self.task_params_to_replay_buffer[self.current_task_param].terminate_episode()
    

    def sample_expert_random_batch(self, num_task_params, batch_size, return_next_obs, return_rew, return_terminals):
        '''
        Return num_task_params x batch_size samples
        '''
        sample_params = sample(self.task_params_to_replay_buffer.keys(), num_task_params)
        batch_list = [
            self.task_params_to_replay_buffer[p].sample_expert_random_batch(
                batch_size, return_next_obs, return_rew, return_terminals
            ) for p in sample_params
        ]
        return batch_list
    
    
    def sample_expert_trajs(self, num_task_params, num_trajs_per_task):
        sample_params = sample(self.task_params_to_replay_buffer.keys(), num_task_params)
        batch_list = [
            self.task_params_to_replay_buffer[p].sample_expert_trajs(num_trajs_per_task) \
            for p in sample_params
        ]
        return batch_list