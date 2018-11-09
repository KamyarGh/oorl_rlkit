import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
            discrete_action_dim=False, policy_uses_pixels=False,
            policy_uses_task_params=False, concat_task_params_to_policy_obs=False
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self.discrete_action_dim = discrete_action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self.policy_uses_pixels = policy_uses_pixels
        self.policy_uses_task_params = policy_uses_task_params
        self.concat_task_params_to_policy_obs = concat_task_params_to_policy_obs
        if isinstance(observation_dim, tuple):
            dims = [d for d in observation_dim]
            dims = [max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims)
            self._next_obs = np.zeros(dims)
        elif isinstance(observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([max_replay_buffer_size] + list(dims))
                else:
                    dims = (max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims)
                self._next_obs[key] = np.zeros(dims)
            
            # DO NOT USE THEM SORTED
            # HOWEVER YOU USE THEM SHOULD MATCH HOWEVER IT IS USED IN RL_ALGORITHM TO PASS TO THE POLICY
            # self.batching_keys = [k for k in sorted(self._observations.keys()) if k not in ['pixels', 'obs_task_params']]
            # if self.policy_uses_pixels:
            #     self.batching_keys.append('pixels')
            # if self.policy_uses_task_params:
            #     self.batching_keys.append('obs_task_params')
        else:
            self._observations = np.zeros((max_replay_buffer_size, observation_dim))
            self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        # if self.discrete_action_dim:
            # action = np.eye(self._action_dim)[action]            
            # action = np.eye(self._action_space.n)[action]
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal

        if isinstance(self._observations, dict):
            for key, obs in observation.items():
                self._observations[key][self._top] = obs
            for key, obs in next_observation.items():
                self._next_obs[key][self._top] = obs
        else:
            self._observations[self._top] = observation
            self._next_obs[self._top] = next_observation
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        if isinstance(self._observations, dict):
            if self.policy_uses_task_params:
                if self.concat_task_params_to_policy_obs:
                    obs_to_return = np.concatenate((self._observations['obs'][indices], self._observations['obs_task_params'][indices]), -1)
                    next_obs_to_return = np.concatenate((self._next_obs['obs'][indices], self._next_obs['obs_task_params'][indices]), -1)
                else:
                    raise NotImplementedError()
            else:
                obs_to_return = self._observations['obs'][indices]
                next_obs_to_return = self._next_obs['obs'][indices]
        else:
            obs_to_return = self._observations[indices]
            next_obs_to_return = self._next_obs[indices]

        return dict(
            observations=obs_to_return,
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=next_obs_to_return,
        )

    def num_steps_can_sample(self):
        return self._size

    def sample_and_remove(self, batch_size):
        assert not isinstance(self._observations, dict), 'not implemented'
        # This function was made for separating out a validation/test set
        # sets the top to the new self._size
        indices = np.random.randint(0, self._size, batch_size)
        samples = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

        self._observations = np.delete(self._observations, indices, 0)
        self._actions = np.delete(self._actions, indices, 0)
        self._rewards = np.delete(self._rewards, indices, 0)
        self._terminals = np.delete(self._terminals, indices, 0)
        self._next_obs = np.delete(self._next_obs, indices, 0)

        self._size -= batch_size
        self._top = self._size % self._max_replay_buffer_size

        return samples

    def set_buffer_from_dict(self, batch_dict):
        assert not isinstance(self._observations, dict), 'not implemented'
        self._max_replay_buffer_size = max(self._max_replay_buffer_size, batch_dict['observations'].shape[0])
        self._observations = batch_dict['observations']
        self._next_obs = batch_dict['next_observations']
        self._actions = batch_dict['actions']
        self._rewards = batch_dict['rewards']
        self._terminals = batch_dict['terminals']
        self._top = batch_dict['observations'].shape[0] % self._max_replay_buffer_size
        self._size = batch_dict['observations'].shape[0]

    def change_max_size_to_cur_size(self):
        assert not isinstance(self._observations, dict), 'not implemented'
        self._max_replay_buffer_size = self._size
        self._observations = self._observations[:self._size]
        self._next_obs = self._next_obs[:self._size]
        self._actions = self._actions[:self._size]
        self._rewards = self._rewards[:self._size]
        self._terminals = self._terminals[:self._size]
        self._top = min(self._top, self._size) % self._size
