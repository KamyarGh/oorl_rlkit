class ExpertReplayBuffer():
    '''
        Expert replay buffer for non-meta-learning setup
    '''
    def __init__(
                self, max_replay_buffer_size, observation_dim, action_dim,
                discrete_action_dim=False, policy_uses_pixels=False,
        ):
            self._observation_dim = observation_dim
            self._action_dim = action_dim
            self.discrete_action_dim = discrete_action_dim
            self._max_replay_buffer_size = max_replay_buffer_size
            self.policy_uses_pixels = policy_uses_pixels
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





class MetaExpertReplayBuffer():
    def sample_state_action_pairs(self, task_params_batch, num_per_task_param):
        pass
    
    def sample_trajs(self, task_params_batch, num_per_task_param):
        pass
