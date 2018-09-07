from rlkit.samplers.util import rollout


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_samples, max_path_length,
            concat_env_params_to_obs=False, normalize_env_params=False, env_params_normalizer=None,
            neural_process=None, latent_repr_fn=None, reward_scale=1, animated=False, env_sampler=None
        ):
        self.env = env
        self.env_sampler = env_sampler
        self.policy = policy
        self.max_path_length = max_path_length
        self.max_samples = max_samples
        self.concat_env_params_to_obs = concat_env_params_to_obs
        self.normalize_env_params = normalize_env_params
        self.env_params_normalizer = env_params_normalizer
        assert (
            max_samples >= max_path_length,
            "Need max_samples >= max_path_length"
        )
        self.neural_process = neural_process
        self.latent_repr_fn = latent_repr_fn
        self.reward_scale = reward_scale
        self.animated = animated

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self):
        paths = []
        n_steps_total = 0
        # only animate one rollout
        already_animated_one = False
        while n_steps_total + self.max_path_length <= self.max_samples:
            animate_this = self.animated and not already_animated_one
            if self.env_sampler is not None:
                self.env, _ = self.env_sampler()
            path = rollout(
                self.env, self.policy, max_path_length=self.max_path_length,
                concat_env_params_to_obs=self.concat_env_params_to_obs,
                normalize_env_params=self.normalize_env_params,
                env_params_normalizer=self.env_params_normalizer,
                neural_process=self.neural_process,
                latent_repr_fn=self.latent_repr_fn,
                reward_scale=self.reward_scale,
                animated=animate_this
            )
            paths.append(path)
            n_steps_total += len(path['observations'])
            already_animated_one = True
        return paths
