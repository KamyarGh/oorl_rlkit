import numpy as np


def rollout(env, agent, max_path_length=np.inf, animated=False,
    concat_env_params_to_obs=False, normalize_env_params=False, env_params_normalizer=None,
    neural_process=None, latent_repr_fn=None, reward_scale=1, policy_uses_pixels=False,
    policy_uses_task_params=False, concat_task_params_to_policy_obs=False,
    do_not_reset=False, first_obs=None):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :return:
    """
    if neural_process is not None:
        assert reward_scale != 1, 'Are you sure?! This might be a bug!!'
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    if do_not_reset:
        o = first_obs
    else:
        o = env.reset()
    if neural_process is not None:
        posterior_state = neural_process.reset_posterior_state()
        latent_repr = latent_repr_fn(posterior_state)
        o = np.concatenate([latent_repr, o])
        extra_obs_dim = latent_repr.shape[0]
    elif concat_env_params_to_obs:
        params_to_concat = env.env_meta_params
        if normalize_env_params:
            params_to_concat = env_params_normalizer(params_to_concat)
        o = np.concatenate([params_to_concat, o])
    next_o = None
    path_length = 0
    if animated:
        env.render()
    
    def process_obs(obs):
        if policy_uses_pixels:
            if policy_uses_task_params:
                raise NotImplementedError()
            return obs['pixels']
        elif isinstance(obs, dict):
            # if policy_uses_task_params:
            #     if concat_task_params_to_policy_obs:
            #         return np.concatenate((obs['obs'], obs['obs_task_params']), -1)
            #     else:
            #         raise NotImplementedError()
            # return obs['obs']
            return obs
        return obs
    
    o = process_obs(o)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        # a, agent_info = agent.get_action(o['obs'])
        next_o, r, d, env_info = env.step(a)
        next_o = process_obs(next_o)
        if neural_process is not None:
            posterior_state = neural_process.update_posterior_state(
                posterior_state,
                o[extra_obs_dim:],
                a,
                r * reward_scale,
                next_o
            )
            latent_repr = latent_repr_fn(posterior_state)
            next_o = np.concatenate([latent_repr, next_o])
        elif concat_env_params_to_obs:
            params_to_concat = env.env_meta_params
            if normalize_env_params:
                params_to_concat = env_params_normalizer(params_to_concat)
            next_o = np.concatenate([params_to_concat, next_o])
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    if not isinstance(observations[0], dict):
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
    else:
        l = len(observations)
        next_observations = observations[1:l]
        observations = observations[0:l-1]
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
