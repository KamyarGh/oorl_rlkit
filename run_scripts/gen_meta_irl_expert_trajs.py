import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.gen_exp_traj_algorithm import ExpertTrajGeneratorAlgorithm

from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.policies import MakeDeterministic

from rlkit.envs import get_meta_env, get_meta_env_params_iters
from rlkit.data_management.env_replay_buffer import MetaEnvReplayBuffer

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
import json


def fill_buffer(
    buffer,
    meta_env,
    expert_policy,
    expert_policy_specs,
    task_params_sampler,
    num_rollouts_per_task,
    max_path_length,
    no_terminal=False,
    wrap_absorbing=False
):
    expert_uses_pixels = expert_policy_specs['policy_uses_pixels']
    expert_uses_task_params = expert_policy_specs['policy_uses_task_params']
    concat_task_params_to_policy_obs = expert_policy_specs['concat_task_params_to_policy_obs']

    for task_params, obs_task_params in task_params_sampler:
        print('Doing Task {}...'.format(task_params))
        meta_env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_id = meta_env.task_identifier

        for i in range(num_rollouts_per_task):
            print('\tRollout %d...' % i)
            observation = meta_env.reset(task_params=task_params, obs_task_params=obs_task_params)
            cur_path_len = 0
            terminal = False

            while (not terminal) and cur_path_len < max_path_length:
                if isinstance(meta_env.observation_space, Dict):
                    if expert_uses_pixels:
                        agent_obs = observation['pixels']
                    else:
                        agent_obs = observation['obs']
                else:
                    agent_obs = observation
                if expert_uses_task_params:
                    if concat_task_params_to_policy_obs:
                        agent_obs = np.concatenate((agent_obs, obs_task_params), -1)
                    else:
                        agent_obs = {'obs': agent_obs, 'obs_task_params': obs_task_params}

                action, agent_info = expert_policy.get_action(agent_obs)
                next_ob, raw_reward, terminal, env_info = (meta_env.step(action))    
                if no_terminal: terminal = False
                reward = raw_reward
                terminal = np.array([terminal])
                reward = np.array([reward])

                buffer.add_sample(
                    observation,
                    action,
                    reward,
                    terminal,
                    next_ob,
                    task_id,
                    agent_info=agent_info,
                    env_info=env_info
                )
                observation = next_ob
                cur_path_len += 1

            if terminal and wrap_absorbing:
                raise NotImplementedError("I think they used 0 actions for this")
            buffer.terminate_episode(task_id)


def experiment(specs):
    # this is just bad nomenclature: specific_exp_dir is the dir where you will find
    # the specific experiment run (with a particular seed etc.) of the expert policy
    # to use for generating trajectories
    with open(path.join(specs['specific_exp_dir'], 'variant.json'), 'r') as f:
        variant = json.load(f)

    # for now the assumption is that the expert was trained with forward rl
    policy = joblib.load(path.join(specs['specific_exp_dir'], 'params.pkl'))['exploration_policy']

    # set up the envs
    env_specs = specs['env_specs']
    meta_train_env, meta_test_env = get_meta_env(env_specs)

    # get the task param iterators for the meta envs
    meta_train_params_sampler, meta_test_params_sampler = get_meta_env_params_iters(env_specs)

    # make the replay buffers
    buffer_constructor = lambda env_for_buffer: MetaEnvReplayBuffer(
        variant['algo_params']['max_path_length'] * specs['num_rollouts_per_task'],
        env_for_buffer,
        policy_uses_pixels=specs['student_policy_uses_pixels'],
        # we don't want the student policy to be looking at true task parameters
        policy_uses_task_params=False,
        concat_task_params_to_policy_obs=False
    )

    train_context_buffer, train_test_buffer = buffer_constructor(meta_train_env), buffer_constructor(meta_train_env)
    test_context_buffer, test_test_buffer = buffer_constructor(meta_test_env), buffer_constructor(meta_test_env)

    if 'wrap_absorbing' in variant['algo_params']:
        wrap_absorbing = variant['algo_params']['wrap_absorbing']
    else:
        wrap_absorbing = False
    # fill the train buffers
    fill_buffer(
        train_context_buffer, meta_train_env, policy, variant['algo_params'],
        meta_train_params_sampler, specs['num_rollouts_per_task'], variant['algo_params']['max_path_length'],
        no_terminal=variant['algo_params']['no_terminal'], wrap_absorbing=wrap_absorbing
    )
    fill_buffer(
        train_test_buffer, meta_train_env, policy, variant['algo_params'],
        meta_train_params_sampler, specs['num_rollouts_per_task'], variant['algo_params']['max_path_length'],
        no_terminal=variant['algo_params']['no_terminal'], wrap_absorbing=wrap_absorbing
    )

    # fill the test buffers
    fill_buffer(
        test_context_buffer, meta_train_env, policy, variant['algo_params'],
        meta_test_params_sampler, specs['num_rollouts_per_task'], variant['algo_params']['max_path_length'],
        no_terminal=variant['algo_params']['no_terminal'], wrap_absorbing=wrap_absorbing
    )
    fill_buffer(
        test_test_buffer, meta_train_env, policy, variant['algo_params'],
        meta_test_params_sampler, specs['num_rollouts_per_task'], variant['algo_params']['max_path_length'],
        no_terminal=variant['algo_params']['no_terminal'], wrap_absorbing=wrap_absorbing
    )

    # save the replay buffers
    d = {
        'meta_train': {
            'context': train_context_buffer,
            'test': train_test_buffer
        },
        'meta_test': {
            'context': test_context_buffer,
            'test': test_test_buffer
        }
    }
    logger.save_extra_data(d)

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
