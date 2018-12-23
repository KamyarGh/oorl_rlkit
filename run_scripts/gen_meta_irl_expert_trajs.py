import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.gen_exp_traj_algorithm import ExpertTrajGeneratorAlgorithm

from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy

from rlkit.envs import get_meta_env, get_meta_env_params_iters
from rlkit.scripted_experts import get_scripted_policy
from rlkit.data_management.env_replay_buffer import MetaEnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder

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
    wrap_absorbing=False,
    policy_is_scripted=False,
    render=False,
    check_for_success=False
):
    expert_uses_pixels = expert_policy_specs['policy_uses_pixels']
    expert_uses_task_params = expert_policy_specs['policy_uses_task_params']
    concat_task_params_to_policy_obs = expert_policy_specs['concat_task_params_to_policy_obs']

    for task_params, obs_task_params in task_params_sampler:
        print('Doing Task {}...'.format(task_params))
        meta_env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_id = meta_env.task_identifier

        num_rollouts_completed = 0
        while num_rollouts_completed < num_rollouts_per_task:
            print('\tRollout %d...' % num_rollouts_completed)
            cur_path_builder = PathBuilder()

            observation = meta_env.reset(task_params=task_params, obs_task_params=obs_task_params)
            if policy_is_scripted:
                expert_policy.reset(meta_env)
            terminal = False

            while (not terminal) and len(cur_path_builder) < max_path_length:
                if render: meta_env.render()
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

                if policy_is_scripted:
                    action, agent_info = expert_policy.get_action(agent_obs, meta_env, len(cur_path_builder))
                else:
                    action, agent_info = expert_policy.get_action(agent_obs)
                next_ob, raw_reward, terminal, env_info = (meta_env.step(action))
                if no_terminal: terminal = False
                reward = raw_reward
                terminal = np.array([terminal])
                reward = np.array([reward])

                cur_path_builder.add_all(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    next_observations=next_ob,
                    terminals=terminal,
                    agent_infos=agent_info,
                    env_infos=env_info
                )
                observation = next_ob

            if terminal and wrap_absorbing:
                raise NotImplementedError("I think they used 0 actions for this")
            
            # if necessary check if it was successful
            if check_for_success:
                was_successful = np.sum([e_info['is_success'] for e_info in cur_path_builder['env_infos']]) > 0
                if was_successful:
                    for timestep in range(len(cur_path_builder)):
                        buffer.add_sample(
                            cur_path_builder['observations'][timestep],
                            cur_path_builder['actions'][timestep],
                            cur_path_builder['rewards'][timestep],
                            cur_path_builder['terminals'][timestep],
                            cur_path_builder['next_observations'][timestep],
                            task_id,
                            agent_info=cur_path_builder['agent_infos'][timestep],
                            env_info=cur_path_builder['env_infos'][timestep]
                        )
                    buffer.terminate_episode(task_id)                
                    num_rollouts_completed += 1
                    print('\t\tSuccessful')
                else:
                    print('\t\tNot Successful')
            else:
                for timestep in range(len(cur_path_builder)):
                    buffer.add_sample(
                        cur_path_builder['observations'][timestep],
                        cur_path_builder['actions'][timestep],
                        cur_path_builder['rewards'][timestep],
                        cur_path_builder['terminals'][timestep],
                        cur_path_builder['next_observations'][timestep],
                        task_id,
                        agent_info=cur_path_builder['agent_infos'][timestep],
                        env_info=cur_path_builder['env_infos'][timestep]
                    )
                buffer.terminate_episode(task_id)                
                num_rollouts_completed += 1



def experiment(specs):
    # this is just bad nomenclature: specific_exp_dir is the dir where you will find
    # the specific experiment run (with a particular seed etc.) of the expert policy
    # to use for generating trajectories
    if not specs['use_scripted_policy']:
        with open(path.join(specs['specific_exp_dir'], 'variant.json'), 'r') as f:
            variant = json.load(f)
        max_path_length = variant['algo_params']['max_path_length']
        if 'wrap_absorbing' in variant['algo_params']:
            wrap_absorbing = variant['algo_params']['wrap_absorbing']
        else:
            wrap_absorbing = False
        expert_policy_specs = variant['algo_params']
        no_terminal = variant['algo_params']['no_terminal']
    else:
        max_path_length = specs['max_path_length']
        wrap_absorbing = specs['wrap_absorbing']
        expert_policy_specs = {
            'policy_uses_pixels': specs['policy_uses_pixels'],
            'policy_uses_task_params': specs['policy_uses_task_params'],
            'concat_task_params_to_policy_obs': specs['concat_task_params_to_policy_obs']
        }
        no_terminal = specs['no_terminal']

    # for now the assumption is that the expert was trained with forward rl
    if specs['use_scripted_policy']:
        policy = get_scripted_policy(specs['scripted_policy_name'])
        policy_is_scripted = True
    else:
        policy = joblib.load(path.join(specs['specific_exp_dir'], 'params.pkl'))['exploration_policy']

    # set up the envs
    env_specs = specs['env_specs']
    meta_train_env, meta_test_env = get_meta_env(env_specs)

    # get the task param iterators for the meta envs
    meta_train_params_sampler, meta_test_params_sampler = get_meta_env_params_iters(env_specs)

    # make the replay buffers
    buffer_constructor = lambda env_for_buffer: MetaEnvReplayBuffer(
        max_path_length * specs['num_rollouts_per_task'],
        env_for_buffer,
        policy_uses_pixels=specs['student_policy_uses_pixels'],
        # we don't want the student policy to be looking at true task parameters
        policy_uses_task_params=False,
        concat_task_params_to_policy_obs=False
    )

    train_context_buffer, train_test_buffer = buffer_constructor(meta_train_env), buffer_constructor(meta_train_env)
    test_context_buffer, test_test_buffer = buffer_constructor(meta_test_env), buffer_constructor(meta_test_env)

    render = specs['render']
    check_for_success = specs['check_for_success']
    # fill the train buffers
    fill_buffer(
        train_context_buffer, meta_train_env, policy, expert_policy_specs,
        meta_train_params_sampler, specs['num_rollouts_per_task'], max_path_length,
        no_terminal=no_terminal, wrap_absorbing=wrap_absorbing,
        policy_is_scripted=policy_is_scripted, render=render,
        check_for_success=check_for_success
    )
    fill_buffer(
        train_test_buffer, meta_train_env, policy, expert_policy_specs,
        meta_train_params_sampler, specs['num_rollouts_per_task'], max_path_length,
        no_terminal=no_terminal, wrap_absorbing=wrap_absorbing,
        policy_is_scripted=policy_is_scripted, render=render,
        check_for_success=check_for_success
    )

    # fill the test buffers
    fill_buffer(
        test_context_buffer, meta_train_env, policy, expert_policy_specs,
        meta_test_params_sampler, specs['num_rollouts_per_task'], max_path_length,
        no_terminal=no_terminal, wrap_absorbing=wrap_absorbing,
        policy_is_scripted=policy_is_scripted, render=render,
        check_for_success=check_for_success
    )
    fill_buffer(
        test_test_buffer, meta_train_env, policy, expert_policy_specs,
        meta_test_params_sampler, specs['num_rollouts_per_task'], max_path_length,
        no_terminal=no_terminal, wrap_absorbing=wrap_absorbing,
        policy_is_scripted=policy_is_scripted, render=render,
        check_for_success=check_for_success
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
