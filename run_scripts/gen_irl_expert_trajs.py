import numpy as np
from random import randint
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

from copy import deepcopy

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.gen_exp_traj_algorithm import ExpertTrajGeneratorAlgorithm

from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy

from rlkit.envs import get_env
from rlkit.scripted_experts import get_scripted_policy
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch.sac.policies import MakeDeterministic

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
import json

from rlkit.envs.ant_multi_target import AntMultiTargetEnv

def fill_buffer(
    buffer,
    env,
    expert_policy,
    expert_policy_specs,
    num_rollouts,
    max_path_length,
    no_terminal=False,
    policy_is_scripted=False,
    render=False,
    check_for_success=False,
    wrap_absorbing=False,
    subsample_factor=1,
):
    expert_uses_pixels = expert_policy_specs['policy_uses_pixels']
    expert_uses_task_params = expert_policy_specs['policy_uses_task_params']
    concat_task_params_to_policy_obs = expert_policy_specs['concat_task_params_to_policy_obs']

    first_complete_list = []

    num_rollouts_completed = 0
    total_rewards = 0.0

    # special handling of AntMultiTargetEnv cause of the weirdness of the situation
    if isinstance(env, AntMultiTargetEnv):
        multi_ant_agg_experts = expert_policy
        expert_policy = None

    while num_rollouts_completed < num_rollouts:
        print('\tRollout %d...' % num_rollouts_completed)
        cur_path_builder = PathBuilder()

        # special handling of AntMultiTargetEnv cause of the weirdness of the situation
        if isinstance(env, AntMultiTargetEnv):
            num_targets = env.valid_targets.shape[2]
            num_rollouts_per_target = int(num_rollouts / num_targets)
            target_idx = int(num_rollouts_completed / num_rollouts_per_target)
            cur_target = env.valid_targets[0,0,target_idx,:]
            print('ANT GOING FOR {}'.format(cur_target))
            expert_policy = multi_ant_agg_experts.get_exploration_policy(cur_target)

            expert_policy.deterministic = True

        observation = env.reset()
        if policy_is_scripted:
            expert_policy.reset(env)
        terminal = False

        subsample_mod = randint(0, subsample_factor-1)
        step_num = 0

        rewards_for_rollout = 0.0

        printed_target_dist = False
        while (not terminal) and step_num < max_path_length:
            if render: env.render()
            if isinstance(env.observation_space, Dict):
                if expert_uses_pixels:
                    agent_obs = observation['pixels']
                else:
                    agent_obs = observation['obs']
            else:
                agent_obs = observation
                if isinstance(env, AntMultiTargetEnv) and env.use_rel_pos_obs:
                    agent_obs = np.concatenate([agent_obs[:-2*env.valid_targets.shape[2]], env.get_body_com("torso").flat])
                    agent_obs = agent_obs.copy()

            if policy_is_scripted:
                action, agent_info = expert_policy.get_action(agent_obs, env, len(cur_path_builder))
            else:
                action, agent_info = expert_policy.get_action(agent_obs)

            next_ob, raw_reward, terminal, env_info = (env.step(action))

            if no_terminal:
                terminal = False
            if wrap_absorbing:
                terminal_array = np.array([False])
            else:
                terminal_array = np.array([terminal])
            
            reward = raw_reward
            reward = np.array([reward])
            rewards_for_rollout += raw_reward

            if isinstance(env, AntMultiTargetEnv):
                if np.linalg.norm(env_info['xy_pos'] - cur_target) < 0.5:
                    # terminal = True
                    if not printed_target_dist:
                        print(step_num)
                        print(env_info['xy_pos'])
                        # print(observation[-8:])
                        print(next_ob[-8:])
                        printed_target_dist = True

            if step_num % subsample_factor == subsample_mod:
                cur_path_builder.add_all(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    next_observations=next_ob,
                    terminals=terminal_array,
                    absorbing=np.array([0.0, 0.0]),
                    agent_infos=agent_info,
                    env_infos=env_info
                )
            observation = next_ob
            step_num += 1

            if terminal:
                print('Terminal')
                print(step_num)
        # print(env_info['xy_pos'])
        # print(step_num)

        if terminal and wrap_absorbing:
            '''
            If we wrap absorbing states, two additional
            transitions must be added: (s_T, s_abs) and
            (s_abs, s_abs). In Disc Actor Critic paper
            they make s_abs be a vector of 0s with last
            dim set to 1. Here we are going to add the following:
            ([next_ob,0], random_action, [next_ob, 1]) and
            ([next_ob,1], random_action, [next_ob, 1])
            This way we can handle varying types of terminal states.
            '''
            # next_ob is the absorbing state
            # for now just sampling random action
            cur_path_builder.add_all(
                observations=next_ob,
                actions=action,
                # the reward doesn't matter
                rewards=0.0,
                next_observations=next_ob,
                terminals=np.array([False]),
                absorbing=np.array([0.0, 1.0]),
                agent_infos=agent_info,
                env_infos=env_info
            )
            cur_path_builder.add_all(
                observations=next_ob,
                actions=action,
                # the reward doesn't matter
                rewards=0.0,
                next_observations=next_ob,
                terminals=np.array([False]),
                absorbing=np.array([1.0, 1.0]),
                agent_infos=agent_info,
                env_infos=env_info
            )
        
        # if necessary check if it was successful
        if check_for_success:
            was_successful = np.sum([e_info['is_success'] for e_info in cur_path_builder['env_infos']]) > 0
            if was_successful:
                print('\t\tSuccessful')
            else:
                print('\t\tNot Successful')
        if (check_for_success and was_successful) or (not check_for_success):
            for timestep in range(len(cur_path_builder)):
                buffer.add_sample(
                    cur_path_builder['observations'][timestep],
                    cur_path_builder['actions'][timestep],
                    cur_path_builder['rewards'][timestep],
                    cur_path_builder['terminals'][timestep],
                    cur_path_builder['next_observations'][timestep],
                    agent_info=cur_path_builder['agent_infos'][timestep],
                    env_info=cur_path_builder['env_infos'][timestep],
                    absorbing=cur_path_builder['absorbing'][timestep]
                )
            buffer.terminate_episode()
            num_rollouts_completed += 1

            total_rewards += rewards_for_rollout
    
    print('Average Return: %f' % (total_rewards / num_rollouts_completed))


def experiment(specs):
    # this is just bad nomenclature: specific_exp_dir is the dir where you will find
    # the specific experiment run (with a particular seed etc.) of the expert policy
    # to use for generating trajectories
    if not specs['use_scripted_policy']:
        if 'ant_multi_target' in specs['env_specs']['base_env_name']:
            # this one is a very weird case so we are handling it on an individual basis
            policy = joblib.load(path.join(specs['expert_dir'], 'extra_data.pkl'))['algorithm']
            policy_is_scripted = False
            # max_path_length = 100
            max_path_length = 75
            # max_path_length = 50
            expert_policy_specs = {
                'max_path_length': max_path_length,
                'policy_uses_pixels': False,
                'policy_uses_task_params': False,
                'concat_task_params_to_policy_obs': False,
                'no_terminal': True
            }
        else:
            alg = joblib.load(path.join(specs['expert_dir'], 'extra_data.pkl'))['algorithm']
            policy = alg.exploration_policy
            policy_is_scripted = False

            max_path_length = alg.max_path_length
            attrs = [
                'max_path_length', 'policy_uses_pixels',
                'policy_uses_task_params', 'concat_task_params_to_policy_obs',
                'no_terminal'
            ]
            expert_policy_specs = {att: getattr(alg, att) for att in attrs}
            expert_policy_specs['wrap_absorbing'] = specs['wrap_absorbing']
    else:
        policy = get_scripted_policy(specs['scripted_policy_name'])
        policy_is_scripted = True

        max_path_length = specs['max_path_length']
        wrap_absorbing = specs['wrap_absorbing']
        expert_policy_specs = {
            'policy_uses_pixels': specs['policy_uses_pixels'],
            'policy_uses_task_params': specs['policy_uses_task_params'],
            'concat_task_params_to_policy_obs': specs['concat_task_params_to_policy_obs']
        }
        if 'no_terminal' in variant['algo_params']:
            no_terminal = variant['algo_params']['no_terminal']
        else:
            no_terminal = False

    # set up the envs
    env_specs = specs['env_specs']
    if env_specs['train_test_env']:
        env, training_env = get_env(env_specs)
    else:
        env, _ = get_env(env_specs)
        training_env, _ = get_env(env_specs)
    env.seed(specs['seed'])

    # make the replay buffers
    if specs['wrap_absorbing']:
        _max_buffer_size = (max_path_length+2) * specs['num_rollouts']        
    else:
        _max_buffer_size = max_path_length * specs['num_rollouts']
    buffer_constructor = lambda env_for_buffer: EnvReplayBuffer(
        _max_buffer_size,
        env_for_buffer,
        policy_uses_pixels=specs['student_policy_uses_pixels'],
        # we don't want the student policy to be looking at true task parameters
        policy_uses_task_params=False,
        concat_task_params_to_policy_obs=False
    )

    train_buffer = buffer_constructor(env)
    test_buffer = buffer_constructor(env)

    render = specs['render']
    check_for_success = specs['check_for_success']

    # fill the train buffer
    fill_buffer(
        train_buffer, env, policy, expert_policy_specs,
        specs['num_rollouts'], max_path_length,
        no_terminal=specs['no_terminal'], wrap_absorbing=specs['wrap_absorbing'],
        policy_is_scripted=policy_is_scripted, render=render,
        check_for_success=check_for_success,
        subsample_factor=specs['subsample_factor']
    )

    # fill the test buffer
    fill_buffer(
        test_buffer, env, policy, expert_policy_specs,
        specs['num_rollouts'], max_path_length,
        no_terminal=specs['no_terminal'], wrap_absorbing=specs['wrap_absorbing'],
        policy_is_scripted=policy_is_scripted, render=render,
        check_for_success=check_for_success,
        subsample_factor=specs['subsample_factor']
    )

    # save the replay buffers
    d = {
        'train': train_buffer,
        'test': test_buffer
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
