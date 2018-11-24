import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

from rlkit.core import logger
from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.networks import MlpPolicy
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.irl.policy_optimizers.sac import NewSoftActorCritic
from rlkit.torch.networks import FlattenMlp
from rlkit.envs.wrapped_absorbing_env import WrappedAbsorbingEnv
from rlkit.envs import get_env
from rlkit.torch.torch_rl_algorithm import np_to_pytorch_batch
from rlkit.core.eval_util import get_average_returns
from rlkit.samplers.in_place import InPlacePathSampler

import torch
from torch import optim

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
from time import sleep

EXPERT_LISTING_YAML_PATH = '/u/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'

def experiment(variant):
    # NEW WAY OF DOING EXPERT REPLAY BUFFERS USING ExpertReplayBuffer class
    with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
        listings = yaml.load(f.read())
    print(listings.keys())
    expert_dir = listings[variant['expert_name']]['exp_dir']
    specific_run = listings[variant['expert_name']]['seed_runs'][variant['expert_seed_run_idx']]
    file_to_load = path.join(expert_dir, specific_run, 'extra_data.pkl')
    expert_replay_buffer = joblib.load(file_to_load)['replay_buffer']

    expert_replay_buffer.policy_uses_task_params = variant['bc_params']['policy_uses_task_params']
    expert_replay_buffer.concat_task_params_to_policy_obs = variant['bc_params']['concat_task_params_to_policy_obs']

    # Now determine how many trajectories you want to use
    num_trajs_to_use = int(variant['num_expert_trajs'])
    assert num_trajs_to_use > 0, 'Dude, you need to use expert demonstrations!'
    idx = expert_replay_buffer.traj_starts[num_trajs_to_use]
    expert_replay_buffer._size = idx

    print(expert_replay_buffer._size)
    print(expert_replay_buffer.traj_starts)
    print(num_trajs_to_use)
    print(sum(expert_replay_buffer._rewards[:expert_replay_buffer._size]/(5 * num_trajs_to_use)))

    # Approximately verify that the expert was getting a good reward
    # exp_rew = expert_replay_buffer.subsampling * np.sum(expert_replay_buffer._rewards[:expert_replay_buffer._size]) / num_trajs_to_use
    # exp_rew /= 5
    # print('\n\nExpert is getting approximately %.4f reward.\n\n' % exp_rew)

    # we have to generate the combinations for the env_specs
    env_specs = variant['env_specs']
    if env_specs['train_test_env']:
        env, training_env = get_env(env_specs)
    else:
        env, _ = get_env(env_specs)
        training_env, _ = get_env(env_specs)

    if variant['wrap_absorbing_state']:
        assert False, 'Not handling train_test_env'
        env = WrappedAbsorbingEnv(env)

    print(env.observation_space)

    if isinstance(env.observation_space, Dict):
        if not variant['bc_params']['policy_uses_pixels']:
            obs_dim = int(np.prod(env.observation_space.spaces['obs'].shape))
            if variant['bc_params']['policy_uses_task_params']:
                if variant['bc_params']['concat_task_params_to_policy_obs']:
                    obs_dim += int(np.prod(env.observation_space.spaces['obs_task_params'].shape))
                else:
                    raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    print(obs_dim, action_dim)
    sleep(3)

    policy_net_size = variant['policy_net_size']
    # policy = ReparamTanhMultivariateGaussianPolicy(
    #     hidden_sizes=[policy_net_size, policy_net_size],
    #     obs_dim=obs_dim,
    #     action_dim=action_dim
    # )
    policy = MlpPolicy(
        [policy_net_size, policy_net_size],
        action_dim,
        obs_dim,
        hidden_activation=torch.nn.functional.tanh,
        layer_norm=variant['bc_params']['use_layer_norm']
    )

    policy_optimizer = optim.Adam(
        policy.parameters(),
        lr=variant['bc_params']['lr']
    )

    # assert False, "Have not added new sac yet!"
    if ptu.gpu_enabled():
        policy.cuda()
    
    eval_policy = policy
    # eval_policy = MakeDeterministic(policy)
    eval_sampler = InPlacePathSampler(
        env=env,
        policy=eval_policy,
        max_samples=variant['bc_params']['max_path_length'] + variant['bc_params']['num_steps_per_eval'],
        max_path_length=variant['bc_params']['max_path_length'],
        policy_uses_pixels=variant['bc_params']['policy_uses_pixels'],
        policy_uses_task_params=variant['bc_params']['policy_uses_task_params'],
        concat_task_params_to_policy_obs=variant['bc_params']['concat_task_params_to_policy_obs']
    )

    batch_size = variant['bc_params']['batch_size']
    freq_eval = variant['bc_params']['freq_eval']
    freq_saving = variant['bc_params']['freq_saving']
    for itr in range(variant['bc_params']['max_iters']):
        policy_optimizer.zero_grad()

        batch = expert_replay_buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        obs, acts = batch['observations'], batch['actions']

        # IF POLICY IS AN MLP
        policy_mean = policy.forward(obs)

        # IF POLICY WAS REPARMAT TANH MVN
        # policy_outputs = policy.forward(obs, return_log_prob=True, return_tanh_normal=True)
        # tanh_normal = policy_outputs[-1]
        # policy_mean, policy_log_std = policy_outputs[1:3]
        # log_prob = tanh_normal.log_prob(acts)

        # log_prob = torch.nn.functional.relu(log_prob + 100) - 100

        # try:
        #     log_prob_mean = torch.mean(log_prob)
        # except:
        #     print(log_prob)
        #     print(acts)

        # log_prob_loss = -1.0 * log_prob_mean
        # add regularization
        # mean_reg_loss = variant['bc_params']['policy_mean_reg_weight'] * (policy_mean**2).mean()
        # std_reg_loss = variant['bc_params']['policy_std_reg_weight'] * (policy_log_std**2).mean()
        # pre_activation_reg_loss = variant['bc_params']['policy_pre_activation_weight'] * (
        #     (pre_tanh_value**2).sum(dim=1).mean()
        # )
        # policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss

        # loss = log_prob_loss + policy_reg_loss
        # loss = log_prob_loss + std_reg_loss
        loss = torch.sum((policy_mean - acts)**2, 1).mean()
        # loss = log_prob_loss

        
        loss.backward()
        policy_optimizer.step()

        if itr % freq_eval == 0:
            test_paths = eval_sampler.obtain_samples()
            average_returns = get_average_returns(test_paths)
            print('\nIter {} Returns:\t{}'.format(itr, average_returns))
            print('Iter {} Loss:\t{}'.format(itr, loss.data[0]))
            # print('Iter {} LogProb:\t{}'.format(itr, log_prob_mean.data[0]))

            logger.record_tabular('Test_Returns_Mean', average_returns)

            # print(acts.data[0].numpy())
            # print(policy_mean.data[0].numpy())
            # print(torch.exp(policy_log_std).data[0].numpy())
            # print()

            logger.dump_tabular(with_prefix=False, with_timestamp=False)
                
        # add saving

        
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
