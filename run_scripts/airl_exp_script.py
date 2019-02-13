import numpy as np
from gym.spaces import Dict

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.irl.policy_optimizers.sac import NewSoftActorCritic
from rlkit.torch.irl.airl import AIRL
from rlkit.torch.irl.airl_with_traj_batches import AIRLWithTrajBatches
from rlkit.torch.networks import FlattenMlp, Mlp

from rlkit.envs import get_env

import torch

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
from time import sleep

EXPERT_LISTING_YAML_PATH = '/h/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'

assert False, 'DONT USE THIS FOR FETCH'

def experiment(variant):
    # get the expert data
    with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
        listings = yaml.load(f.read())
    print(listings.keys())
    expert_dir = listings[variant['expert_name']]['exp_dir']
    specific_run = listings[variant['expert_name']]['seed_runs'][variant['expert_seed_run_idx']]
    file_to_load = path.join(expert_dir, specific_run, 'extra_data.pkl')
    expert_replay_buffer = joblib.load(file_to_load)['replay_buffer']
    expert_replay_buffer.policy_uses_task_params = variant['algo_params']['policy_uses_task_params']
    expert_replay_buffer.concat_task_params_to_policy_obs = variant['algo_params']['concat_task_params_to_policy_obs']

    # set up the env
    env_specs = variant['env_specs']
    if env_specs['train_test_env']:
        env, training_env = get_env(env_specs)
    else:
        env, _ = get_env(env_specs)
        training_env, _ = get_env(env_specs)
    env.seed(variant['seed'])
    training_env.seed(variant['seed'])
    print(env.observation_space)

    # compute obs_dim and action_dim
    if isinstance(env.observation_space, Dict):
        if not variant['algo_params']['policy_uses_pixels']:
            obs_dim = int(np.prod(env.observation_space.spaces['obs'].shape))
            if variant['algo_params']['policy_uses_task_params']:
                if variant['algo_params']['concat_task_params_to_policy_obs']:
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

    # set up the policy models
    policy_net_size = variant['policy_net_size']
    hidden_sizes = [policy_net_size] * variant['num_hidden_layers']
    qf1 = FlattenMlp(
        hidden_sizes=hidden_sizes,
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=hidden_sizes,
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=hidden_sizes,
        input_size=obs_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=hidden_sizes,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # set up the discriminator models
    print(disc_model)
    print(disc_model.clamp_magnitude)

    # set up the RL algorithm used to train the policy
    policy_optimizer = NewSoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['policy_params']
    )

    # set up the AIRL algorithm
    if variant['use_traj_batches']:
        airl_class = AIRLWithTrajBatches
    else:
        airl_class = AIRL
    algorithm = airl_class(
        env,
        policy,
        disc_model,
        policy_optimizer,
        expert_replay_buffer,
        training_env=training_env,
        wrap_absorbing=variant['wrap_absorbing_state'],
        **variant['algo_params']
    )
    print(algorithm.use_target_disc)
    print(algorithm.soft_target_disc_tau)
    print(algorithm.exploration_policy)
    print(algorithm.eval_policy)
    print(algorithm.policy_optimizer.policy_optimizer.defaults['lr'])
    print(algorithm.policy_optimizer.qf1_optimizer.defaults['lr'])
    print(algorithm.policy_optimizer.qf2_optimizer.defaults['lr'])
    print(algorithm.policy_optimizer.vf_optimizer.defaults['lr'])
    print(algorithm.disc_optimizer.defaults['lr'])

    # train
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    if exp_specs['use_gpu']:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
