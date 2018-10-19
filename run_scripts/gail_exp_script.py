import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.irl.policy_optimizers.sac import NewSoftActorCritic
from rlkit.torch.irl.gail import GAIL
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.irl.disc_models.gail_disc import Model as GAILDiscModel

from rlkit.envs import get_env 

import yaml
import argparse
import importlib
import psutil
import os
import argparse
import joblib


def experiment(variant):
    # assert False, "This method sucks because your replay for pendulum is not full and you're just gonna get zeros now"
    # assert False, "Right now this is just a workaround and you have to check that at least the workaround is good"

    # Get the expert trajectories
    expert_replay_buffer = joblib.load(variant['expert_save_dict'])['replay_buffer']
    K = variant['last_k_expert_steps']
    top = expert_replay_buffer._top

    # rewards = expert_replay_buffer._rewards
    # print(top)
    # print(top-K, top)
    # print(sum(expert_replay_buffer._rewards[49000:50000])/50)
    # # print(expert_replay_buffer._rewards[0:1000])

    # print(rewards.shape)
    # rewards = np.sum(expert_replay_buffer._rewards)/250
    # print(rewards)
    # 1/0

    assert top != 0
    assert top >= K, "Your hack is not working now"
    expert_replay_buffer._max_replay_buffer_size = K
    expert_replay_buffer._observations = expert_replay_buffer._observations[top-K:top]
    expert_replay_buffer._next_obs = expert_replay_buffer._next_obs[top-K:top]
    expert_replay_buffer._actions = expert_replay_buffer._actions[top-K:top]
    expert_replay_buffer._rewards = expert_replay_buffer._rewards[top-K:top]
    expert_replay_buffer._terminals = expert_replay_buffer._terminals[top-K:top]
    expert_replay_buffer._top = 0
    expert_replay_buffer._size = K

    # we have to generate the combinations for the env_specs
    env_specs = variant['env_specs']
    env, _ = get_env(env_specs)

    print(env.observation_space)

    if isinstance(env.observation_space, Dict):
        if not variant['algo_params']['policy_uses_pixels']:
            obs_dim = int(np.prod(env.observation_space.spaces['obs'].shape))
        else:
            raise NotImplementedError()
    else:
        obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    policy_net_size = variant['policy_net_size']
    qf1 = FlattenMlp(
        hidden_sizes=[policy_net_size, policy_net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[policy_net_size, policy_net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[policy_net_size, policy_net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=[policy_net_size, policy_net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    disc_model = GAILDiscModel(obs_dim + action_dim)

    policy_optimizer = NewSoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['policy_params']
    )
    algorithm = GAIL(
        env,
        policy,
        disc_model,
        policy_optimizer,
        expert_replay_buffer,
        **variant['gail_params']
    )
    # assert False, "Have not added new sac yet!"
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
    
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
