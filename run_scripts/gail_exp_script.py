import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.irl.policy_optimizers.sac import NewSoftActorCritic
from rlkit.torch.irl.gail import GAIL
from rlkit.torch.irl.gail_with_traj_batches import GAILWithTrajBatches
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.irl.disc_models.gail_disc import Model as GAILDiscModel
from rlkit.torch.irl.disc_models.gail_disc import MlpGAILDisc
from rlkit.torch.irl.disc_models.gail_disc import ResnetDisc
from rlkit.torch.irl.disc_models.gail_disc import SingleColorFetchCustomDisc, SecondVersionSingleColorFetchCustomDisc
from rlkit.envs.wrapped_absorbing_env import WrappedAbsorbingEnv

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
    # assert False, "This method sucks because your replay for pendulum is not full and you're just gonna get zeros now"
    # assert False, "Right now this is just a workaround and you have to check that at least the workaround is good"

    # OLD WAY OF DOING EXPERT REPLAY BUFFERS
    # # Get the expert trajectories
    # expert_replay_buffer = joblib.load(variant['expert_save_dict'])['replay_buffer']
    # K = variant['last_k_expert_steps']
    # top = expert_replay_buffer._top

    # # rewards = expert_replay_buffer._rewards
    # # print(top)
    # # print(top-K, top)
    # # print(sum(expert_replay_buffer._rewards[49000:50000])/50)
    # # # print(expert_replay_buffer._rewards[0:1000])

    # # print(rewards.shape)
    # # rewards = np.sum(expert_replay_buffer._rewards)/250
    # # print(rewards)
    # # 1/0

    # assert top != 0
    # assert top >= K, "Your hack is not working now"
    # expert_replay_buffer._max_replay_buffer_size = K
    # expert_replay_buffer._observations = expert_replay_buffer._observations[top-K:top]
    # expert_replay_buffer._next_obs = expert_replay_buffer._next_obs[top-K:top]
    # expert_replay_buffer._actions = expert_replay_buffer._actions[top-K:top]
    # expert_replay_buffer._rewards = expert_replay_buffer._rewards[top-K:top]
    # expert_replay_buffer._terminals = expert_replay_buffer._terminals[top-K:top]
    # expert_replay_buffer._top = 0
    # expert_replay_buffer._size = K

    # NEW WAY OF DOING EXPERT REPLAY BUFFERS USING ExpertReplayBuffer class
    with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
        listings = yaml.load(f.read())
    print(listings.keys())
    expert_dir = listings[variant['expert_name']]['exp_dir']
    specific_run = listings[variant['expert_name']]['seed_runs'][variant['expert_seed_run_idx']]
    file_to_load = path.join(expert_dir, specific_run, 'extra_data.pkl')
    expert_replay_buffer = joblib.load(file_to_load)['replay_buffer']
    # this script is for the non-meta-learning GAIL
    expert_replay_buffer.policy_uses_task_params = variant['gail_params']['policy_uses_task_params']
    expert_replay_buffer.concat_task_params_to_policy_obs = variant['gail_params']['concat_task_params_to_policy_obs']

    # Now determine how many trajectories you want to use
    if 'num_expert_trajs' in variant: raise NotImplementedError('Not implemented during the transition away from ExpertReplayBuffer')
    # num_trajs_to_use = int(variant['num_expert_trajs'])
    # assert num_trajs_to_use > 0, 'Dude, you need to use expert demonstrations!'
    # idx = expert_replay_buffer.traj_starts[num_trajs_to_use]
    # expert_replay_buffer._size = idx

    # print(expert_replay_buffer._size)
    # print(expert_replay_buffer.traj_starts)
    # print(num_trajs_to_use)
    # print(sum(expert_replay_buffer._rewards[:expert_replay_buffer._size]/(5 * num_trajs_to_use)))
    # ---------------
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
    env.seed(variant['seed'])
    training_env.seed(variant['seed'])

    if variant['wrap_absorbing_state']:
        assert False, 'Not handling train_test_env'
        env = WrappedAbsorbingEnv(env)

    print(env.observation_space)

    if isinstance(env.observation_space, Dict):
        if not variant['gail_params']['policy_uses_pixels']:
            obs_dim = int(np.prod(env.observation_space.spaces['obs'].shape))
            if variant['gail_params']['policy_uses_task_params']:
                if variant['gail_params']['concat_task_params_to_policy_obs']:
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

    # disc_model = GAILDiscModel(
    #     obs_dim + action_dim,
    #     num_layer_blocks=variant['disc_num_blocks'],
    #     hid_dim=variant['disc_hid_dim'],
    #     use_bn=variant['use_bn_in_disc'],
    # )
    # disc_model = MlpGAILDisc(
    #     hidden_sizes=variant['disc_hidden_sizes'],
    #     output_size=1,
    #     input_size=obs_dim+action_dim,
    #     # hidden_activation=torch.nn.functional.relu,
    #     hidden_activation=torch.nn.functional.tanh,
    #     layer_norm=variant['disc_uses_layer_norm']
    #     # output_activation=identity,
    # )
    # disc_model = ResnetDisc(
    #     variant['disc_hidden_size'],
    #     variant['disc_n_layers'],
    #     1,
    #     obs_dim + action_dim,
    #     hidden_activation=torch.nn.functional.tanh
    # )
    
    # if variant['disc_hid_act'] == 'relu':
    #     hid_act = torch.nn.functional.relu
    # elif variant['disc_hid_act'] == 'tanh':
    #     hid_act = torch.nn.functional.tanh
    # else:
    #     raise Exception()
    # disc_model = MlpGAILDisc(
    #     hidden_sizes=variant['disc_hidden_sizes'],
    #     output_size=1,
    #     input_size=obs_dim+action_dim,
    #     hidden_activation=hid_act,
    #     batch_norm=variant['disc_uses_batch_norm'],
    #     clamp_magnitude=variant['disc_clamp_magnitude']
    #     # output_activation=identity,
    # )
    # print(disc_model)
    # print(disc_model.hidden_activation)
    # print(disc_model.clamp_magnitude)

    disc_model = SingleColorFetchCustomDisc(
        clamp_magnitude=variant['disc_clamp_magnitude'],
        state_only=variant['state_only_disc']
    )
    print(disc_model)
    print(disc_model.clamp_magnitude)

    # disc_model = SecondVersionSingleColorFetchCustomDisc(clamp_magnitude=variant['disc_clamp_magnitude'])
    # print(disc_model)
    # print(disc_model.clamp_magnitude)

    policy_optimizer = NewSoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['policy_params']
    )

    if variant['use_traj_batches']:
        gail_class = GAILWithTrajBatches
    else:
        gail_class = GAIL
    algorithm = gail_class(
        env,
        policy,
        disc_model,
        policy_optimizer,
        expert_replay_buffer,
        training_env=training_env,
        wrap_absorbing=variant['wrap_absorbing_state'],
        **variant['gail_params']
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

    if exp_specs['use_gpu']:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
