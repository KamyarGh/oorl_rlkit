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
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer

import torch
from torch import optim
from torch.autograd import Variable

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
    if 'num_expert_trajs' in variant: raise NotImplementedError()
    # num_trajs_to_use = int(variant['num_expert_trajs'])
    # assert num_trajs_to_use > 0, 'Dude, you need to use expert demonstrations!'
    # idx = expert_replay_buffer.traj_starts[num_trajs_to_use]
    # expert_replay_buffer._size = idx
  
    # print(expert_replay_buffer._size)
    # print(expert_replay_buffer.traj_starts)
    # print(num_trajs_to_use)
    # print(sum(expert_replay_buffer._rewards[:expert_replay_buffer._size]/(5 * num_trajs_to_use)))
    # ------------------
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

    # The policy for reacher
    # policy = MlpPolicy(
    #     [policy_net_size, policy_net_size],
    #     action_dim,
    #     obs_dim,
    #     hidden_activation=torch.nn.functional.tanh,
    #     layer_norm=variant['bc_params']['use_layer_norm']
    # )
    # The policy for fetch
    hidden_sizes = [policy_net_size] * variant['num_policy_layers']

    # mean_acts = Variable(ptu.from_numpy(np.array([[0.0005593, 0.00024555, 0.10793256, 0.0104]])), requires_grad=False)
    # std_acts = Variable(ptu.from_numpy(np.array([[0.01485482, 0.0138236, 0.4666197, 0.02469494]])), requires_grad=False)
    # class OutputNormalizedPolicy(MlpPolicy):
    #     def forward(self, obs, unnormalized=True):
    #         acts = super().forward(obs)
    #         if unnormalized:
    #             acts = (acts * std_acts) + mean_acts
    #         return acts

    # policy = OutputNormalizedPolicy(
    policy = MlpPolicy(
        hidden_sizes,
        action_dim,
        obs_dim,
        # hidden_activation=torch.nn.functional.relu,
        hidden_activation=torch.nn.functional.tanh,
        output_activation=torch.nn.functional.tanh,
        layer_norm=variant['bc_params']['use_layer_norm']
        # batch_norm=True
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


    # FOR EASY FETCH ENV ----------------------------------------------------------
    # acts_max = Variable(ptu.from_numpy(np.array([0.11622048, 0.11837779, 1., 0.05])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.11406593, -0.11492375, -0.48009082, -0.005])), requires_grad=False)

    # obs_max = np.array([ 1.35211534e+00,  7.59012039e-01,  8.74170327e-01,  1.35216868e+00,
    # 7.59075514e-01,  8.65117304e-01,  9.99349991e-03,  9.97504859e-03,
    # -5.73782252e-04,  5.14756901e-02,  5.14743797e-02,  3.06240725e-03,
    # 1.60782802e-02,  9.09377515e-03,  1.45024249e-03,  1.55772198e-03,
    # 1.27349030e-02,  2.10399698e-02,  3.87118880e-03,  1.10660038e-02,
    # 2.63549517e-03,  3.08370689e-03,  2.64278933e-02,  2.67708565e-02,
    # 2.67707824e-02])
    # obs_min = np.array([ 1.32694457e+00,  7.39177494e-01,  4.25007763e-01,  1.33124808e+00,
    # 7.39111105e-01,  4.24235324e-01, -9.98595942e-03, -9.98935859e-03,
    # -1.10015137e-01,  2.55108763e-06, -8.67902630e-08, -2.71974527e-03,
    # -9.63782682e-03, -4.56146656e-04, -1.68586348e-03, -1.55750811e-03,
    # -7.64317184e-04, -2.08764492e-02, -3.56580593e-03, -1.05306888e-02,
    # -3.47314426e-03, -3.00819907e-03, -1.27082374e-02, -3.65293252e-03,
    # -3.65292508e-03])
    # goal_max = np.array([1.35216868, 0.75907551, 0.87419374])
    # goal_min = np.array([1.33124808, 0.73911111, 0.42423532])
    # observation_max = Variable(ptu.from_numpy(np.concatenate((obs_max, goal_max), axis=-1)), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.concatenate((obs_min, goal_min), axis=-1)), requires_grad=False)

    # SCALE = 0.99
    # -----------------------------------------------------------------------------

    # FOR SUPER EASY FETCH ENV ----------------------------------------------------
    # acts_max = Variable(ptu.from_numpy(np.array([0.24968111, 0.24899998, 0.24999904, 0.01499934])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24993695, -0.24931063, -0.24999953, -0.01499993])), requires_grad=False)
    # observation_max = Variable(ptu.from_numpy(np.array([0.0152033 , 0.01572069, 0.00401832, 0.02023052, 0.03041435,
    #     0.20169743, 0.05092416, 0.05090878, 0.01017929, 0.01013457])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([-1.77039428e-02, -1.64070528e-02, -1.10015137e-01, -2.06485778e-02,
    #     -2.99603855e-02, -3.43990285e-03,  0.00000000e+00, -8.67902630e-08,
    #     -9.50872658e-03, -9.28206220e-03])), requires_grad=False)
    # SCALE = 0.99
    # -----------------------------------------------------------------------------

    # FOR FETCH ENV ----------------------------------------------------
    # observation_max = Variable(ptu.from_numpy(np.array([0.14997844, 0.14999457, 0.0066419 , 0.2896332 , 0.29748688,
    #    0.4510363 , 0.05095725, 0.05090321, 0.01027833, 0.01043796])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([-0.14985769, -0.14991582, -0.11001514, -0.29275747, -0.28962639,
    #    -0.01673591, -0.00056493, -0.00056452, -0.00953662, -0.00964976])), requires_grad=False)
    # acts_max = Variable(ptu.from_numpy(np.array([0.24999679, 0.24999989, 0.24999854, 0.01499987])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24999918, -0.24999491, -0.24998883, -0.01499993])), requires_grad=False)
    # SCALE = 0.99
    # -----------------------------------------------------------------------------

    # FOR EASY IN AIR FETCH ---------------------------------------------------
    # observation_max = Variable(ptu.from_numpy(np.array([ 0.04999746,  0.04979575,  0.00102964,  0.09834792,  0.10275888,
    #     0.2026911 ,  0.05087222,  0.05089798,  0.01014106,  0.01024989])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([ -4.97249838e-02,  -4.99201765e-02,  -1.10015137e-01,
    #     -9.57695575e-02,  -9.56882197e-02,  -2.95093730e-03,
    #      0.00000000e+00,  -8.67902630e-08,  -9.48171330e-03,
    #     -9.57788163e-03])), requires_grad=False)
    # acts_max = Variable(ptu.from_numpy(np.array([ 0.24997477,  0.24999408,  0.24999995,  0.01499998])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24999714, -0.24999004, -0.24999967, -0.01499985])), requires_grad=False)
    # SCALE = 0.99
    # -------------------------------------------------------------------------

    # FOR EASY IN AIR FETCH LARGER RANGE ---------------------------------------------------
    # acts_max = Variable(ptu.from_numpy(np.array([0.24999906, 0.2499996 , 0.24999867, 0.01499948])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24999676, -0.2499984 , -0.24999669, -0.01499992])), requires_grad=False)
    # observation_max = Variable(ptu.from_numpy(np.array([0.14967261, 0.14953164, 0.00056922, 0.28737584, 0.29375757,
    # 0.30215514, 0.05092484, 0.05089244, 0.01006456, 0.01010476])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([-1.49660926e-01, -1.49646858e-01, -1.10015137e-01, -2.82999770e-01,
    # -2.85085491e-01, -4.58114691e-03,  0.00000000e+00, -8.67902630e-08,
    # -9.47718257e-03, -9.47846722e-03])), requires_grad=False)
    # SCALE = 0.99
    # -------------------------------------------------------------------------

    # FOR EASY IN AIR FETCH LARGER Z RANGE ------------------------------------
    # acts_max = Variable(ptu.from_numpy(np.array([0.24995736, 0.2499716 , 0.24999983, 0.01499852])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24989959, -0.24995068, -0.2499989 , -0.01499998])), requires_grad=False)
    # observation_max = Variable(ptu.from_numpy(np.array([0.0499439 , 0.04998455, 0.00098634, 0.09421162, 0.10457129,
    #    0.3022664 , 0.05094975, 0.05090175, 0.01024486, 0.01029508])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([-4.98090099e-02, -4.97771561e-02, -1.10015137e-01, -9.60775777e-02,
    #    -1.03508767e-01, -3.50153560e-03,  0.00000000e+00, -8.67902630e-08,
    #    -9.47353981e-03, -9.62584145e-03])), requires_grad=False)
    # SCALE = 0.99
    # -------------------------------------------------------------------------

    # FOR EASY IN AIR FETCH LARGER X-Y RANGE ------------------------------------
    acts_max = Variable(ptu.from_numpy(np.array([0.24999749, 0.2499975 , 0.2499998 , 0.01499951])), requires_grad=False)
    acts_min = Variable(ptu.from_numpy(np.array([-0.24999754, -0.24999917, -0.24999704, -0.01499989])), requires_grad=False)
    observation_max = Variable(ptu.from_numpy(np.array([0.14953716, 0.14865454, 0.00155898, 0.28595684, 0.27644423,
    0.20200016, 0.05094223, 0.05082468, 0.01033346, 0.0103368 ])), requires_grad=False)
    observation_min = Variable(ptu.from_numpy(np.array([-1.49931348e-01, -1.49895902e-01, -1.10015137e-01, -2.80037372e-01,
    -2.82756899e-01, -3.44387360e-03,  0.00000000e+00, -8.67902630e-08,
    -9.53356933e-03, -9.71619128e-03])), requires_grad=False)
    SCALE = 0.99
    # -------------------------------------------------------------------------


    def normalize_obs(observation):
        observation = (observation - observation_min) / (observation_max - observation_min)
        observation *= 2 * SCALE
        observation -= SCALE
        return observation
    
    def normalize_acts(action):
        action = (action - acts_min) / (acts_max - acts_min)
        action *= 2 * SCALE
        action -= SCALE
        return action

    print('RUNNING')

    batch_size = variant['bc_params']['batch_size']
    freq_eval = variant['bc_params']['freq_eval']
    freq_saving = variant['bc_params']['freq_saving']
    for itr in range(variant['bc_params']['max_iters']):
        policy_optimizer.zero_grad()

        batch = expert_replay_buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        obs, acts = batch['observations'], batch['actions']

        obs = normalize_obs(obs)
        acts = normalize_acts(acts)

        # acts = (acts - mean_acts) / std_acts

        # IF POLICY IS AN MLP
        # policy_mean = policy.forward(obs, unnormalized=False)
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
            policy.eval()
            test_paths = eval_sampler.obtain_samples()
            average_returns = get_average_returns(test_paths)
            print('\nIter {} Returns:\t{}'.format(itr, average_returns))
            print('Iter {} Loss:\t{}'.format(itr, loss.data[0]))
            # print('Iter {} LogProb:\t{}'.format(itr, log_prob_mean.data[0]))

            logger.record_tabular('Test_Returns_Mean', average_returns)

            if variant['env_specs']['base_env_name'] in ['fetch_pick_and_place', 'debug_fetch_reacher', 'debug_fetch_reach_and_lift', 'easy_fetch_pick_and_place', 'scaled_easy_fetch_pick_and_place', 'scaled_super_easy_fetch_pick_and_place', 'scaled_and_wrapped_fetch', 'scaled_and_wrapped_target_in_air_easy']:
                solved = [sum(path["rewards"])[0] > -1.0 * path["rewards"].shape[0] for path in test_paths]
                print(solved)
                print([sum(path["rewards"])[0] for path in test_paths])
                percent_solved = sum(solved) / float(len(solved))
                logger.record_tabular('Percent_Solved', percent_solved)

            # print(acts.data[0].numpy())
            # print(policy_mean.data[0].numpy())
            # print(torch.exp(policy_log_std).data[0].numpy())
            # print()

            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            policy.train()
                
        # add saving
        # if itr % freq_saving == 0:
        #     logger.save_itr_params(itr, policy)
        
    return 1


if __name__ == '__main__':
    # Arguments
    print('RUNNING')

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
    if exp_specs['use_gpu']:
        ptu.set_gpu_mode(True)

    print('RUNNING')
    experiment(exp_specs)
