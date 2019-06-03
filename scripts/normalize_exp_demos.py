'''
The normalization I'm using here is different than the one for the meta version
'''
import numpy as np
import joblib
import yaml
import os
from os import path as osp

from rlkit.core.vistools import plot_histogram

EXPERT_LISTING_YAML_PATH = '/h/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'

NORMALIZE_OBS = True
NORMALIZE_ACTS = False

def get_normalized(data, size, mean=None, std=None, return_stats=False):
    print('\n\nSIZE IS: %d' % size)
    if mean is None:
        mean = np.mean(data[:size], axis=0, keepdims=True)
    if std is None:
        std = np.std(data[:size], axis=0, keepdims=True)
        # check for a pathology where some axes are constant
        std = np.where(std == 0, np.ones(std.shape), std)
    
    if return_stats:
        return (data - mean) / std, mean, std
    return (data - mean) / std


def do_the_thing(data_path, save_path, plot_obs_histogram=False):
    d = joblib.load(data_path)
    d['obs_mean'] = None
    d['obs_std'] = None
    d['acts_mean'] = None
    d['acts_std'] = None
    if NORMALIZE_OBS:
        print(d['train']._size)
        print(d['train']._top)
        print(np.max(d['train']._observations[:d['train']._size]))
        print(np.min(d['train']._observations[:d['train']._size]))
        print(d['train']._observations.shape)
        if plot_obs_histogram:
            for i in range(d['train']._observations.shape[1]):
                if i % 4 != 0: continue
                print(i)
                plot_histogram(d['train']._observations[:d['train']._size,i], 100, 'obs %d'%i, 'plots/junk_histos/obs_%d.png'%i)
        d['train']._observations, mean, std = get_normalized(d['train']._observations, d['train']._size, return_stats=True)
        d['train']._next_obs = get_normalized(d['train']._next_obs, d['train']._size, mean=mean, std=std)
        d['test']._observations = get_normalized(d['test']._observations, d['test']._size, mean=mean, std=std)
        d['test']._next_obs = get_normalized(d['test']._next_obs, d['test']._size, mean=mean, std=std)
        d['obs_mean'] = mean
        d['obs_std'] = std
        print(np.max(d['train']._observations[:d['train']._size]))
        print(np.min(d['train']._observations[:d['train']._size]))
        print('\nObservations:')
        print('Mean:')
        print(mean)
        print('Std:')
        print(std)

        print('\nPost Normalization Check')
        print('Train')
        print('Obs')
        print(np.mean(d['train']._observations, axis=0))
        print(np.std(d['train']._observations, axis=0))
        print(np.max(d['train']._observations, axis=0))
        print(np.min(d['train']._observations, axis=0))
        print('Next Obs')
        print(np.mean(d['train']._next_obs, axis=0))
        print(np.std(d['train']._next_obs, axis=0))
        print(np.max(d['train']._next_obs, axis=0))
        print(np.min(d['train']._next_obs, axis=0))
        print('Test')
        print('Obs')
        print(np.mean(d['test']._observations, axis=0))
        print(np.std(d['test']._observations, axis=0))
        print(np.max(d['test']._next_obs, axis=0))
        print(np.min(d['test']._next_obs, axis=0))
        print('Next Obs')
        print(np.mean(d['test']._next_obs, axis=0))
        print(np.std(d['test']._next_obs, axis=0))
        print(np.max(d['test']._next_obs, axis=0))
        print(np.min(d['test']._next_obs, axis=0))
    if NORMALIZE_ACTS:
        raise NotImplementedError('Must take into account d[\'train\']._size')
        # d['train']._actions, mean, std = get_normalized(d['train']._actions, return_stats=True)
        # d['test']._actions = get_normalized(d['test']._actions, mean=mean, std=std)
        # d['acts_mean'] = mean
        # d['acts_std'] = std
        # print('\nActions:')
        # print('Mean:')
        # print(mean)
        # print('Std:')
        # print(std)

    print(save_path)
    joblib.dump(d, osp.join(save_path), compress=3)


# data_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-halfcheetah-demos-250-no-subsampling/correct_halfcheetah_demos_250_no_subsampling_2019_02_16_18_01_27_0000--s-0/extra_data.pkl'
# save_path = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/norm_HC_250_demos_no_subsampling'
with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
    listings = yaml.load(f.read())

for i, expert in enumerate([
    # 'ant_multi_valid_target_demos_8_target_8_each_no_sub'
    # 'deterministic_ant_multi_valid_target_demos_8_target_32_each_no_sub_path_len_50'
    # 'ant_multi_4_directions_32_det_demos_per_task_no_sub_path_len_50'
    # 'ant_multi_4_directions_4_distance_32_det_demos_per_task_no_sub_path_len_75'
    # 'rel_pos_obs_ant_multi_4_directions_4_distance_32_det_demos_per_task_no_sub_path_len_75'
    'rel_pos_obs_ant_multi_4_directions_4_distance_32_det_demos_per_task_no_sub_path_terminates_within_0p5_of_target'

    # 'halfcheetah_256_demos_20_subsampling',
    # 'halfcheetah_128_demos_20_subsampling',
    # 'halfcheetah_64_demos_20_subsampling',
    # 'halfcheetah_32_demos_20_subsampling',
    # 'halfcheetah_16_demos_20_subsampling',
    # 'halfcheetah_8_demos_20_subsampling',
    # 'halfcheetah_4_demos_20_subsampling',

    # 'ant_256_demos_20_subsampling',
    # 'ant_128_demos_20_subsampling',
    # 'ant_64_demos_20_subsampling',
    # 'ant_32_demos_20_subsampling',
    # 'ant_16_demos_20_subsampling',
    # 'ant_8_demos_20_subsampling',
    # 'ant_4_demos_20_subsampling',
    # 'humanoid_64_demos_20_subsampling',
    # 'humanoid_128_demos_20_subsampling',
    # 'humanoid_192_demos_20_subsampling',
    # 'humanoid_256_demos_20_subsampling'
  ]):
  data_path = osp.join(listings[expert]['exp_dir'], listings[expert]['seed_runs'][0], 'extra_data.pkl')
  # save_dir = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/norm_'+expert
  save_dir = '/scratch/hdd001/home/kamyar/expert_demos/norm_'+expert
  os.makedirs(save_dir, exist_ok=True)
  save_path = osp.join(save_dir, 'extra_data.pkl')
  do_the_thing(data_path, save_path, i == 0)
