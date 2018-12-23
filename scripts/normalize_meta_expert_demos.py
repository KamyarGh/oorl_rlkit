import numpy as np
import joblib

path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/few-shot-larger-object-range-expert/few_shot_larger_object_range_expert_2018_12_22_17_09_50_0000--s-0/extra_data.pkl'
save_path = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/normalized_few_shot_larger_object_range_expert/extra_data.pkl'

obs_max = np.array([0.19673975, 0.19944288, 0.20234512, 0.19673975, 0.19944288,
    0.20234512, 0.28635685, 0.29541265, 0.00469703, 0.28635685,
    0.29541265, 0.00469703, 1.27175999, 1.26395128, 1.21729739,
    1.27175999, 1.26395128, 1.21729739, 0.05095022, 0.05092848,
    0.01019219, 0.01034121])
obs_min = np.array([-1.94986926e-01, -1.97374503e-01, -3.04622497e-03, -1.94986926e-01,
    -1.97374503e-01, -3.04622497e-03, -3.00136632e-01, -2.82639213e-01,
    -2.17494754e-01, -3.00136632e-01, -2.82639213e-01, -2.17494754e-01,
    -1.23604834e+00, -1.27612583e+00, -1.23701436e+00, -1.23604834e+00,
    -1.27612583e+00, -1.23701436e+00,  2.55108763e-06, -8.67902630e-08,
    -9.42624227e-03, -9.39642018e-03])
acts_max = np.array([0.24999889, 0.2499995 , 0.2499997 , 0.01499927])
acts_min = np.array([-0.24999355, -0.24999517, -0.24999965, -0.01499985])
SCALE = 0.99

def normalize_obs(observation):
    observation = (observation - obs_min) / (obs_max - obs_min)
    observation *= 2 * SCALE
    observation -= SCALE
    return observation

def normalize_acts(action):
    action = (action - acts_min) / (acts_max - acts_min)
    action *= 2 * SCALE
    action -= SCALE
    return action

d = joblib.load(path_to_extra_data)
for meta_split in ['meta_train', 'meta_test']:
    for sub_split in ['context', 'test']:
        for rb in d[meta_split][sub_split].task_replay_buffers.values():
            rb._observations['obs'] = normalize_obs(rb._observations['obs'])
            rb._next_obs['obs'] = normalize_obs(rb._next_obs['obs'])
            rb._actions = normalize_acts(rb._actions)
joblib.dump(d, save_path, compress=3)
