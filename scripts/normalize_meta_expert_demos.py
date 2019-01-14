import numpy as np
import joblib

# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-zero-few-shot-fetch-traj-gen/final_zero_few_shot_fetch_traj_gen_2018_12_31_22_14_57_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/linear-demos-zero-few-shot-fetch-traj-gen/linear_demos_zero_few_shot_fetch_traj_gen_2019_01_04_18_22_15_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/10K-linear-demos-zero-few-shot-fetch-traj-gen/10K_linear_demos_zero_few_shot_fetch_traj_gen_2019_01_06_02_07_59_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-correct-1K-linear-demos-zero-few-shot-reach-traj-gen/final_correct_1K_linear_demos_zero_few_shot_reach_traj_gen_2019_01_09_20_56_48_0000--s-0/extra_data.pkl'
path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-10K-wrap-absorbing-linear-demos-zero-few-shot-fetch-traj-gen/final_10K_wrap_absorbing_linear_demos_zero_few_shot_fetch_traj_gen_2019_01_13_23_15_41_0000--s-0/extra_data.pkl'
save_path = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/scale_0p9_wrap_absorbing_linear_10K_demos_zero_fetch_traj_gen/extra_data.pkl'

# SCALE = 0.99
SCALE = 0.9

# obs_max = np.array([0.19673975, 0.19944288, 0.20234512, 0.19673975, 0.19944288,
#     0.20234512, 0.28635685, 0.29541265, 0.00469703, 0.28635685,
#     0.29541265, 0.00469703, 1.3, 1.3, 1.3,
#     1.3, 1.3, 1.3, 0.05095022, 0.05092848,
#     0.01019219, 0.01034121])
# obs_min = np.array([-1.94986926e-01, -1.97374503e-01, -3.04622497e-03, -1.94986926e-01,
#     -1.97374503e-01, -3.04622497e-03, -3.00136632e-01, -2.82639213e-01,
#     -2.17494754e-01, -3.00136632e-01, -2.82639213e-01, -2.17494754e-01,
#     -1.3, -1.3, -1.3, -1.3,
#     -1.3, -1.3, 2.55108763e-06, -8.67902630e-08,
#     -9.42624227e-03, -9.39642018e-03])
# acts_max = np.array([0.24999889, 0.2499995 , 0.2499997 , 0.01499927])
# acts_min = np.array([-0.24999355, -0.24999517, -0.24999965, -0.01499985])

# obs_max = np.array([0.22051651, 0.22935722, 0.20480309, 0.22051651, 0.22935722,
#     0.20480309, 0.30151219, 0.29303502, 0.00444365, 0.30151219,
#     0.29303502, 0.00444365, 1.3, 1.3, 1.3,
#     1.3, 1.3, 1.3, 0.05099135, 0.05091496,
#     0.01034575, 0.0103919 ])
# obs_min = np.array([-1.98124936e-01, -2.04234846e-01, -8.51241789e-03, -1.98124936e-01,
#     -2.04234846e-01, -8.51241789e-03, -3.03874692e-01, -3.00712133e-01,
#     -2.30561716e-01, -3.03874692e-01, -3.00712133e-01, -2.30561716e-01,
#     -1.3, -1.3, -1.3, -1.3,
#     -1.3, -1.3,  2.55108763e-06, -8.67902630e-08,
#     -1.20198677e-02, -9.60486720e-03])
# acts_max = np.array([0.3667496 , 0.3676551 , 0.37420813, 0.015])
# acts_min = np.array([-0.27095875, -0.26862562, -0.27479879, -0.015])

obs_max = np.array([0.22691067, 0.24073516, 0.20616085, 0.22691067, 0.24073516,
    0.20616085, 0.30655007, 0.31246556, 0.00573548, 0.30655007,
    0.31246556, 0.00573548, 1.3, 1.3, 1.3,
    1.3, 1.3, 1.3, 0.05101679, 0.05100176,
    0.01049234, 0.01052882])
obs_min = np.array([-2.07510251e-01, -2.21086958e-01, -3.47862349e-03, -2.07510251e-01,
    -2.21086958e-01, -3.47862349e-03, -3.12571681e-01, -3.14835529e-01,
    -2.17068484e-01, -3.12571681e-01, -3.14835529e-01, -2.17068484e-01,
    -1.3, -1.3, -1.3, -1.3,
    -1.3, -1.3,  0.00000000e+00, -8.67902630e-08,
    -1.23168940e-02, -1.09300949e-02])
acts_max = np.array([0.36900074, 0.36956025, 0.37478169, 0.015])
acts_min = np.array([-0.26874253, -0.27001242, -0.27486427, -0.015])

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
