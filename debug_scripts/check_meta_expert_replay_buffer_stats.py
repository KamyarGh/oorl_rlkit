import os
import joblib
import numpy as np
from rlkit.core.vistools import plot_histogram

path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/few-shot-larger-object-range-expert/few_shot_larger_object_range_expert_2018_12_22_17_09_50_0000--s-0/extra_data.pkl'
print(path_to_expert_rb)
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/normalized_basic_few_shot_fetch_demos/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-basic-few-shot-fetch-traj-gen/correct_basic_few_shot_fetch_traj_gen_2018_12_19_08_57_56_0000--s-0/extra_data.pkl'
plot_dir = 'plots/expert_demos_stats/basic_few_shot_fetch'

# we should only use the train set to find the scaling factors
meta_train_rb = joblib.load(path_to_expert_rb)['meta_train']
context_rb = meta_train_rb['context']
test_rb = meta_train_rb['test']

obs = [rb._observations['obs'] for rb in context_rb.task_replay_buffers.values()] + [rb._observations['obs'] for rb in test_rb.task_replay_buffers.values()]
obs = np.concatenate(obs, 0)
acts = [rb._actions for rb in context_rb.task_replay_buffers.values()] + [rb._actions for rb in test_rb.task_replay_buffers.values()]
acts = np.concatenate(acts, 0)

print('\n----------\nObs:')
obj2goal_max = np.max(np.concatenate([obs[:,:3], obs[:,3:6]], 0), axis=0)
objrel_max = np.max(np.concatenate([obs[:,6:9], obs[:,9:12]], 0), axis=0)
objcolor_max = np.max(np.concatenate([obs[:,12:15], obs[:,15:18]], 0), axis=0)
gripper_state_max = np.max(obs[:,18:20], axis=0)
gripper_vel_max = np.max(obs[:,20:22], axis=0)

obj2goal_min = np.min(np.concatenate([obs[:,:3], obs[:,3:6]], 0), axis=0)
objrel_min = np.min(np.concatenate([obs[:,6:9], obs[:,9:12]], 0), axis=0)
objcolor_min = np.min(np.concatenate([obs[:,12:15], obs[:,15:18]], 0), axis=0)
gripper_state_min = np.min(obs[:,18:20], axis=0)
gripper_vel_min = np.min(obs[:,20:22], axis=0)

obs_max = np.concatenate(
    [
        obj2goal_max,
        obj2goal_max,
        objrel_max,
        objrel_max,
        objcolor_max,
        objcolor_max,
        gripper_state_max,
        gripper_vel_max
    ],
    axis=-1
)
obs_min = np.concatenate(
    [
        obj2goal_min,
        obj2goal_min,
        objrel_min,
        objrel_min,
        objcolor_min,
        objcolor_min,
        gripper_state_min,
        gripper_vel_min
    ],
    axis=-1
)

print(repr(obs_max))
print(repr(obs_min))

print('\n----------\nActs:')
print(repr(np.max(acts, axis=0)))
print(repr(np.min(acts, axis=0)))

# print('obs')
# print(repr(np.mean(obs, axis=0)))
# print(repr(np.std(obs, axis=0)))
# print(repr(np.max(obs, axis=0)))
# print(repr(np.min(obs, axis=0)))

# print('\nacts')
# print(repr(np.mean(acts, axis=0)))
# print(repr(np.std(acts, axis=0)))
# print(repr(np.max(acts, axis=0)))
# print(repr(np.min(acts, axis=0)))

# SCALE = 0.99
# norm_obs = (obs - np.min(obs, axis=0)) / (np.max(obs, axis=0) - np.min(obs, axis=0))
# norm_obs *= 2 * SCALE
# norm_obs -= SCALE

# norm_acts = (acts - np.min(acts, axis=0)) / (np.max(acts, axis=0) - np.min(acts, axis=0))
# norm_acts *= 2 * SCALE
# norm_acts -= SCALE

# for i in range(obs.shape[1]):
#     plot_histogram(obs[:,i], 100, 'obs dim %d' % i, os.path.join(plot_dir, 'obs_%d.png'%i))
#     plot_histogram(norm_obs[:,i], 100, 'norm obs dim %d' % i, os.path.join(plot_dir, 'norm_obs_%d.png'%i))
# for i in range(acts.shape[1]):
#     plot_histogram(acts[:,i], 100, 'acts dim %d' % i, os.path.join(plot_dir, 'acts_%d.png'%i))
#     plot_histogram(norm_acts[:,i], 100, 'norm acts dim %d' % i, os.path.join(plot_dir, 'norm_acts_%d.png'%i))