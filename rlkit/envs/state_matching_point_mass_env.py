import numpy as np
from collections import OrderedDict
from gym import utils
from gym import spaces
import os

# from rlkit.envs.mujoco_env import MujocoEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv

from rlkit.core.vistools import plot_seaborn_heatmap, plot_scatter


class StateMatchingPointMassEnv():
    def __init__(self):
        # 1 x 1 x 8 x 2
        self.valid_targets = np.array(
            [[[
                [3.0, 0.0],
                [0.0, 3.0],
                [-3.0, 0.0],
                [0.0, -3.0],
            ]]]
        )
        self.cur_pos = np.zeros([2])

        self.max_action_magnitude = 0.2
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype='float32')
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype='float32')

    def seed(self, num):
        pass

    def step(self, a):
        # a = np.clip(a, -1.0, 1.0)
        # a will be between -1.0 to 1.0

        reward = 0.0 # we don't need a reward for what we want to do with this env

        self.cur_pos += self.max_action_magnitude * a
        
        return self.cur_pos.copy(), reward, False, dict(
            xy_pos=self.cur_pos.copy()
        )

    def reset(self):
        # self.cur_pos = np.random.normal(loc=0.0, scale=0.1, size=2)
        # self.cur_pos = np.array([0.0, -3.0]) + np.random.normal(loc=0.0, scale=0.1, size=2)
        # self.cur_pos = np.array([0.0, 3.0]) + np.random.normal(loc=0.0, scale=0.1, size=2)
        self.cur_pos = np.array([0.0, 0.0]) + np.random.normal(loc=0.0, scale=0.1, size=2)

        return self.cur_pos.copy()


    def log_new_ant_multi_statistics(self, paths, epoch, log_dir):
        # turn the xy pos into arrays you can work with
        # N_paths x path_len x 2
        # xy_pos = np.array([[d['xy_pos'] for d in path["env_infos"]] for path in paths])

        # plot using seaborn heatmap
        xy_pos = [np.array([d['xy_pos'] for d in path["env_infos"]]) for path in paths]
        xy_pos = np.array([d['xy_pos'] for path in paths for d in path['env_infos']])
        
        plot_seaborn_heatmap(
            xy_pos[:,0],
            xy_pos[:,1],
            30,
            'Point-Mass Heatmap Epoch %d'%epoch,
            os.path.join(log_dir, 'heatmap_epoch_%d.png'%epoch),
            [[-5,5], [-5,5]]
        )
        plot_scatter(
            xy_pos[:,0],
            xy_pos[:,1],
            30,
            'Point-Mass Scatter Epoch %d'%epoch,
            os.path.join(log_dir, 'scatter_epoch_%d.png'%epoch),
            [[-5,5], [-5,5]]
        )

        return {}

        # xy_pos = [np.array([d['xy_pos'] for d in path["env_infos"]]) for path in paths]
        # max_len = max([a.shape[0] for a in xy_pos])
        # full_xy_pos = np.zeros((len(xy_pos), max_len, 2))
        # for i in range(len(xy_pos)):
        #     full_xy_pos[i][:xy_pos[i].shape[0]] = xy_pos[i]
        # xy_pos = full_xy_pos

        # # N_paths x path_len x 1 x 2
        # xy_pos = np.expand_dims(xy_pos, 2)

        # # compute for each path which target it gets closest to and how close
        # d = np.linalg.norm(xy_pos - self.valid_targets, axis=-1)
        # within_traj_min = np.min(d, axis=1)
        # min_ind = np.argmin(within_traj_min, axis=1)
        # min_val = np.min(within_traj_min, axis=1)

        # return_dict = OrderedDict()
        # for i in range(self.valid_targets.shape[-2]):
        #     return_dict['Target %d Perc'%i] = np.mean(min_ind == i)
        
        # for i in range(self.valid_targets.shape[-2]):
        #     min_dist_for_target_i = min_val[min_ind == i]
        #     if len(min_dist_for_target_i) == 0:
        #         return_dict['Target %d Dist Mean'%i] = np.mean(-1)
        #         return_dict['Target %d Dist Std'%i] = np.std(-1)
        #     else:
        #         return_dict['Target %d Dist Mean'%i] = np.mean(min_dist_for_target_i)
        #         return_dict['Target %d Dist Std'%i] = np.std(min_dist_for_target_i)

        # return return_dict
