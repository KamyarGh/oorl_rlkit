import numpy as np
from collections import OrderedDict
from gym import utils
import os

# from rlkit.envs.mujoco_env import MujocoEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv


class AntMultiTargetEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = os.path.join(os.path.dirname(__file__), "assets", 'low_gear_ratio_ant.xml')
        MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

        # 1 x 1 x 8 x 2
        self.valid_targets = np.array(
            [[[
                [2.0, 0.0],
                [1.41, 1.41],
                [0.0, 2.0],
                [-1.41, 1.41],
                [-2.0, 0.0],
                [-1.41, -1.41],
                [0.0, -2.0],
                [1.41, -1.41]
            ]]]
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")
        reward = 0.0 # we don't need a reward for what we want to do with this env
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            xy_pos=xposafter[:2].copy()
        )

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso").flat
        ])
        return obs.copy()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def reset(self):
        obs = super().reset()
        return obs

    def log_statistics(self, paths):
        # turn the xy pos into arrays you can work with
        # N_paths x path_len x 2
        xy_pos = np.array([[d['xy_pos'] for d in path["env_infos"]] for path in paths])
        # N_paths x path_len x 1 x 2
        xy_pos = np.expand_dims(xy_pos, 2)

        # compute for each path which target it gets closest to and how close
        d = np.linalg.norm(xy_pos - self.valid_targets, axis=-1)
        within_traj_min = np.min(d, axis=1)
        min_ind = np.argmin(within_traj_min, axis=1)
        min_val = np.min(within_traj_min, axis=1)

        return_dict = OrderedDict()
        for i in range(self.valid_targets.shape[-2]):
            return_dict['Target %d Perc'%i] = np.mean(min_ind == i)
        
        for i in range(self.valid_targets.shape[-2]):
            min_dist_for_target_i = min_val[min_ind == i]
            if len(min_dist_for_target_i) == 0:
                return_dict['Target %d Dist Mean'%i] = np.mean(-1)
                return_dict['Target %d Dist Std'%i] = np.std(-1)
            else:
                return_dict['Target %d Dist Mean'%i] = np.mean(min_dist_for_target_i)
                return_dict['Target %d Dist Std'%i] = np.std(min_dist_for_target_i)

        return return_dict
