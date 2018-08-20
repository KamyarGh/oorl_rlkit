import numpy as np
from numpy.random import choice, randint

import gym
from gym.spaces import Discrete, Box
from time import sleep

debug = False

colors = [
    [76, 144, 255],
    [255, 180, 76],
    [18, 173, 10],
    [140, 10, 173],
    [13, 17, 147],
    [209, 14, 192],
]
colors = np.array(colors) / 255.0


# REMEMBER TO ADD TEXTURE TO THE BACKGROUND LATER


def random_free(grid):
    h, w = grid.shape[1], grid.shape[2]
    x, y = randint(0,h-1), randint(0, w-1)

    while any(grid[:, x, y]):
        x, y = randint(0,h-1), randint(0, w-1)
    return x, y


class MetaMaze(gym.Env):
    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.num_object_types = self.env_specs['num_object_types']
        self.num_per_object = self.env_specs['num_per_object']
        self.maze_h = self.env_specs['maze_h']
        self.maze_w = self.env_specs['maze_w']
        self.flat_repr = self.env_specs['flat_repr']
        self.one_hot_repr = self.env_specs['one_hot_repr']
        self.shuffle = self.env_specs['shuffle']
        self.reward_values = np.array(self.env_specs['reward_values'])
        self.scale = self.env_specs['scale']

        if self.flat_repr:
            if self.one_hot_repr:
                s = self.maze_h*self.maze_w*(self.num_object_types + 1)
            else:
                s = self.maze_h*self.maze_w*3*self.scale*self.scale
            self.observation_space = Box(low=0, high=1, shape=(s,), dtype=np.float32)
        else:
            if self.one_hot_repr:
                s = (self.num_object_types+1, self.maze_h, self.maze_w)
            else:
                s = (3, self.maze_h*self.scale, self.maze_w*self.scale)
            self.observation_space = Box(low=0, high=1, shape=s)
        self.action_space = Discrete(4)

        if not self.shuffle:
            np.random.seed(env_specs['seed'])
            self.chosen_colors = colors[choice(colors.shape[0], size=self.num_object_types+1, replace=False)]
            self.chosen_rewards = self.reward_values[choice(self.reward_values.shape[0], size=self.num_object_types, replace=False)]
        
        # print(self.chosen_colors)
        # print(self.chosen_rewards)
        # sleep(1)


    def reset(self):
        if debug: print('--------')
        if self.shuffle:
            self.chosen_colors = colors[choice(colors.shape[0], size=self.num_object_types+1, replace=False)]
            self.chosen_rewards = self.reward_values[choice(self.reward_values.shape[0], size=self.num_object_types, replace=False)]
        
        self._one_hot = np.zeros((self.num_object_types + 1, self.maze_h, self.maze_w))
        if not self.one_hot_repr:
            self.maze = np.zeros((3, self.maze_h, self.maze_w))
            self.bkgd = np.zeros((3, self.maze_h, self.maze_w))

        # add the objects
        for i in range(self.num_object_types):
            c = self.chosen_colors[i]
            for _ in range(self.num_per_object):
                x, y = random_free(self._one_hot)
                self._one_hot[i,x,y] = 1.0
                if not self.one_hot_repr:
                    self.maze[:,x,y] = c

        # add the agent
        x, y = random_free(self._one_hot)
        self.cur_pos = [x,y]
        self._one_hot[-1,x,y] = 1
        if not self.one_hot_repr:
            self.maze[:,x,y] = self.chosen_colors[-1]

        if self.flat_repr:
            if self.one_hot_repr:
                return self._one_hot.flatten()
            return self.maze.flatten()
        else:
            if self.one_hot_repr:
                return self._one_hot
            elif self.scale > 1:
                return np.kron(self.maze, np.ones((1,self.scale,self.scale)))
            return self.maze


    def step(self, action):
        if action == 0:
            dx, dy = 1, 0
        elif action == 1:
            dx, dy = 0, 1
        elif action == 2:
            dx, dy = -1, 0
        elif action == 3:
            dx, dy = 0, -1
        
        if (0 <= self.cur_pos[0] + dx <= self.maze_w-1) and (0 <= self.cur_pos[1] + dy <= self.maze_h-1):
            x = self.cur_pos[0]
            y = self.cur_pos[1]
            self._one_hot[:,x,y] = 0
            if not self.one_hot_repr:
                self.maze[:,x,y] = self.bkgd[:,x,y]

            x += dx
            y += dy

            reward = sum(self._one_hot[:-1,x,y] * self.chosen_rewards)
            self._one_hot[:,x,y] = 0
            self._one_hot[-1,x,y] = 1
            if not self.one_hot_repr:
                self.maze[:,x,y] = self.chosen_colors[-1]

            self.cur_pos = [x, y]
        else:
            reward = 0.
        
        # if reward != 0:
        if debug:
            print(reward)
        
        if self.flat_repr:
            if self.one_hot_repr:
                feats = self._one_hot.flatten()
            else:
                feats = self.maze.flatten()
        else:
            if self.one_hot_repr:
                feats = self._one_hot
            else:
                feats = self.maze
                if self.scale > 1:
                    feats = np.kron(feats, np.ones((1,self.scale,self.scale)))
        return feats, reward, 0., {}


if __name__ == '__main__':
    from scipy.misc import imsave
    from numpy.random import randint

    maze = MetaMaze(4, 3, 10, 10, np.array([1.0, 2.0, -1.0, -2.0]), flat_repr=False, shuffle=True)
    obs = maze.reset()
    obs = np.kron(obs, np.ones((1,10,10)))
    obs = np.transpose(obs, [1,2,0])
    imsave('plots/debug_maze_env/obs_0.png', obs)
    for i in range(1,101):
        obs, r, d, _ = maze.step(randint(4))
        obs = np.kron(obs, np.ones((1,10,10)))
        obs = np.transpose(obs, [1,2,0])
        imsave('plots/debug_maze_env/obs_%d.png' % i, obs)
        print(r, d)

    # print('-----')

    # obs = maze.reset()
    # obs = np.kron(obs, np.ones((1,10,10)))
    # obs = np.transpose(obs, [1,2,0])
    # imsave('plots/debug_maze_env/obs_2_0.png', obs)
    # for i in range(1,21):
    #     obs, r, d, _ = maze.step(randint(4))
    #     obs = np.kron(obs, np.ones((1,10,10)))
    #     obs = np.transpose(obs, [1,2,0])
    #     imsave('plots/debug_maze_env/obs_2_%d.png' % i, obs)
    #     print(r, d)
