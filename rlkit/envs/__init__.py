import os
from os import path as osp
from random import randrange

from numpy import array
from numpy.random import uniform

import gym
from dm_control import suite

from rlkit.envs.base_inverted_pendulum import BaseInvertedPendulumEnv
from rlkit.envs.reacher import MetaReacherEnv
from rlkit.envs.hopper import MetaHopperEnv
from rlkit.envs.meta_ant import MetaAntEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs.dmcs_wrapper import DmControlWrapper
from rlkit.envs.dmcs_envs.simple_reacher import build_simple_reacher as build_dmcv_simple_reacher
from rlkit.envs.dmcs_envs.simple_meta_reacher import build_simple_meta_reacher

from dm_control.suite.wrappers import pixels

BASE_ASSETS_DIR = osp.join(os.path.dirname(os.path.realpath(__file__)), 'base_assets')
CUSTOM_ASSETS_DIR = osp.join(os.path.dirname(os.path.realpath(__file__)), 'custom_assets')

all_envs = {
    'gravity_gear_inverted_pendulum': {
        'base_xml': 'base_gravity_gear_pendulum.txt',
        'env_class': BaseInvertedPendulumEnv
    },
    'meta_gears_reacher': {
        'base_xml': 'base_reacher.txt',
        'env_class': MetaReacherEnv
    },
    'meta_gears_hopper': {
        'base_xml': 'base_hopper.txt',
        'env_class': MetaHopperEnv
    },
    'meta_gears_ant': {
        'base_xml': 'base_ant.txt',
        'env_class': MetaAntEnv
    },
}

fixed_envs = {
    'ant_v2': lambda: gym.envs.make('Ant-v2'),
    'swimmer_v2': lambda: gym.envs.make('Swimmer-v2'),
    'halfcheetah_v2': lambda: gym.envs.make('HalfCheetah-v2'),
    'hopper_v2': lambda: gym.envs.make('Hopper-v2'),
    'reacher_v2': lambda: gym.envs.make('Reacher-v2'),
    'pendulum_v0': lambda: gym.envs.make('Pendulum-v0'),
    'dmcs_reacher_hard': lambda: DmControlWrapper(suite.load(domain_name='reacher', task_name='hard')),
    'dmcs_reacher_easy': lambda: DmControlWrapper(suite.load(domain_name='reacher', task_name='easy')),
    'dmcs_simple_reacher': lambda: DmControlWrapper(build_dmcv_simple_reacher()),
    # in render_kwargs can specify: height, width, depth, camera_id
    'dmcs_simple_reacher_with_pixels': lambda: DmControlWrapper(
        pixels.Wrapper(
            build_dmcv_simple_reacher(),
            pixels_only=False,
            render_kwargs={'height':80, 'width':80, 'camera_id':0}
        )
    ),
    'dmcs_simple_meta_reacher': lambda: DmControlWrapper(build_simple_meta_reacher()),
    'dmcs_simple_meta_reacher_with_pixels': lambda: DmControlWrapper(
        pixels.Wrapper(
            build_simple_meta_reacher(),
            pixels_only=False,
            render_kwargs={'height':80, 'width':80, 'camera_id':0}
        )
    )
}


train_test_envs = {
    'dmcs_simple_meta_reacher': {
        'train': lambda: DmControlWrapper(build_simple_meta_reacher(train_env=True)),
        'test': lambda: DmControlWrapper(build_simple_meta_reacher(train_env=False))
    }
}


def get_env(env_specs):
    base_env_name = env_specs['base_env_name']
    spec_name = '_'.join(
        map(
            lambda t: t[0]+str(t[1]),
            sorted(env_specs.items())
        )
    )
    spec_name = base_env_name + spec_name
    # spec_names can get too long, this will work almost always :P

    if len(spec_name) > 128: spec_name = spec_name[:128]
    fname = spec_name + '.xml'
    fpath = osp.join(CUSTOM_ASSETS_DIR, fname)

    # generate a vector of the meta parameters
    # right now only supporting float meta parameters
    meta_params = []
    for k in sorted(env_specs.keys()):
        if k not in ['base_env_name', 'normalized']:
            v = env_specs[k]
            if isinstance(v, int):
                v = float(v)
            assert isinstance(v, float), 'meta parameter is not a float!'
            meta_params.append(v)
    meta_params = array(meta_params)

    if env_specs['train_test_env']:
        env_dict = train_test_envs[base_env_name]
        train_env, test_env = env_dict['train'](), env_dict['test']()

        if env_specs['normalized']:
            train_env = NormalizedBoxEnv(train_env)
            test_env = NormalizedBoxEnv(test_env)
        
        return train_env, test_env
    else:
        if base_env_name in fixed_envs:
            env = fixed_envs[base_env_name]()
        else:
            try:
                env = all_envs[base_env_name]['env_class'](fpath, meta_params)
            except:
                # read the base xml string and fill the env_specs values
                with open(osp.join(BASE_ASSETS_DIR, all_envs[base_env_name]['base_xml']), 'r') as f:
                    base_xml = f.read()
                env_xml = base_xml.format(**env_specs)
                with open(fpath, 'w') as f:
                    f.write(env_xml)
                    f.flush()
                env = all_envs[base_env_name]['env_class'](fpath, meta_params)

                # remove the file to avoid getting a million spec files
                try:
                    os.remove(fpath)
                except:
                    pass
            
        if env_specs['normalized']:
            env = NormalizedBoxEnv(env)
            print('\n\nNormalized\n\n')

        return env, spec_name


class EnvSampler():
    def __init__(self, env_specs_list):
        self.envs = {}
        self.env_names = []
        for spec in env_specs_list:
            env, name = get_env(spec)
            self.envs[name] = env
            self.env_names.append(name)
        self.num_envs = len(self.env_names)
    

    def __call__(self, name=''):
        if name == '':
            i = randrange(self.num_envs)
            return self.envs[self.env_names[i]], self.env_names[i]
        return self.envs[name], name


class OnTheFlyEnvSampler():
    def __init__(self, env_specs):
        # any env specs that is a list is considered to be a list
        # containing two floats marking the upper and lower bound
        # of a range to sample uniformly from
        self.env_specs = env_specs


    def gen_random_specs(self):
        new_dict = {}
        for k, v in self.env_specs.items():
            if not isinstance(v, list):
                new_dict[k] = v
            else:
                low, high = v[0], v[1]
                value = uniform(low, high)
                new_dict[k] = value
        return new_dict


    def __call__(self, specs=None):
        if specs is not None:
            env, _ = get_env(specs)
            return env, specs
        specs = self.gen_random_specs()
        env, _ = get_env(specs)
        return env, specs
