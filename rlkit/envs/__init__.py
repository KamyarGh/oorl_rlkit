import os
from os import path as osp
from random import randrange

from numpy import array

from rlkit.envs.base_inverted_pendulum import BaseInvertedPendulumEnv
from rlkit.envs.reacher import MetaReacherEnv


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
    }
}


def get_meta_env(env_specs):
    base_env_name = env_specs['base_env_name']
    spec_name = '_'.join(
        map(
            lambda t: t[0]+str(t[1]),
            sorted(env_specs.items())
        )
    )
    spec_name = base_env_name + spec_name
    fname = spec_name + '.xml'
    fpath = osp.join(CUSTOM_ASSETS_DIR, fname)

    # generate a vector of the meta parameters
    # right now only supporting float meta parameters
    meta_params = []
    for k in sorted(env_specs.keys()):
        if k != 'base_env_name':
            v = env_specs[k]
            if isinstance(v, int):
                v = float(v)
            assert isinstance(v, float), 'meta parameter is not a float!'
            meta_params.append(v)
    meta_params = array(meta_params)

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

    return env, spec_name


class EnvSampler():
    def __init__(self, env_specs_list):
        self.envs = {}
        self.env_names = []
        for spec in env_specs_list:
            env, name = get_meta_env(spec)
            self.envs[name] = env
            self.env_names.append(name)
        self.num_envs = len(self.env_names)
    

    def __call__(self, name=''):
        if name == '':
            i = randrange(self.num_envs)
            return self.envs[self.env_names[i]], self.env_names[i]
        return self.envs[name], name
