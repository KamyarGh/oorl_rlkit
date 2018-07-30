import os
from os import path as osp

from rlkit.envs.base_inverted_pendulum import BaseInvertedPendulumEnv


BASE_ASSETS_DIR = osp.join(os.path.dirname(os.path.realpath(__file__)), 'base_assets')
CUSTOM_ASSETS_DIR = osp.join(os.path.dirname(os.path.realpath(__file__)), 'custom_assets')

all_envs = {
    'gravity_gear_inverted_pendulum': {
        'base_xml': 'base_gravity_gear_pendulum.txt',
        'env_class': BaseInvertedPendulumEnv
    }
}

def get_meta_env(env_specs):
    base_env_name = env_specs['base_env_name']
    fname = '_'.join(
        map(
            lambda t: t[0]+str(t[1]),
            sorted(env_specs.items())
        )
    )
    fname = base_env_name + fname + '.xml'
    fpath = osp.join(CUSTOM_ASSETS_DIR, fname)

    try:
        env = all_envs[base_env_name]['env_class'](fpath)
    except:
        # read the base xml string and fill the env_specs values
        with open(osp.join(BASE_ASSETS_DIR, all_envs[base_env_name]['base_xml']), 'r') as f:
            base_xml = f.read()
        env_xml = base_xml.format(**env_specs)
        with open(fpath, 'w') as f:
            f.write(env_xml)
        env = all_envs[base_env_name]['env_class'](fpath)

    return env
