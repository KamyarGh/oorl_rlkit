import os
from os import path as osp
from random import randrange
import abc

from numpy import array
from numpy.random import uniform

import gym
# from dm_control import suite

from rlkit.envs.base_inverted_pendulum import BaseInvertedPendulumEnv
from rlkit.envs.reacher import MetaReacherEnv
from rlkit.envs.hopper import MetaHopperEnv
from rlkit.envs.meta_ant import MetaAntEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
# from rlkit.envs.dmcs_wrapper import DmControlWrapper
# from rlkit.envs.dmcs_envs.simple_reacher import build_simple_reacher as build_dmcv_simple_reacher
# from rlkit.envs.dmcs_envs.simple_meta_reacher import build_simple_meta_reacher
from rlkit.envs.picker import PickerEnv

# fetch env
from rlkit.envs.wrapped_goal_envs import WrappedFetchPickAndPlaceEnv, DebugReachFetchPickAndPlaceEnv, DebugFetchReachAndLiftEnv
from rlkit.envs.wrapped_goal_envs import WrappedEasyFetchPickAndPlaceEnv
# from rlkit.envs.wrapped_goal_envs import WrappedRotatedFetchReachAnywhereEnv, WrappedEasyFetchPickAndPlaceEnv
from rlkit.envs.wrapped_goal_envs import ScaledWrappedEasyFetchPickAndPlaceEnv, ScaledWrappedSuperEasyFetchPickAndPlaceEnv
from rlkit.envs.wrapped_goal_envs import ScaledWrappedFetchPickAndPlaceEnv, ScaledWrappedTargetOnlyInAirFetchPickAndPlaceEnv

from rlkit.envs.few_shot_fetch_env import BasicFewShotFetchEnv, ScaledBasicFewShotFetchEnv, ZeroScaledFewShotFetchEnv, ZeroUnscaledFewShotFetchEnv
from rlkit.envs.few_shot_fetch_env import ZeroScaled0p9FewShotFetchEnv, ZeroScaled0p9LinearFewShotFetchEnv
from rlkit.envs.few_shot_fetch_env import StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv
from rlkit.envs.few_shot_fetch_env import WrapAbsZeroScaled0p9LinearFewShotFetchEnv
from rlkit.envs.few_shot_fetch_env import get_task_params_iterator as few_shot_fetch_env_get_task_params_iterator
from rlkit.envs.few_shot_fetch_env import get_some_task_params_iterator as few_shot_fetch_env_get_some_task_params_iterator
from rlkit.envs.few_shot_fetch_env import get_debug_task_params_iterator as debug_few_shot_fetch_env_get_task_params_iterator
from rlkit.envs.few_shot_fetch_env import get_zero_task_params_iterator as zero_few_shot_fetch_env_get_task_params_iterator
from rlkit.envs.few_shot_fetch_env import _BaseParamsSampler as FewShotFetchBaseParamsSampler

from rlkit.envs.few_shot_reach_env import BasicFewShotReachEnv, ZeroScaled0p9LinearFewShotReachEnv
from rlkit.envs.few_shot_reach_env import get_zero_task_params_iterator as zero_few_shot_reach_env_get_task_params_iterator

# Halfcheetah rand vel imports
from rlkit.envs.halfcheetah_rand_vel import HalfCheetahRandVelEnv
from rlkit.envs.halfcheetah_rand_vel import _TrainParamsSampler as HalfCheetahRandVelTrainSampler
from rlkit.envs.halfcheetah_rand_vel import _TestParamsSampler as HalfCheetahRandVelTestSampler
from rlkit.envs.halfcheetah_rand_vel import _MetaTrainParamsSampler as HalfCheetahRandVelMetaTrainSampler
from rlkit.envs.halfcheetah_rand_vel import _MetaTestParamsSampler as HalfCheetahRandVelMetaTestSampler
from rlkit.envs.halfcheetah_rand_vel import _DebugMetaTrainSampler as HalfCheetahRandVelDebugMetaTrainSampler
from rlkit.envs.halfcheetah_rand_vel import _DebugMetaTestSampler as HalfCheetahRandVelDebugMetaTestSampler

# Ant goal pos imports
from rlkit.envs.ant_rand_goal import AntRandGoalEnv
from rlkit.envs.ant_rand_goal import _ExpertTrainParamsSampler as AntRandGoalExpertTrainSampler
from rlkit.envs.ant_rand_goal import _ExpertTestParamsSampler as AntRandGoalExpertTestSampler
from rlkit.envs.ant_rand_goal import _Expert120DegreesParamsSampler as AntRandGoal120DegreesSampler
from rlkit.envs.ant_rand_goal import _Expert60DegreesParamsSampler as AntRandGoal60DegreesSampler
from rlkit.envs.ant_rand_goal import _Expert2DirectionsParamsSampler as AntRandGoal2DirectionsSampler
from rlkit.envs.ant_rand_goal import _ExpertOneDirectionParamsSampler as AntRandGoalOneDirectionSampler
from rlkit.envs.ant_rand_goal import _ExpertOpposite2DirectionsParamsSampler as AntRandGoalOpposite2DirectionsSampler
from rlkit.envs.ant_rand_goal import _Expert45DegFartherParamsSampler as AntRandGoal45DegFartherParamsSampler
from rlkit.envs.ant_rand_goal import _ExpertMiddle60DegreesParamsSampler as AntRandGoalMiddle60ParamsSampler
from rlkit.envs.ant_rand_goal import _ExpertLineParamsSampler as AntRandGoalLineParamsSampler
from rlkit.envs.ant_rand_goal import _Expert90DegApartParamsSampler as AntRandGoal90DegApart
from rlkit.envs.ant_rand_goal import _ExpertFivePointsParamsSampler as AntRandGoalFivePoints
from rlkit.envs.ant_rand_goal import _ExpertTwoPointsParamsSampler as AntRandGoalTwoPoints
from rlkit.envs.ant_rand_goal import _ExpertEightPointsParamsSampler as AntRandGoalEightPoints
from rlkit.envs.ant_rand_goal import _Expert24PointsParamsSampler as AntRandGoal24Points
from rlkit.envs.ant_rand_goal import _Expert16PointsTrainParamsSampler as AntRandGoal16PointsTrain
from rlkit.envs.ant_rand_goal import _Expert16PointsTestParamsSampler as AntRandGoal16PointsTest
from rlkit.envs.ant_rand_goal import _Expert32PointsParamsSampler as AntRandGoal32Points
from rlkit.envs.ant_rand_goal import _ExpertTestTasksFor32PointsParamsSampler as AntRandGoalTestTasksFor32Points

# Ant rand direc 2D imports
from rlkit.envs.ant_rand_direc_2d import AntRandDirec2DEnv
from rlkit.envs.ant_rand_direc_2d import _Expert180DegreesParamsSampler as AntRandDirec180DegSampler
from rlkit.envs.ant_rand_direc_2d import _DebugParamsSamplerV1 as AntRandDirecDebugSamplerV1

# Ant multiple valid targets
from rlkit.envs.ant_multi_target import AntMultiTargetEnv

# for meta simple meta reacher
# from rlkit.envs.dmcs_envs.meta_simple_meta_reacher import build_meta_simple_meta_reacher
# from rlkit.envs.dmcs_envs.meta_simple_meta_reacher import get_params_iterators as get_meta_simple_meta_reacher_params_iters

from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.humanoid import HumanoidEnv

# from dm_control.suite.wrappers import pixels
# from rlkit.envs.dmcs_envs import pixels

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
    'ant_multi_target': lambda: AntMultiTargetEnv(),
    'humanoid_v2': lambda: HumanoidEnv(),
    'ant_v2': lambda: AntEnv(),
    'swimmer_v2': lambda: gym.envs.make('Swimmer-v2'),
    'halfcheetah_v2': lambda: HalfCheetahEnv(),
    'hopper_v2': lambda: gym.envs.make('Hopper-v2'),
    'reacher_v2': lambda: gym.envs.make('Reacher-v2'),
    'pendulum_v0': lambda: gym.envs.make('Pendulum-v0'),
    # 'dmcs_reacher_hard': lambda: DmControlWrapper(suite.load(domain_name='reacher', task_name='hard')),
    # 'dmcs_reacher_easy': lambda: DmControlWrapper(suite.load(domain_name='reacher', task_name='easy')),
    # 'dmcs_simple_reacher': lambda: DmControlWrapper(build_dmcv_simple_reacher()),
    # in render_kwargs can specify: height, width, depth, camera_id
    # 'dmcs_simple_reacher_with_pixels': lambda: DmControlWrapper(
    #     pixels.Wrapper(
    #         build_dmcv_simple_reacher(),
    #         pixels_only=False,
    #         render_kwargs={'height':80, 'width':80, 'camera_id':0}
    #     )
    # ),
    # 'dmcs_simple_meta_reacher': lambda: DmControlWrapper(build_simple_meta_reacher()),
    # 'dmcs_simple_meta_reacher_with_pixels': lambda: DmControlWrapper(
    #     pixels.Wrapper(
    #         build_simple_meta_reacher(),
    #         pixels_only=False,
    #         render_kwargs={'height':80, 'width':80, 'camera_id':0}
    #     )
    # ),
    'jaco_picker': lambda: PickerEnv(),
    'fetch_pick_and_place': lambda: WrappedFetchPickAndPlaceEnv(),
    'debug_fetch_reacher': lambda: DebugReachFetchPickAndPlaceEnv(),
    'debug_fetch_reach_and_lift': lambda: DebugFetchReachAndLiftEnv(),
    # 'rotated_no_head_fetch_reach_anywhere': lambda: WrappedRotatedFetchReachAnywhereEnv(),
    'easy_fetch_pick_and_place': lambda: WrappedEasyFetchPickAndPlaceEnv(),
    'scaled_easy_fetch_pick_and_place': lambda: ScaledWrappedEasyFetchPickAndPlaceEnv(
        acts_max=array([0.11622048, 0.11837779, 1., 0.05]),
        acts_min=array([-0.11406593, -0.11492375, -0.48009082, -0.005]),
        obs_max=array([ 1.35211534e+00,  7.59012039e-01,  8.74170327e-01,  1.35216868e+00,
        7.59075514e-01,  8.65117304e-01,  9.99349991e-03,  9.97504859e-03,
       -5.73782252e-04,  5.14756901e-02,  5.14743797e-02,  3.06240725e-03,
        1.60782802e-02,  9.09377515e-03,  1.45024249e-03,  1.55772198e-03,
        1.27349030e-02,  2.10399698e-02,  3.87118880e-03,  1.10660038e-02,
        2.63549517e-03,  3.08370689e-03,  2.64278933e-02,  2.67708565e-02,
        2.67707824e-02]),
        obs_min=array([ 1.32694457e+00,  7.39177494e-01,  4.25007763e-01,  1.33124808e+00,
        7.39111105e-01,  4.24235324e-01, -9.98595942e-03, -9.98935859e-03,
       -1.10015137e-01,  2.55108763e-06, -8.67902630e-08, -2.71974527e-03,
       -9.63782682e-03, -4.56146656e-04, -1.68586348e-03, -1.55750811e-03,
       -7.64317184e-04, -2.08764492e-02, -3.56580593e-03, -1.05306888e-02,
       -3.47314426e-03, -3.00819907e-03, -1.27082374e-02, -3.65293252e-03,
       -3.65292508e-03]),
       goal_max=array([1.35216868, 0.75907551, 0.87419374]),
       goal_min=array([1.33124808, 0.73911111, 0.42423532]),
       SCALE=0.99
    ),
    'scaled_super_easy_fetch_pick_and_place': lambda: ScaledWrappedSuperEasyFetchPickAndPlaceEnv(
        acts_max=array([0.24968111, 0.24899998, 0.24999904, 0.01499934]),
        acts_min=array([-0.24993695, -0.24931063, -0.24999953, -0.01499993]),
        obs_max=array([0.0152033 , 0.01572069, 0.00401832, 0.02023052, 0.03041435,
        0.20169743, 0.05092416, 0.05090878, 0.01017929, 0.01013457]),
        obs_min=array([-1.77039428e-02, -1.64070528e-02, -1.10015137e-01, -2.06485778e-02,
        -2.99603855e-02, -3.43990285e-03,  0.00000000e+00, -8.67902630e-08,
        -9.50872658e-03, -9.28206220e-03]),
        SCALE=0.99
    ),
    'scaled_and_wrapped_fetch': lambda: ScaledWrappedFetchPickAndPlaceEnv(
        acts_max=array([0.24999679, 0.24999989, 0.24999854, 0.01499987]),
        acts_min=array([-0.24999918, -0.24999491, -0.24998883, -0.01499993]),
        obs_max=array([0.14997844, 0.14999457, 0.0066419 , 0.2896332 , 0.29748688,
        0.4510363 , 0.05095725, 0.05090321, 0.01027833, 0.01043796]),
        obs_min=array([-0.14985769, -0.14991582, -0.11001514, -0.29275747, -0.28962639,
        -0.01673591, -0.00056493, -0.00056452, -0.00953662, -0.00964976]),
        SCALE=0.99
    ),
    # 'scaled_and_wrapped_target_in_air_easy': lambda: ScaledWrappedTargetOnlyInAirFetchPickAndPlaceEnv(
    #     acts_max=array([ 0.24997477,  0.24999408,  0.24999995,  0.01499998]),
    #     acts_min=array([-0.24999714, -0.24999004, -0.24999967, -0.01499985]),
    #     obs_max=array([ 0.04999746,  0.04979575,  0.00102964,  0.09834792,  0.10275888,
    #     0.2026911 ,  0.05087222,  0.05089798,  0.01014106,  0.01024989]),
    #     obs_min=array([ -4.97249838e-02,  -4.99201765e-02,  -1.10015137e-01,
    #     -9.57695575e-02,  -9.56882197e-02,  -2.95093730e-03,
    #      0.00000000e+00,  -8.67902630e-08,  -9.48171330e-03,
    #     -9.57788163e-03]),
    #     SCALE=0.99
    # )
    # 'scaled_and_wrapped_target_in_air_easy': lambda: ScaledWrappedTargetOnlyInAirFetchPickAndPlaceEnv(
    #     acts_max=array([0.24999906, 0.2499996 , 0.24999867, 0.01499948]),
    #     acts_min=array([-0.24999676, -0.2499984 , -0.24999669, -0.01499992]),
    #     obs_max=array([0.14967261, 0.14953164, 0.00056922, 0.28737584, 0.29375757,
    #     0.30215514, 0.05092484, 0.05089244, 0.01006456, 0.01010476]),
    #     obs_min=array([-1.49660926e-01, -1.49646858e-01, -1.10015137e-01, -2.82999770e-01,
    #     -2.85085491e-01, -4.58114691e-03,  0.00000000e+00, -8.67902630e-08,
    #     -9.47718257e-03, -9.47846722e-03]),
    #     SCALE=0.99
    # ),
    # 'scaled_and_wrapped_target_in_air_easy': lambda: ScaledWrappedTargetOnlyInAirFetchPickAndPlaceEnv(
    #     acts_max=array([0.24995736, 0.2499716 , 0.24999983, 0.01499852]),
    #     acts_min=array([-0.24989959, -0.24995068, -0.2499989 , -0.01499998]),
    #     obs_max=array([0.0499439 , 0.04998455, 0.00098634, 0.09421162, 0.10457129,
    #     0.3022664 , 0.05094975, 0.05090175, 0.01024486, 0.01029508]),
    #     obs_min=array([-4.98090099e-02, -4.97771561e-02, -1.10015137e-01, -9.60775777e-02,
    #     -1.03508767e-01, -3.50153560e-03,  0.00000000e+00, -8.67902630e-08,
    #     -9.47353981e-03, -9.62584145e-03]),
    #     SCALE=0.99
    # ),
    # FOR LARGER X-Y RANGE
    # 'scaled_and_wrapped_target_in_air_easy': lambda: ScaledWrappedTargetOnlyInAirFetchPickAndPlaceEnv(
    #     acts_max=array([0.24999749, 0.2499975 , 0.2499998 , 0.01499951]),
    #     acts_min=array([-0.24999754, -0.24999917, -0.24999704, -0.01499989]),
    #     obs_max=array([0.14953716, 0.14865454, 0.00155898, 0.28595684, 0.27644423,
    #     0.20200016, 0.05094223, 0.05082468, 0.01033346, 0.0103368 ]),
    #     obs_min=array([-1.49931348e-01, -1.49895902e-01, -1.10015137e-01, -2.80037372e-01,
    #     -2.82756899e-01, -3.44387360e-03,  0.00000000e+00, -8.67902630e-08,
    #     -9.53356933e-03, -9.71619128e-03]),
    #     SCALE=0.99
    # )
    # FOR LARGER OBJECT RANGE
    'scaled_and_wrapped_target_in_air_easy': lambda: ScaledWrappedTargetOnlyInAirFetchPickAndPlaceEnv(
        acts_max=array([0.24999844, 0.24999035, 0.24999848, 0.01499987]),
        acts_min=array([-0.24999948, -0.24999969, -0.24999971, -0.01499985]),
        obs_max=array([0.14981718, 0.14922823, 0.00105448, 0.19316468, 0.20144443,
        0.20205348, 0.05088978, 0.05087405, 0.01012868, 0.01011336]),
        obs_min=array([-1.49439076e-01, -1.49636276e-01, -1.10015137e-01, -1.99832936e-01,
        -1.96645722e-01, -3.35041414e-03,  0.00000000e+00, -8.67902630e-08,
        -9.49761703e-03, -9.71219664e-03]),
        SCALE=0.99
    ),
    # 'scaled_easy_fetch_pick_and_place': lambda: ScaledActionsEnv(
    #     WrappedEasyFetchPickAndPlaceEnv(),
    #     array([0.0005593, 0.00024555, 0.10793256, 0.0104]),
    #     array([0.01485482, 0.0138236, 0.4666197, 0.02469494])
    # ),
    'zero_few_shot_fetch_env': lambda: ZeroScaledFewShotFetchEnv(),
    'zero_0p9_scale_few_shot_fetch_env': lambda: ZeroScaled0p9FewShotFetchEnv(),
    'zero_0p9_linear_scale_few_shot_fetch_env': lambda: ZeroScaled0p9LinearFewShotFetchEnv(),
    'unscaled_zero_few_shot_fetch_env': lambda: ZeroUnscaledFewShotFetchEnv(),
    'wrap_abs_0p9_linear_scale_few_shot_fetch_env': lambda: WrapAbsZeroScaled0p9LinearFewShotFetchEnv(),

    'zero_0p9_linear_scale_few_shot_reach_env': lambda: ZeroScaled0p9LinearFewShotReachEnv(),
}

meta_envs = {
    'meta_simple_meta_reacher': {
        'meta_train': lambda: build_meta_simple_meta_reacher(train_env=True),
        'meta_test': lambda: build_meta_simple_meta_reacher(train_env=False),
        'info': {
            'is_dmcs_env': True
        }
    },
    'unscaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: BasicFewShotFetchEnv(),
        'meta_test': lambda: BasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'scaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: ScaledBasicFewShotFetchEnv(),
        'meta_test': lambda: ScaledBasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'debug_unscaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: BasicFewShotFetchEnv(),
        'meta_test': lambda: BasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'debug_scaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: ScaledBasicFewShotFetchEnv(),
        'meta_test': lambda: ScaledBasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'zero_unscaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: BasicFewShotFetchEnv(),
        'meta_test': lambda: BasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'zero_scaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: ScaledBasicFewShotFetchEnv(),
        'meta_test': lambda: ScaledBasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'zero_unscaled_basic_few_shot_fetch_env_with_terminal': {
        'meta_train': lambda: BasicFewShotFetchEnv(terminate_on_success=True),
        'meta_test': lambda: BasicFewShotFetchEnv(terminate_on_success=True),
        'info': {
            'is_dmcs_env': False
        }
    },
    'scale_0p9_linear_demos_50_tasks_25_each': {
        'meta_train': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'meta_test': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'info': {
            'is_dmcs_env': False
        }
    },
    'unscaled_16_tasks_few_shot_fetch_env': {
        'meta_train': lambda: BasicFewShotFetchEnv(),
        'meta_test': lambda: BasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'unscaled_32_tasks_few_shot_fetch_env': {
        'meta_train': lambda: BasicFewShotFetchEnv(),
        'meta_test': lambda: BasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'unscaled_16_eval_tasks_few_shot_fetch_env': {
        'meta_train': lambda: BasicFewShotFetchEnv(),
        'meta_test': lambda: BasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'unscaled_24_tasks_few_shot_fetch_env': {
        'meta_train': lambda: BasicFewShotFetchEnv(),
        'meta_test': lambda: BasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'unscaled_20_tasks_few_shot_fetch_env': {
        'meta_train': lambda: BasicFewShotFetchEnv(),
        'meta_test': lambda: BasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'unscaled_12_tasks_few_shot_fetch_env': {
        'meta_train': lambda: BasicFewShotFetchEnv(),
        'meta_test': lambda: BasicFewShotFetchEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'scale_0p9_linear_demos_16_tasks': {
        'meta_train': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'meta_test': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'info': {
            'is_dmcs_env': False
        }
    },
    'unscaled_linear_demos_16_tasks': {
        'meta_train': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'meta_test': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'info': {
            'is_dmcs_env': False
        }
    },
    'scale_0p9_linear_demos_32_tasks': {
        'meta_train': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'meta_test': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'info': {
            'is_dmcs_env': False
        }
    },
    'scale_0p9_linear_demos_24_tasks': {
        'meta_train': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'meta_test': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'info': {
            'is_dmcs_env': False
        }
    },
    'scale_0p9_linear_demos_20_tasks': {
        'meta_train': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'meta_test': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'info': {
            'is_dmcs_env': False
        }
    },
    'scale_0p9_linear_demos_12_tasks': {
        'meta_train': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'meta_test': lambda: StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(terminate_on_success=False),
        'info': {
            'is_dmcs_env': False
        }
    },
    'halfcheetah_rand_vel': {
        'meta_train': lambda: HalfCheetahRandVelEnv(),
        'meta_test': lambda: HalfCheetahRandVelEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'halfcheetah_rand_vel_25_meta_train_25_meta_test': {
        'meta_train': lambda: HalfCheetahRandVelEnv(),
        'meta_test': lambda: HalfCheetahRandVelEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'halfcheetah_rand_vel_0p1_separated': {
        'meta_train': lambda: HalfCheetahRandVelEnv(),
        'meta_test': lambda: HalfCheetahRandVelEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'halfcheetah_rand_vel_debug_meta_sampelr': {
        'meta_train': lambda: HalfCheetahRandVelEnv(),
        'meta_test': lambda: HalfCheetahRandVelEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_200_points': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_120_degrees': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_60_degrees': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_2_directions': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_45_deg_farther': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_opposite_2_directions': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_one_direction': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_middle_60': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_line_params_sampler': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_90_deg_apart': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_five_points': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_eight_points': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_sixteen_points_train_and_test': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_32_points': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_24_points': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_goal_test_tasks_for_32_points': {
        'meta_train': lambda: AntRandGoalEnv(),
        'meta_test': lambda: AntRandGoalEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_direc_2d_180_degrees': {
        'meta_train': lambda: AntRandDirec2DEnv(),
        'meta_test': lambda: AntRandDirec2DEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    'ant_rand_direc_2d_debug_v1': {
        'meta_train': lambda: AntRandDirec2DEnv(),
        'meta_test': lambda: AntRandDirec2DEnv(),
        'info': {
            'is_dmcs_env': False
        }
    },
    # 'zero_unscaled_basic_few_shot_reach_env': {
    #     'meta_train': lambda: BasicFewShotReachEnv(),
    #     'meta_test': lambda: BasicFewShotReachEnv(),
    #     'info': {
    #         'is_dmcs_env': False
    #     }
    # },
}

meta_env_task_params_iterators = {
    'meta_simple_meta_reacher': {
        'meta_train': lambda: get_meta_simple_meta_reacher_params_iters(train_env=True),
        'meta_test': lambda: get_meta_simple_meta_reacher_params_iters(train_env=False)
    },
    'unscaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: few_shot_fetch_env_get_task_params_iterator(train_env=True),
        'meta_test': lambda: few_shot_fetch_env_get_task_params_iterator(train_env=False)
    },
    'scaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: few_shot_fetch_env_get_task_params_iterator(train_env=True),
        'meta_test': lambda: few_shot_fetch_env_get_task_params_iterator(train_env=False)
    },
    'debug_unscaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: debug_few_shot_fetch_env_get_task_params_iterator(train_env=True),
        'meta_test': lambda: debug_few_shot_fetch_env_get_task_params_iterator(train_env=False)
    },
    'debug_scaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: debug_few_shot_fetch_env_get_task_params_iterator(train_env=True),
        'meta_test': lambda: debug_few_shot_fetch_env_get_task_params_iterator(train_env=False)
    },
    'zero_unscaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: zero_few_shot_fetch_env_get_task_params_iterator(train_env=True),
        'meta_test': lambda: zero_few_shot_fetch_env_get_task_params_iterator(train_env=False)
    },
    'zero_scaled_basic_few_shot_fetch_env': {
        'meta_train': lambda: zero_few_shot_fetch_env_get_task_params_iterator(train_env=True),
        'meta_test': lambda: zero_few_shot_fetch_env_get_task_params_iterator(train_env=False)
    },
    'zero_unscaled_basic_few_shot_fetch_env_with_terminal': {
        'meta_train': lambda: zero_few_shot_fetch_env_get_task_params_iterator(train_env=True),
        'meta_test': lambda: zero_few_shot_fetch_env_get_task_params_iterator(train_env=False)
    },
    'scale_0p9_linear_demos_50_tasks_25_each': {
        'meta_train': lambda: few_shot_fetch_env_get_task_params_iterator(train_env=True),
        'meta_test': lambda: few_shot_fetch_env_get_task_params_iterator(train_env=False)
    },
    'unscaled_16_tasks_few_shot_fetch_env': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=16),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=16)
    },
    'unscaled_32_tasks_few_shot_fetch_env': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=32),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=32)
    },
    'unscaled_16_eval_tasks_few_shot_fetch_env': {
        'meta_train': lambda: FewShotFetchBaseParamsSampler(random=52269, num_colors=16),
        'meta_test': lambda: FewShotFetchBaseParamsSampler(random=52269, num_colors=16)
    },
    'unscaled_24_tasks_few_shot_fetch_env': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=24),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=24)
    },
    'unscaled_20_tasks_few_shot_fetch_env': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=20),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=20)
    },
    'unscaled_12_tasks_few_shot_fetch_env': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=12),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=12)
    },
    'scale_0p9_linear_demos_32_tasks': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=32),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=32)
    },
    'scale_0p9_linear_demos_24_tasks': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=24),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=24)
    },
    'scale_0p9_linear_demos_20_tasks': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=20),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=20)
    },
    'scale_0p9_linear_demos_12_tasks': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=12),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=12)
    },
    'scale_0p9_linear_demos_16_tasks': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=16),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=16)
    },
    'unscaled_linear_demos_16_tasks': {
        'meta_train': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=True, num=16),
        'meta_test': lambda: few_shot_fetch_env_get_some_task_params_iterator(train_env=False, num=16)
    },
    'halfcheetah_rand_vel': {
        'meta_train': lambda: HalfCheetahRandVelTrainSampler(num_samples=100),
        'meta_test': lambda: HalfCheetahRandVelTestSampler(num_samples=25),
    },
    'halfcheetah_rand_vel_25_meta_train_25_meta_test': {
        'meta_train': lambda: HalfCheetahRandVelMetaTrainSampler(),
        'meta_test': lambda: HalfCheetahRandVelMetaTestSampler(),
    },
    'halfcheetah_rand_vel_0p1_separated': {
        'meta_train': lambda: HalfCheetahRandVelMetaTrainSampler(),
        'meta_test': lambda: HalfCheetahRandVelMetaTestSampler(),
    },
    'halfcheetah_rand_vel_debug_meta_sampelr': {
        'meta_train': lambda: HalfCheetahRandVelDebugMetaTrainSampler(),
        'meta_test': lambda: HalfCheetahRandVelDebugMetaTestSampler(),
    },
    'ant_rand_goal_200_points': {
        'meta_train': lambda: AntRandGoalExpertTrainSampler(),
        'meta_test': lambda: AntRandGoalExpertTestSampler(),
    },
    'ant_rand_goal_120_degrees': {
        'meta_train': lambda: AntRandGoal120DegreesSampler(),
        'meta_test': lambda: AntRandGoal120DegreesSampler(),
    },
    'ant_rand_goal_60_degrees': {
        'meta_train': lambda: AntRandGoal60DegreesSampler(),
        'meta_test': lambda: AntRandGoal60DegreesSampler(),
    },
    'ant_rand_goal_2_directions': {
        'meta_train': lambda: AntRandGoal2DirectionsSampler(),
        'meta_test': lambda: AntRandGoal2DirectionsSampler()
    },
    'ant_rand_goal_45_deg_farther': {
        'meta_train': lambda: AntRandGoal45DegFartherParamsSampler(),
        'meta_test': lambda: AntRandGoal45DegFartherParamsSampler()
    },
    'ant_rand_goal_opposite_2_directions': {
        'meta_train': lambda: AntRandGoalOpposite2DirectionsSampler(),
        'meta_test': lambda: AntRandGoalOpposite2DirectionsSampler()
    },
    'ant_rand_goal_one_direction': {
        'meta_train': lambda: AntRandGoalOneDirectionSampler(),
        'meta_test': lambda: AntRandGoalOneDirectionSampler()
    },
    'ant_rand_goal_middle_60': {
        'meta_train': lambda: AntRandGoalMiddle60ParamsSampler(),
        'meta_test': lambda: AntRandGoalMiddle60ParamsSampler()
    },
    'ant_rand_goal_line_params_sampler': {
        'meta_train': lambda: AntRandGoalLineParamsSampler(),
        'meta_test': lambda: AntRandGoalLineParamsSampler()
    },
    'ant_rand_goal_90_deg_apart': {
        'meta_train': lambda: AntRandGoal90DegApart(),
        'meta_test': lambda: AntRandGoal90DegApart()
    },
    'ant_rand_goal_five_points': {
        'meta_train': lambda: AntRandGoalFivePoints(),
        'meta_test': lambda: AntRandGoalFivePoints()
    },
    'ant_rand_goal_eight_points': {
        'meta_train': lambda: AntRandGoalEightPoints(),
        'meta_test': lambda: AntRandGoalEightPoints()
    },
    'ant_rand_goal_sixteen_points_train_and_test': {
        'meta_train': lambda: AntRandGoal16PointsTrain(),
        'meta_test': lambda: AntRandGoal16PointsTest()
    },
    'ant_rand_goal_32_points': {
        'meta_train': lambda: AntRandGoal32Points(),
        'meta_test': lambda: AntRandGoal32Points()
    },
    'ant_rand_goal_24_points': {
        'meta_train': lambda: AntRandGoal24Points(),
        'meta_test': lambda: AntRandGoal24Points()
    },
    'ant_rand_goal_two_points': {
        'meta_train': lambda: AntRandGoalTwoPoints(),
        'meta_test': lambda: AntRandGoalTwoPoints()
    },
    'ant_rand_goal_test_tasks_for_32_points': {
        'meta_train': lambda: AntRandGoalTestTasksFor32Points(),
        'meta_test': lambda: AntRandGoalTestTasksFor32Points()
    },
    'ant_rand_direc_2d_180_degrees': {
        'meta_train': lambda: AntRandDirec180DegSampler(),
        'meta_test': lambda: AntRandDirec180DegSampler()
    },
    'ant_rand_direc_2d_debug_v1': {
        'meta_train': lambda: AntRandDirecDebugSamplerV1(),
        'meta_test': lambda: AntRandDirecDebugSamplerV1()
    },
    # 'zero_unscaled_basic_few_shot_reach_env': {
    #     'meta_train': lambda: zero_few_shot_reach_env_get_task_params_iterator(train_env=True),
    #     'meta_test': lambda: zero_few_shot_reach_env_get_task_params_iterator(train_env=False)
    # },
}

train_test_envs = {
    # 'dmcs_simple_meta_reacher': {
    #     'train': lambda: DmControlWrapper(build_simple_meta_reacher(train_env=True)),
    #     'test': lambda: DmControlWrapper(build_simple_meta_reacher(train_env=False))
    # }
}


def get_meta_env(env_specs):
    base_env_name = env_specs['base_env_name']
    env_dict = meta_envs[base_env_name]
    meta_train_env, meta_test_env = meta_envs[base_env_name]['meta_train'](), meta_envs[base_env_name]['meta_test']()
    if env_specs['need_pixels']:
        if env_dict['info']['is_dmcs_env']:
            meta_train_env = pixels.Wrapper(
                meta_train_env,
                pixels_only=False,
                render_kwargs=env_specs['render_kwargs']
            )
            meta_test_env = pixels.Wrapper(
                meta_test_env,
                pixels_only=False,
                render_kwargs=env_specs['render_kwargs']
            )
        else:
            raise NotImplementedError()
    # if its a dmcs env we need to wrap it to look like a gym env
    if env_dict['info']['is_dmcs_env']:
        meta_train_env = DmControlWrapper(meta_train_env)
        meta_test_env = DmControlWrapper(meta_test_env)
    if env_specs['normalized']:
        meta_train_env, meta_test_env = NormalizedBoxEnv(meta_train_env), NormalizedBoxEnv(meta_test_env)
    return meta_train_env, meta_test_env


def get_meta_env_params_iters(env_specs):
    base_env_name = env_specs['base_env_name']
    meta_train_iter = meta_env_task_params_iterators[base_env_name]['meta_train']()
    meta_test_iter = meta_env_task_params_iterators[base_env_name]['meta_test']()
    return meta_train_iter, meta_test_iter


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
