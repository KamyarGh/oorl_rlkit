import argparse
import os
import yaml
from itertools import product

from rlkit.core.eval_util import plot_experiment_returns


def plot_results(exp_name, variables_to_permute, plot_mean=False, y_axis_lims=None):
    output_dir = '/ais/gobi6/kamyar/oorl_rlkit/output'
    plots_dir = '/u/kamyar/oorl_rlkit/plots'
    variant_dir = os.path.join(output_dir, exp_name.replace('-', '_'))
    sub_dir_name = os.listdir(variant_dir)[0]
    variant_file_path = os.path.join(variant_dir, sub_dir_name, 'exp_spec_definition.yaml')
    with open(variant_file_path, 'r') as spec_file:
        spec_string = spec_file.read()
        all_exp_specs = yaml.load(spec_string)
    
    exp_plot_dir = os.path.join(plots_dir, exp_name)
    os.makedirs(exp_plot_dir, exist_ok=True)

    split_var_names = [s.split('.') for s in variables_to_permute]
    format_names = [s[-1] for s in split_var_names]
    name_to_format = '_'.join(s + '_{}' for s in format_names)

    all_variable_values = []
    for path in split_var_names:
        v = all_exp_specs['variables']
        for k in path: v = v[k]
        all_variable_values.append(v)
    
    for p in product(*all_variable_values):
        constraints = {
            k:v for k,v in zip(variables_to_permute, p)
        }
        name = name_to_format.format(*p)
        try:
            plot_experiment_returns(
                os.path.join(output_dir, exp_name),
                name,
                os.path.join(exp_plot_dir, '{}.png'.format(name)),
                y_axis_lims=y_axis_lims,
                plot_mean=plot_mean,
                constraints=constraints
            )
        except Exception as e:
            # raise(e)
            print('failed ')
    



# for r in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
#     # for soft_t in [0.005, 0.01]:
#     # for normalized in [True, False]:
#     constraints = {
#         'algo_params.reward_scale': r,
#         # 'algo_params.soft_target_tau': soft_t,
#         # 'env_specs.normalized': normalized
#     }
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/gym-reacher-hype-search',
#         'gym_reacher_hype_search_rew_scale_{}'.format(r),
#         '/u/kamyar/oorl_rlkit/plots/gym_reacher_hype_search_{}.png'.format(r),
#         # y_axis_lims=[0, 70],
#         plot_mean=False,
#         constraints=constraints
#     )

# for b in ['ant_v2', 'halfcheetah_v2', 'swimmer_v2']:
#     # for soft_t in [0.005, 0.01]:
#     for normalized in [True, False]:
#         constraints = {
#             # 'algo_params.reward_scale': r,
#             'env_specs.base_env_name': b,
#             # 'algo_params.soft_target_tau': soft_t,
#             'env_specs.normalized': normalized
#         }
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/correct-get-gym-env-experts',
#             'gym_{}_normalized_{}_expert'.format(b, normalized),
#             '/u/kamyar/oorl_rlkit/plots/gym_{}_normalized_{}_expert.png'.format(b, normalized),
#             # y_axis_lims=[0, 70],
#             plot_mean=False,
#             constraints=constraints
#         )

# ------------------------------------------------------------------------
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/fixed-halfcheetah-gpu',
#     'fixed halfcheetah gpu with no terminal condition',
#     '/u/kamyar/oorl_rlkit/plots/no_terminal_halfcheetah.png',
#     # y_axis_lims=[-25, 0],
#     plot_mean=False,
#     # constraints=constraints
# )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/halfcheetah-gail-1000-replay',
#     'halfcheetah_gail_1000_replay',
#     '/u/kamyar/oorl_rlkit/plots/halfcheetah_gail_1000_replay.png',
#     # y_axis_lims=[-25, 0],
#     plot_mean=False,
#     # constraints=constraints
# )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/halfcheetah-gail-10000-replay',
#     'halfcheetah_gail_10000_replay',
#     '/u/kamyar/oorl_rlkit/plots/halfcheetah_gail_10000_replay.png',
#     # y_axis_lims=[-25, 0],
#     plot_mean=False,
#     # constraints=constraints
# )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/halfcheetah-gail-100000-replay',
#     'halfcheetah_gail_100000_replay',
#     '/u/kamyar/oorl_rlkit/plots/halfcheetah_gail_100000_replay.png',
#     # y_axis_lims=[-25, 0],
#     plot_mean=False,
#     # constraints=constraints
# )

# ------------------------------------------------------------------------
# for r_iters in [1, 3]:
#     for p_iters in [1,3,32]:
#         for rew in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
#             constraints = {
#                 'gail_params.num_reward_updates': r_iters,
#                 'gail_params.num_policy_updates': p_iters,
#                 'policy_params.reward_scale': rew
#             }

#             base_name = 'r_updates_{}_p_updates_{}_rew_{}'.format(r_iters, p_iters, rew)

#             plot_experiment_returns(
#                 '/ais/gobi6/kamyar/oorl_rlkit/output/halfcheetah-gail-1000-replay',
#                 base_name + ' buffer size 1000',
#                 '/u/kamyar/oorl_rlkit/plots/gail_halfcheetah_buffer_1000/{}.png'.format(base_name),
#                 y_axis_lims=[-1000, 3000],
#                 plot_mean=False,
#                 constraints=constraints
#             )

#             plot_experiment_returns(
#                 '/ais/gobi6/kamyar/oorl_rlkit/output/halfcheetah-gail-10000-replay',
#                 base_name + ' buffer size 10000',
#                 '/u/kamyar/oorl_rlkit/plots/gail_halfcheetah_buffer_10000/{}.png'.format(base_name),
#                 y_axis_lims=[-1000, 3000],
#                 plot_mean=False,
#                 constraints=constraints
#             )

#             plot_experiment_returns(
#                 '/ais/gobi6/kamyar/oorl_rlkit/output/halfcheetah-gail-100000-replay',
#                 base_name + ' buffer size 100000',
#                 '/u/kamyar/oorl_rlkit/plots/gail_halfcheetah_buffer_100000/{}.png'.format(base_name),
#                 y_axis_lims=[-1000, 3000],
#                 plot_mean=False,
#                 constraints=constraints
#             )


# for rb_size in [1000, 10000, 100000, 1000000]:
#     constraints = {
#         'gail_params.replay_buffer_size': rb_size
#     }
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/halfcheetah-gail-longer-run-hyper-search',
#         'halfcheetah longer run iters 1-32 rew 5 buffer %d' % rb_size,
#         '/u/kamyar/oorl_rlkit/plots/halfcheetah_gail_buffer_%d.png' % rb_size,
#         y_axis_lims=[-100, 6000],
#         plot_mean=False,
#         constraints=constraints
#     )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/halfcheetah-gail-sumsample-20',
#     'halfcheetah subsampling 20',
#     '/u/kamyar/oorl_rlkit/plots/halfcheetah_gail_subsample_20.png',
#     y_axis_lims=[-100, 6000],
#     plot_mean=False,
# )


# # exp_name = 'just_sin_cos_dmcs_simple_meta_reacher_with_sin_cos'
# # exp_name = 'correct-dmcs-simple-meta-reacher-with-sin-cos'
# # these next two are the same plus either finger-to-target vector or finger pos
# exp_name = 'longer-run-with-finger-pos-dmcs-simple-meta-reacher-with-sin-cos'
# # exp_name = 'longer-run-with-to-target-dmcs-simple-meta-reacher-with-sin-cos'
# for rew in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
#     constraints = {
#         'algo_params.reward_scale': rew,
#         'env_specs.normalized': False,
#         'env_specs.train_test_env': True
#     }
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/{}'.format(exp_name),
#         'sin-cos simple meta reacher hype search rew %f' % rew,
#         '/u/kamyar/oorl_rlkit/plots/unnorm_{}_rew_{}.png'.format(exp_name, rew),
#         y_axis_lims=[0, 70],
#         plot_mean=False,
#         constraints=constraints
#     )


# !!!!!! test_halfcheetah_dac_25_trajs_20_subsampling_disc_with_batchnorm
# halfcheetah-dac-25-trajs-20-subsampling-disc-with-batchnorm
# halfcheetah-dac-25-trajs-20-subsampling-disc-without-batchnorm
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/test-halfcheetah-dac-25-trajs-20-subsampling-disc-with-batchnorm',
#     'test_halfcheetah_dac_25_trajs_20_subsampling_disc_with_batchnorm',
#     '/u/kamyar/oorl_rlkit/plots/test_halfcheetah_dac_25_trajs_20_subsampling_disc_with_batchnorm.png',
#     y_axis_lims=[-1000, 10000],
#     plot_mean=False,
#     plot_horizontal_lines_at=[8191.5064, 8191.5064 + 629.3840, 8191.5064 - 629.3840],
#     horizontal_lines_names=['expert_mean', 'expert_high_std', 'expert_low_std']
# )



# for num_exp_traj in [25, 100]:
#     for exp_name in [
#             'good_halfcheetah_100_trajs_1_subsampling',
#             'good_halfcheetah_100_trajs_20_subsampling'
#         ]:
#         for net_size in [256, 100]:
#             constraints = {
#                 'expert_name': exp_name,
#                 'num_expert_trajs': num_exp_traj,
#                 'policy_net_size': net_size
#             }
#             name = 'expert_{expert_name}_num_trajs_used_{num_expert_trajs}_net_size_{policy_net_size}'.format(**constraints)
#             plot_experiment_returns(
#                 '/ais/gobi6/kamyar/oorl_rlkit/output/halfcheetah-bc-100-trajs',
#                 'halfcheetah behaviour cloning {}'.format(name),
#                 '/u/kamyar/oorl_rlkit/plots/{}.png'.format(name),
#                 y_axis_lims=[-1000, 10000],
#                 plot_mean=False,
#                 constraints=constraints
#                 # plot_horizontal_lines_at=[8191.5064, 8191.5064 + 629.3840, 8
# 191.5064 - 629.3840],
#                 # horizontal_lines_names=['expert_mean', 'expert_high_std', 'expert_low_std']
#             )

# for num_exp_traj in [25, 100]:
#     for exp_name in [
#             'dmcs_with_finger_pos_and_sin_cos_simple_meta_reacher_100_trajs_20_subsampling',
#             'dmcs_with_finger_pos_and_sin_cos_simple_meta_reacher_100_trajs_1_subsampling'
#         ]:
#         for net_size in [256, 100]:
#             constraints = {
#                 'expert_name': exp_name,
#                 'num_expert_trajs': num_exp_traj,
#                 'policy_net_size': net_size
#             }
#             name = 'expert_{expert_name}_num_trajs_used_{num_expert_trajs}_net_size_{policy_net_size}'.format(**constraints)
#             plot_experiment_returns(
#                 '/ais/gobi6/kamyar/oorl_rlkit/output/with-logging-reacher-bc-100-trajs',
#                 'reacher behaviour cloning {}'.format(name),
#                 '/u/kamyar/oorl_rlkit/plots/reacher_bc/{}.png'.format(name),
#                 y_axis_lims=[0, 80],
#                 plot_mean=False,
#                 constraints=constraints
#                 # plot_horizontal_lines_at=[8191.5064, 8191.5064 + 629.3840, 8191.5064 - 629.3840],
#                 # horizontal_lines_names=['expert_mean', 'expert_high_std', 'expert_low_std']
#             )


# ----------------------------------------------------------------------------------------------
# '''
# gt_z_2_layer_relu_disc_no_bn_16_pre_rollout_1_rollout_per_epoch_with_grad_pen
# gt_z_3_layer_tanh_disc_with_bn_16_pre_rollout_1_rollout_per_epoch_with_grad_pen
# gt_z_2_layer_relu_disc_with_bn_16_pre_rollout_1_rollout_per_epoch_with_grad_pen
# gt_z_2_layer_tanh_disc_no_bn_16_pre_rollout_1_rollout_per_epoch_with_grad_pen
# gt_z_2_layer_tanh_disc_with_bn_16_pre_rollout_1_rollout_per_epoch_with_grad_pen

# fixed_* version of some of the aboves

# first_np_airl_proper_run_tanh_with_bn_disc
# first_np_airl_proper_run_tanh_with_bn_disc
# '''
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/first-np-airl-proper-run-tanh-with-bn-disc',
#     'first try np airl',
#     '/u/kamyar/oorl_rlkit/plots/msmr_proper_np_airl.png',
#     y_axis_lims=[0, 80],
#     plot_mean=False,
#     # plot_horizontal_lines_at=[8191.5064, 8191.5064 + 629.3840, 8191.5064 - 629.3840],
#     # horizontal_lines_names=['expert_mean', 'expert_high_std', 'expert_low_std']
# )
# ----------------------------------------------------------------------------------------------
# Result for the fetch environment v1 from openai gym for the pick and place task using airl
# for rew in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
#     for disc_blocks in [2, 3]:
#         constraints = {
#             'policy_params.reward_scale': rew,
#             'disc_num_blocks': disc_blocks
#         }
#     name = 'disc_hid_256_disc_num_blocks_{}_rew_scale_{}'.format(disc_blocks, rew)
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/correct-fetch-dac-disc-256',
#         name,
#         '/u/kamyar/oorl_rlkit/plots/{}.png'.format(name),
#         y_axis_lims=[-50, 0],
#         plot_mean=False,
#         constraints=constraints
#         # plot_horizontal_lines_at=[8191.5064, 8191.5064 + 629.3840, 8191.5064 - 629.3840],
#         # horizontal_lines_names=['expert_mean', 'expert_high_std', 'expert_low_std']
#     )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/try-fetch-dac-2-layer-policy',
#     'fetch dac 2 layer policy',
#     '/u/kamyar/oorl_rlkit/plots/fetch_dac_2_layer_policy.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved'
# )
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/corect-fetch-dac-4-layer-policy-disc-100',
#     'fetch dac 4 layer policy disc hidden 100',
#     '/u/kamyar/oorl_rlkit/plots/fetch_dac_4_layer_policy_disc_hid_100.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved'
# )
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/corect-fetch-dac-4-layer-policy-disc-256',
#     'fetch dac 4 layer policy disc hidden 256',
#     '/u/kamyar/oorl_rlkit/plots/fetch_dac_4_layer_policy_disc_hid_256.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved'
# )

# for rew in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
#     for disc_blocks in [2 ,3]:
#         constraints = {
#             'policy_params.reward_scale': rew,
#             'disc_num_blocks': disc_blocks
#         }
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/correct-debug-fetch-reacher-2-layer-policy',
#             'fetch reacher rew %f disc blocks %d' % (rew, disc_blocks),
#             '/u/kamyar/oorl_rlkit/plots/fetch_reacher/fetch_reacher_rew_%.0f_disc_blocks_%d.png' % (rew, disc_blocks),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/fixed-debug-fetch-reach-and-lift-dac',
#     'fetch reach and lift dac from 100 trajs of len 50',
#     '/u/kamyar/oorl_rlkit/plots/fetch_reach_and_lift_dac.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved'
# )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/fixed-fetch-anywhere-reach',
#     'fetch reach anywhere SAC',
#     '/u/kamyar/oorl_rlkit/plots/fetch_reach_anywhere_SAC.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved'
# )
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/fixed-fetch-anywhere-reach-10x-shaping',
#     'fetch reach anywhere 10x shaping SAC',
#     '/u/kamyar/oorl_rlkit/plots/fetch_reach_anywhere_10x_shaping_SAC.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved'
# )
# for rew in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0]:
#     constraints = {
#         'algo_params.reward_scale': rew
#     }
#     name = 'fetch_anywhere_10x_shaping_rew_%d' % rew
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/fixed-fetch-anywhere-reach-10x-shaping',
#         'fetch reach anywhere 10x shaping SAC rew scale %d' % rew,
#         '/u/kamyar/oorl_rlkit/plots/fetch_reach_anywhere_10x_shaping_SAC_rew_%d.png' % rew,
#         y_axis_lims=[0, 1.0],
#         plot_mean=False,
#         column_name='Percent_Solved',
#         constraints=constraints
#     )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/batch-1-fetch-pick-and-place-with-noisy-demos-3-lay-pol-2-lay-disc',
#     'fetch pick and place noisy demos',
#     '/u/kamyar/oorl_rlkit/plots/batch_1_fetch_pick_and_place_with_noisy_demos_3_lay_pol_2_lay_disc.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved'
# )
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/batch-2-fetch-pick-and-place-with-noisy-demos-3-lay-pol-2-lay-disc',
#     'fetch pick and place noisy demos',
#     '/u/kamyar/oorl_rlkit/plots/batch_2_fetch_pick_and_place_with_noisy_demos_3_lay_pol_2_lay_disc.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved'
# )
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/batch-3-fetch-pick-and-place-with-noisy-demos-3-lay-pol-2-lay-disc',
#     'fetch pick and place noisy demos',
#     '/u/kamyar/oorl_rlkit/plots/batch_3_fetch_pick_and_place_with_noisy_demos_3_lay_pol_2_lay_disc.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved'
# )

# for pol_size in [100, 256]:
#     for pol_lays in [2, 3, 4]:
#         constraints = {
#             'policy_net_size': pol_size,
#             'num_policy_layers': pol_lays
#         }
#         plot_experiment_returns(
#             '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-larger-object-range-easy-in-the-air-fetch-bc',
#             'larger object range easy fetch pick and place pol layers %d pol size %d' % (pol_lays, pol_size),
#             '/h/kamyar/oorl_rlkit/plots/larger_object_range_easy_fetch_bc/pol_lays_%d_pol_size_%d.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/scratch/gobi2/kamyar/oorl_rlkit/output/test-larger-x-y-range-easy-in-the-air-fetch-bc',
#             'larger x-y range easy fetch pick and place pol layers %d pol size %d' % (pol_lays, pol_size),
#             '/h/kamyar/oorl_rlkit/plots/larger_x_y_range_easy_fetch_bc/pol_lays_%d_pol_size_%d.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/scratch/gobi2/kamyar/oorl_rlkit/output/test-larger-z-range-easy-in-the-air-fetch-bc',
#             'easy fetch pick and place pol layers %d pol size %d' % (pol_lays, pol_size),
#             '/h/kamyar/oorl_rlkit/plots/larger_z_range_easy_fetch_bc/pol_lays_%d_pol_size_%d.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/test-easy-fetch-bc',
#             'easy fetch pick and place pol layers %d pol size %d' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/pol_lays_%d_pol_size_%d.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/test-easy-fetch-bc-with-normalized-acts',
#             'easy fetch pick and place pol layers %d pol size %d with norm acts' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/pol_lays_%d_pol_size_%d_with_norm_acts.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/test-easy-fetch-bc-with-normalized-acts-with-bn',
#             'easy fetch pick and place pol layers %d pol size %d with norm acts with bn' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/pol_lays_%d_pol_size_%d_with_norm_acts_with_bn.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/test-easy-fetch-bc-with-normalized-acts-with-layer-norm',
#             'easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/large-batch-easy-fetch-bc-with-normalized-acts-with-layer-norm',
#             'large batch easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/large_batch_pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/full-batch-easy-fetch-bc-with-normalized-acts-with-layer-norm',
#             'full batch easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/full_batch_pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/tanh-large-batch-easy-fetch-bc-with-normalized-acts-with-layer-norm',
#             'Tanh large batch easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/tanh_large_batch_pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/tanh-full-batch-easy-fetch-bc-with-normalized-acts-with-layer-norm',
#             'Tanh full batch easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/tanh_full_batch_pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         # this next one is actually large batch (1024) not full batch
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/1000-demos-tanh-full-batch-easy-fetch-bc-with-normalized-acts-with-layer-norm',
#             '1000 demos Tanh 1024 batch easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/1000_demos_tanh_1024_batch_pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/lower-lr-1000-demos-tanh-1024-batch-easy-fetch-bc-with-normalized-acts-with-layer-norm',
#             'lower lr 1000 demos Tanh 1024 batch easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/lower_lr_1000_demos_tanh_1024_batch_pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/fixed-norm-all-lower-lr-1000-clipped-demos-tanh-1024-batch-easy-fetch-bc-with-normalized-acts-with-layer-norm/',
#             'all normalized lower lr 1000 demos Tanh 1024 batch easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/all_norm_lower_lr_1000_demos_tanh_1024_batch_pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/noisy-eval-fixed-norm-all-lower-lr-1000-clipped-demos-tanh-1024-batch-easy-fetch-bc-with-normalized-acts-with-layer-norm',
#             'noisy eval all normalized lower lr 1000 demos Tanh 1024 batch easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/noisy_eval_all_norm_lower_lr_1000_demos_tanh_1024_batch_pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/fixed-norm-all-lower-lr-1000-clipped-demos-tanh-1024-batch-easy-fetch-bc-with-normalized-acts-no-layer-norm',
#             'no layer norm all normalized lower lr 1000 demos Tanh 1024 batch easy fetch pick and place pol layers %d pol size %d with norm acts with layer norm' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/easy_fetch_bc/no_layer_norm_all_norm_lower_lr_1000_demos_tanh_1024_batch_pol_lays_%d_pol_size_%d_with_norm_acts_with_layer_norm.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/disc-2-256-pol-2-100-easy-fetch-pick-and-place',
#     'disc 2 - 256, pol 2 - 100',
#     '/u/kamyar/oorl_rlkit/plots/easy_fetch_dac/disc_2_256_pol_2_100.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved',
# )
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/disc-2-256-pol-2-256-easy-fetch-pick-and-place',
#     'disc 2 - 256, pol 2 - 256',
#     '/u/kamyar/oorl_rlkit/plots/easy_fetch_dac/disc_2_256_pol_2_256.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved',
# )
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/batch-1-disc-2-100-pol-2-256-easy-fetch-pick-and-place',
#     'disc 2 - 100, pol 2 - 256',
#     '/u/kamyar/oorl_rlkit/plots/easy_fetch_dac/batch_1_disc_2_100_pol_2_256.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved',
# )
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/batch-2-disc-2-100-pol-2-256-easy-fetch-pick-and-place',
#     'disc 2 - 100, pol 2 - 256',
#     '/u/kamyar/oorl_rlkit/plots/easy_fetch_dac/batch_2_disc_2_100_pol_2_256.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved',
# )
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/disc-2-100-pol-2-100-easy-fetch-pick-and-place',
#     'disc 2 - 100, pol 2 - 100',
#     '/u/kamyar/oorl_rlkit/plots/easy_fetch_dac/disc_2_100_pol_2_100.png',
#     y_axis_lims=[0, 1.0],
#     plot_mean=False,
#     column_name='Percent_Solved',
# )

# for pol_size in [100, 256]:
#     for pol_lays in [2, 3, 4]:
#         constraints = {
#             'policy_net_size': pol_size,
#             'num_policy_layers': pol_lays
#         }
#         plot_experiment_returns(
#             '/scratch/gobi2/kamyar/oorl_rlkit/output/test-easy-in-the-air-fetch-bc',
#             'fetch pick and place bc pol layers %d pol size %d' % (pol_lays, pol_size),
#             '/h/kamyar/oorl_rlkit/plots/in_the_air_easy_fetch/pol_lays_%d_pol_size_%d.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/fetch-bc-with-max-ent-demos',
#             'fetch pick and place bc pol layers %d pol size %d' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/fetch_bc/pol_lays_%d_pol_size_%d.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/super-easy-fetch-bc',
#             'super easy fetch pick and place pol layers %d pol size %d' % (pol_lays, pol_size),
#             '/u/kamyar/oorl_rlkit/plots/super_easy_fetch_bc/pol_lays_%d_pol_size_%d.png' % (pol_lays, pol_size),
#             y_axis_lims=[0, 1.0],
#             plot_mean=False,
#             column_name='Percent_Solved',
#             constraints=constraints
#         )

# for rew in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0]: 
#     constraints = {
#         'policy_params.reward_scale': rew
#     }   
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/disc-100-100-pol-256-256-super-easy-fetch-dac',
#         'disc 100-100, pol 256-256, rew %d' % rew,
#         '/u/kamyar/oorl_rlkit/plots/super_easy_fetch_dac/disc_100_100_pol_256_256_rew_%d.png' % rew,
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#         constraints=constraints
#     )
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/disc-100-64-pol-256-256-super-easy-fetch-dac',
#         'disc 100-64, pol 256-256, rew %d' % rew,
#         '/u/kamyar/oorl_rlkit/plots/super_easy_fetch_dac/disc_100_64_pol_256_256_rew_%d.png' % rew,
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#         constraints=constraints
#     )
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/seed-5914-disc-100-100-pol-100-100-super-easy-fetch-dac',
#         'disc 100-100, pol 100-100, rew %d' % rew,
#         '/u/kamyar/oorl_rlkit/plots/super_easy_fetch_dac/seed_5914_disc_100_100_pol_100_100_rew_%d.png' % rew,
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#         constraints=constraints
#     )
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/seed-9783-disc-100-100-pol-100-100-super-easy-fetch-dac',
#         'disc 100-100, pol 100-100, rew %d' % rew,
#         '/u/kamyar/oorl_rlkit/plots/super_easy_fetch_dac/seed_9783_disc_100_100_pol_100_100_rew_%d.png' % rew,
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#         constraints=constraints
#     )
#     plot_experiment_returns(
#         '/ais/gobi6/kamyar/oorl_rlkit/output/disc-100-64-pol-100-100-super-easy-fetch-dac',
#         'disc 100-64, pol 100-100, rew %d' % rew,
#         '/u/kamyar/oorl_rlkit/plots/super_easy_fetch_dac/disc_100_64_pol_100_100_rew_%d.png' % rew,
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#         constraints=constraints
#     )

# for rew in [2.0, 4.0, 6.0, 8.0]:
#     for grad_pen in [1.0, 3.0, 5.0, 7.0]:
#         constraints = {
#             'policy_params.reward_scale': rew,
#             'grad_pen_weight': grad_pen
#         }
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/correct-disc-100-64-pol-100-100-100-fetch-dac-with-max-ent-demos',
#             'disc 100-64, pol 100-100-100, rew %d, grad_pen %.1f' % (rew, grad_pen),
#             '/u/kamyar/oorl_rlkit/plots/fetch_dac/disc_100_64_pol_100_100_100_rew_%d_grad_pen_%.1f.png' % (rew, grad_pen),
#             y_axis_lims=[-0.05, 1.05],
#             plot_mean=False,
#             column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#             constraints=constraints
#         )
#         plot_experiment_returns(
#             '/ais/gobi6/kamyar/oorl_rlkit/output/narrower-search-disc-100-64-pol-100-100-super-easy-fetch-dac',
#             'disc 100-64, pol 100-100, rew %d, grad_pen %.1f' % (rew, grad_pen),
#             '/u/kamyar/oorl_rlkit/plots/super_easy_fetch_dac/narrower_search/disc_100_64_pol_100_100_rew_%d_grad_pen_%.1f.png' % (rew, grad_pen),
#             y_axis_lims=[-0.05, 1.05],
#             plot_mean=False,
#             column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#             constraints=constraints
#         )

# for rew in [2.0, 4.0, 6.0, 8.0]:
#     for disc_hid in [[100, 100], [256, 256], [100, 100, 100], [256, 256, 256]]:
#         for grad_pen in [0.0, 1.0, 5.0, 10.0]:
#             # for num_rew_upd in [65, 32]:
#             constraints = {
#                 'policy_params.reward_scale': rew,
#                 'disc_hidden_sizes': disc_hid,
#                 'gail_params.grad_pen_weight': grad_pen,
#                 # 'gail_params.num_reward_updates': num_rew_upd
#             }
#             plot_experiment_returns(
#                 '/scratch/gobi2/kamyar/oorl_rlkit/output/more-disc-iters-larger-object-range-easy-in-the-air-fetch-dac/',
#                 'cpu disc hid {}, pol 100-100-100, rew {}, grad_pen {}, num rew updates 140, num pol updates 70'.format(disc_hid, rew, grad_pen),
#                 '/h/kamyar/oorl_rlkit/plots/more_disc_iters_larger_object_range_easy_in_the_air_fetch_dac/disc_{}_pol_100_100_rew_{}_grad_pen_{}_num_rew_upd_128_num_pol_upd_65.png'.format(disc_hid, rew, grad_pen),
#                 y_axis_lims=[-0.05, 1.05],
#                 plot_mean=False,
#                 column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#                 constraints=constraints
#             )
#             plot_experiment_returns(
#                 '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-grad-pen-more-disc-iters-larger-x-y-range-easy-in-the-air-fetch-dac/',
#                 'cpu disc hid {}, pol 100-100-100-100, rew {}, grad_pen {}, num rew updates 140, num pol updates 70'.format(disc_hid, rew, grad_pen),
#                 '/h/kamyar/oorl_rlkit/plots/correct_grad_pen_more_disc_iters_larger_x_y_range_easy_in_the_air_fetch_dac/disc_{}_pol_100_100_rew_{}_grad_pen_{}_num_rew_upd_128_num_pol_upd_65.png'.format(disc_hid, rew, grad_pen),
#                 y_axis_lims=[-0.05, 1.05],
#                 plot_mean=False,
#                 column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#                 constraints=constraints
#             )
        # plot_experiment_returns(
        #     '/scratch/gobi2/kamyar/oorl_rlkit/output/cpu-more-disc-iters-larger-x-y-range-easy-in-the-air-fetch-dac',
        #     'cpu disc hid {}, pol 100-100-100-100, rew {}, grad_pen 10, num rew updates 140, num pol updates 70'.format(disc_hid, rew),
        #     '/h/kamyar/oorl_rlkit/plots/cpu_more_disc_iters_larger_x_y_range_easy_in_the_air_fetch_dac/disc_{}_pol_100_100_rew_{}_grad_pen_10_num_rew_upd_128_num_pol_upd_65.png'.format(disc_hid, rew),
        #     y_axis_lims=[-0.05, 1.05],
        #     plot_mean=False,
        #     column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
        #     constraints=constraints
        # )
        # plot_experiment_returns(
        #     '/scratch/gobi2/kamyar/oorl_rlkit/output/gpu-more-disc-iters-larger-x-y-range-easy-in-the-air-fetch-dac',
        #     'gpu disc hid {}, pol 100-100-100-100, rew {}, grad_pen 10, num rew updates 140, num pol updates 70'.format(disc_hid, rew),
        #     '/h/kamyar/oorl_rlkit/plots/gpu_more_disc_iters_larger_x_y_range_easy_in_the_air_fetch_dac/disc_{}_pol_100_100_rew_{}_grad_pen_10_num_rew_upd_128_num_pol_upd_65.png'.format(disc_hid, rew),
        #     y_axis_lims=[-0.05, 1.05],
        #     plot_mean=False,
        #     column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
        #     constraints=constraints
        # )
        # plot_experiment_returns(
        #     '/scratch/gobi2/kamyar/oorl_rlkit/output/airly-more-disc-iters-larger-x-y-range-easy-in-the-air-fetch-dac',
        #     'airly disc hid {}, pol 100-100-100-100, rew {}, grad_pen 10, num rew updates 16, num pol updates 8, buffer size 17500'.format(disc_hid, rew),
        #     '/h/kamyar/oorl_rlkit/plots/airly_more_disc_iters_larger_x_y_range_easy_in_the_air_fetch_dac/disc_{}_pol_100_100_rew_{}_grad_pen_10.png'.format(disc_hid, rew),
        #     y_axis_lims=[-0.05, 1.05],
        #     plot_mean=False,
        #     column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
        #     constraints=constraints
        # )
#             plot_experiment_returns(
#                 '/scratch/gobi2/kamyar/oorl_rlkit/output/more-disc-iters-larger-z-range-easy-in-the-air-fetch-dac',
#                 'disc hid {}, pol 100-100, rew {}, grad_pen {}, num rew updates 128, num pol updates 65'.format(disc_hid, rew, grad_pen),
#                 '/h/kamyar/oorl_rlkit/plots/more_disc_iters_larger_z_range_easy_in_the_air_fetch_dac/disc_{}_{}_pol_100_100_rew_{}_grad_pen_{}_num_rew_upd_128_num_pol_upd_65.png'.format(disc_hid[0], disc_hid[1], rew, grad_pen),
#                 y_axis_lims=[-0.05, 1.05],
#                 plot_mean=False,
#                 column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#                 constraints=constraints
#             )
#             plot_experiment_returns(
#                 '/scratch/gobi2/kamyar/oorl_rlkit/output/with-layer-norm-more-disc-iters-larger-z-range-easy-in-the-air-fetch-dac',
#                 'with layer norm disc hid {}, pol 100-100, rew {}, grad_pen {}, num rew updates 128, num pol updates 65'.format(disc_hid, rew, grad_pen),
#                 '/h/kamyar/oorl_rlkit/plots/with_layer_norm_more_disc_iters_larger_z_range_easy_in_the_air_fetch_dac/with_layer_norm_disc_{}_{}_pol_100_100_rew_{}_grad_pen_{}_num_rew_upd_128_num_pol_upd_65.png'.format(disc_hid[0], disc_hid[1], rew, grad_pen),
#                 y_axis_lims=[-0.05, 1.05],
#                 plot_mean=False,
#                 column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#                 constraints=constraints
#             )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/larger-z-range-easy-in-the-air-fetch-dac',
                #     'disc hid {}, pol 100-100, rew {}, grad_pen {}, num rew updates {}'.format(disc_hid, rew, grad_pen, num_rew_upd),
                #     '/h/kamyar/oorl_rlkit/plots/larger_z_range_easy_in_the_air_fetch_dac/disc_{}_{}_pol_100_100_rew_{}_grad_pen_{}_num_rew_upd_{}.png'.format(disc_hid[0], disc_hid[1], rew, grad_pen, num_rew_upd),
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
                #     constraints=constraints
                # )
        
# for rew in [1.0, 2.0, 4.0]:
#     for grad_pen in [1.0, 5.0, 10.0]:
#         constraints = {
#             'policy_params.reward_scale': rew,
#             'grad_pen_weight': grad_pen,
#         }
#         plot_experiment_returns(
#             '/scratch/gobi2/kamyar/oorl_rlkit/output/online-1-1-larger-z-range-easy-in-the-air-fetch-dac',
#             'disc hid 100-64, pol 100-100, rew {}, grad_pen {}'.format(rew, grad_pen),
#             '/h/kamyar/oorl_rlkit/plots/online_1_1_larger_z_range_easy_in_the_air_fetch_dac/disc_100_64_pol_100_100_rew_{}_grad_pen_{}.png'.format(rew, grad_pen),
#             y_axis_lims=[-0.05, 1.05],
#             plot_mean=False,
#             column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#             constraints=constraints
#         )

#             # plot_experiment_returns(
#             #     '/scratch/gobi2/kamyar/oorl_rlkit/output/easy-in-the-air-fetch-dac',
#             #     'disc hid {}, pol 100-100, rew {}, grad_pen {}'.format(disc_hid, rew, grad_pen),
#             #     '/h/kamyar/oorl_rlkit/plots/easy_in_the_air_fetch_dac/disc_{}_{}_pol_100_100_rew_{}_grad_pen_{}.png'.format(disc_hid[0], disc_hid[1], rew, grad_pen),
#             #     y_axis_lims=[-0.05, 1.05],
#             #     plot_mean=False,
#             #     column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
#             #     constraints=constraints
#             # )
            # plot_experiment_returns(
            #     '/scratch/gobi2/kamyar/oorl_rlkit/output/larger-range-easy-in-the-air-fetch-dac',
            #     'disc hid {}, pol 100-100, rew {}, grad_pen {}'.format(disc_hid, rew, grad_pen),
            #     '/h/kamyar/oorl_rlkit/plots/larger_range_easy_in_the_air_fetch_dac/disc_{}_{}_pol_100_100_rew_{}_grad_pen_{}.png'.format(disc_hid[0], disc_hid[1], rew, grad_pen),
            #     y_axis_lims=[-0.05, 1.05],
            #     plot_mean=False,
            #     column_name=['Percent_Solved', 'Disc_Acc', 'Disc_Loss'],
            #     constraints=constraints
            # )


# for pol_size in [100, 256]:
#     for pol_layers in [3, 4]:
#         for test_batch_size in [256, 512]:
#             for z_dim in [50, 75, 100, 125]:
#                 for timestep_enc in [
#                     {'hidden_sizes': [50], 'output_size': 50},
#                     {'hidden_sizes': [100], 'output_size': 100}
#                 ]:
#             # for z_dim in [20, 40, 60, 80]:
#                     constraints = {
#                         'algo_params.policy_net_size': pol_size,
#                         'algo_params.policy_num_layers': pol_layers,
#                         'algo_params.test_batch_size_per_task': test_batch_size,
#                         'algo_params.np_params.z_dim': z_dim,
#                         'algo_params.np_params.traj_enc_params.timestep_enc_params': timestep_enc
#                     }
#                     # plot_experiment_returns(
#                     #     '/scratch/gobi2/kamyar/oorl_rlkit/output/first-try-few-shot-np-bc',
#                     #     'pol size {}, pol layers {}, test_batch_size {}, z_dim {}'.format(pol_size, pol_layers, test_batch_size, z_dim),
#                     #     '/h/kamyar/oorl_rlkit/plots/first_try_few_shot_np_bc/pol_size_{}_pol_layers_{}_test_batch_size_{}_z_dim_{}.png'.format(pol_size, pol_layers, test_batch_size, z_dim),
#                     #     y_axis_lims=[-0.05, 1.05],
#                     #     plot_mean=False,
#                     #     column_name=['Percent_Solved_meta_train', 'Percent_Solved_meta_test'],
#                     #     constraints=constraints
#                     # )
#                     if timestep_enc['hidden_sizes'][0] == 50:
#                         enc_dim = 50
#                     else:
#                         enc_dim = 100
#                     plot_experiment_returns(
#                         '/scratch/gobi2/kamyar/oorl_rlkit/output/second-try-few-shot-np-bc',
#                         'pol size {}, pol layers {}, test_batch_size {}, z_dim {}, timestep_enc {}'.format(pol_size, pol_layers, test_batch_size, z_dim, enc_dim),
#                         '/h/kamyar/oorl_rlkit/plots/second_try_few_shot_np_bc/pol_size_{}_pol_layers_{}_test_batch_size_{}_z_dim_{}_timestep_enc_{}.png'.format(pol_size, pol_layers, test_batch_size, z_dim, enc_dim),
#                         y_axis_lims=[-0.05, 1.05],
#                         plot_mean=False,
#                         column_name=['Percent_Solved_meta_train', 'Percent_Solved_meta_test'],
#                         constraints=constraints
#                     )


# os.makedirs('/h/kamyar/oorl_rlkit/plots/second_try_few_shot_np_bc/pol_arch', exist_ok=True)
# for pol_size in [100, 256]:
#     for pol_layers in [3, 4]:
#         constraints = {
#             'algo_params.policy_net_size': pol_size,
#             'algo_params.policy_num_layers': pol_layers,
#         }
#         plot_experiment_returns(
#             '/scratch/gobi2/kamyar/oorl_rlkit/output/second-try-few-shot-np-bc',
#             'pol size {}, pol layers {}'.format(pol_size, pol_layers),
#             '/h/kamyar/oorl_rlkit/plots/second_try_few_shot_np_bc/pol_arch/pol_size_{}_pol_layers_{}.png'.format(pol_size, pol_layers),
#             x_axis_lims=[0, 150],
#             y_axis_lims=[-0.05, 1.05],
#             plot_mean=False,
#             column_name=['Percent_Solved_meta_train', 'Percent_Solved_meta_test'],
#             constraints=constraints
#         )

# os.makedirs('/h/kamyar/oorl_rlkit/plots/second_try_few_shot_np_bc/test_batch_size', exist_ok=True)
# for test_batch_size in [256, 512]:
#     constraints = {
#         'algo_params.test_batch_size_per_task': test_batch_size,
#     }
#     plot_experiment_returns(
#         '/scratch/gobi2/kamyar/oorl_rlkit/output/second-try-few-shot-np-bc',
#         'test_batch_size {}'.format(test_batch_size),
#         '/h/kamyar/oorl_rlkit/plots/second_try_few_shot_np_bc/test_batch_size/test_batch_size_{}.png'.format(test_batch_size),
#         x_axis_lims=[0, 150],
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=['Percent_Solved_meta_train', 'Percent_Solved_meta_test'],
#         constraints=constraints
#     )

# os.makedirs('/h/kamyar/oorl_rlkit/plots/second_try_few_shot_np_bc/z_dim', exist_ok=True)
# for z_dim in [50, 75, 100, 125]:
#     constraints = {
#         'algo_params.np_params.z_dim': z_dim,
#     }
#     plot_experiment_returns(
#         '/scratch/gobi2/kamyar/oorl_rlkit/output/second-try-few-shot-np-bc',
#         'z_dim {}'.format(z_dim),
#         '/h/kamyar/oorl_rlkit/plots/second_try_few_shot_np_bc/z_dim/z_dim_{}.png'.format(z_dim),
#         x_axis_lims=[0, 150],
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=['Percent_Solved_meta_train', 'Percent_Solved_meta_test'],
#         constraints=constraints
#     )

# os.makedirs('/h/kamyar/oorl_rlkit/plots/second_try_few_shot_np_bc/timestep_enc', exist_ok=True)
# for timestep_enc in [
#         {'hidden_sizes': [50], 'output_size': 50},
#         {'hidden_sizes': [100], 'output_size': 100}
#     ]:
#     constraints = {
#         'algo_params.np_params.traj_enc_params.timestep_enc_params': timestep_enc
#     }
#     if timestep_enc['hidden_sizes'][0] == 50:
#         enc_dim = 50
#     else:
#         enc_dim = 100
#     plot_experiment_returns(
#         '/scratch/gobi2/kamyar/oorl_rlkit/output/second-try-few-shot-np-bc',
#         'timestep_enc {}'.format(enc_dim),
#         '/h/kamyar/oorl_rlkit/plots/second_try_few_shot_np_bc/timestep_enc/timestep_enc_{}.png'.format(enc_dim),
#         x_axis_lims=[0, 150],
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=['Percent_Solved_meta_train', 'Percent_Solved_meta_test'],
#         constraints=constraints
#     )


# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/gpu-normal-try-few-shot-np-bc',
#     'gpu normal',
#     '/h/kamyar/oorl_rlkit/plots/gpu_normal.png',
#     x_axis_lims=[0, 1000],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=['Percent_Solved_meta_train', 'Percent_Solved_meta_test'],
# )
# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/gpu-version-1-try-few-shot-np-bc',
#     'gpu version 1',
#     '/h/kamyar/oorl_rlkit/plots/gpu_version_1.png',
#     x_axis_lims=[0, 1000],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=['Percent_Solved_meta_train', 'Percent_Solved_meta_test'],
# )
# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/gpu-version-2-try-few-shot-np-bc',
#     'gpu version 2',
#     '/h/kamyar/oorl_rlkit/plots/gpu_version_2.png',
#     x_axis_lims=[0, 1000],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=['Percent_Solved_meta_train', 'Percent_Solved_meta_test'],
# )

# -------
# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/test-gpu-sum-dist-last-30-more-info-normal-try-few-shot-np-bc',
#     'gpu sum dist percent good reach',
#     '/h/kamyar/oorl_rlkit/plots/gpu_sum_dist_percent_good_reach.png',
#     x_axis_lims=[0, 600],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved_meta_train',
#         'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         'Percent_Good_Reach_meta_test',
#     ],
# )

# def plot_meta_train_meta_test(name):
#     N = 100
#     plot_experiment_returns(
#         '/scratch/gobi2/kamyar/oorl_rlkit/output/'+name,
#         'meta-train ' + name,
#         '/h/kamyar/oorl_rlkit/plots/{}_meta_train.png'.format(name),
#         x_axis_lims=[0, N],
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=[
#             'Percent_Solved_meta_train',
#             'Percent_Good_Reach_meta_train',
#         ],
#     )
#     plot_experiment_returns(
#         '/scratch/gobi2/kamyar/oorl_rlkit/output/'+name,
#         'meta-test {}'.format(name),
#         '/h/kamyar/oorl_rlkit/plots/{}_meta_test.png'.format(name),
#         x_axis_lims=[0, N],
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=[
#             'Percent_Solved_meta_test',
#             'Percent_Good_Reach_meta_test',
#         ],
#     )

# exp_names = ['correct-samples-np-bc', 'test-more-correct-samples-np-bc',
#             'pol-and-z-256-correct-samples-np-bc', 'pol-100-dim-5-6-layers-and-z-100-dim-correct-samples-np-bc',
#             'smaller-models-np-bc', 'even-smaller-models-np-bc',
#             'crazy-even-smaller-models-np-bc']
# for name in exp_names: plot_meta_train_meta_test(name)

# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/first-try-best-enc-size-models-np-airl',
#     'first_try_np_airl',
#     '/h/kamyar/oorl_rlkit/plots/first_try_np_airl.png',
#     x_axis_lims=[0, 1500],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         # 'Percent_Solved_meta_train',
#         'Percent_Good_Reach_meta_train',
#         # 'Percent_Solved_meta_test',
#         'Percent_Good_Reach_meta_test',
#     ],
# )



# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/please-work-fully-specified-train-single-task-np-airl',
#     'fucker percent solved',
#     '/h/kamyar/oorl_rlkit/plots/single_task_np_airl_percent_solved.png',
#     x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )
# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/please-work-fully-specified-train-single-task-np-airl',
#     'fucker good reach',
#     '/h/kamyar/oorl_rlkit/plots/single_task_np_airl_good_reach.png',
#     x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
        # # 'Percent_Solved_meta_train',
        # 'Percent_Good_Reach_meta_train',
        # # 'Percent_Solved_meta_test',
        # 'Percent_Good_Reach_meta_test',
#     ],
# )
# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/please-work-fully-specified-train-single-task-np-airl',
#     'fucker disc stats',
#     '/h/kamyar/oorl_rlkit/plots/single_task_np_airl_disc_stats.png',
#     x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Disc_Loss',
#         'Disc_Acc'
#     ],
# )
# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/please-work-fully-specified-train-single-task-np-airl',
#     'fucker other stats',
#     '/h/kamyar/oorl_rlkit/plots/single_task_np_airl_other_stats.png',
#     x_axis_lims=[0, 400],
#     # y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Avg_Min_Dist_to_Cor',
#         'Avg_Min_Cor_Z'
#     ],
# )

# for grad_pen_weight in [1.0, 5.0, 10.0]:
#     for rew_scale in [2.0, 4.0, 6.0, 8.0]:
#         for pol_num_layers in [3, 4]:
#             for disc_hid in [
#                 [100, 100],
#                 [256, 256],
#                 [100, 100, 100],
#             ]:
#                 constraints = {
#                     'algo_params.grad_pen_weight': grad_pen_weight,
#                     'algo_params.policy_params.reward_scale': rew_scale,
#                     'algo_params.policy_num_layers': pol_num_layers,
#                     'disc_hidden_sizes': disc_hid
#                 }
#                 name = 'pol_num_layers_{}_disc_hid_{}_rew_scale_{}_grad_pen_{}'.format(pol_num_layers, disc_hid, rew_scale, grad_pen_weight)
#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/please-work-fully-specified-train-single-task-np-airl',
#                     name,
#                     '/h/kamyar/oorl_rlkit/plots/single_task/{}.png'.format(name),
#                     x_axis_lims=[0, 400],
#                     y_axis_lims=[-0.05, 1.05],
#                     plot_mean=False,
#                     column_name=[
#                         'Percent_Solved_meta_train',
#                         'Percent_Good_Reach_meta_train',
#                         'Percent_Solved_meta_test',
#                         'Percent_Good_Reach_meta_test',
#                     ],
#                     constraints=constraints
#                 )


# for grad_pen_weight in [1.0, 5.0, 10.0]:
#     for rew_scale in [4.0, 6.0, 10.0, 20.0, 40.0, 80.0]:
#         for pol_num_layers in [3, 4]:
#             for disc_hid in [
#                 [100, 100],
#                 [256, 256],
#                 [100, 100, 100],
#             ]:
#                 constraints = {
#                     'algo_params.grad_pen_weight': grad_pen_weight,
#                     'algo_params.policy_params.reward_scale': rew_scale,
#                     'algo_params.policy_num_layers': pol_num_layers,
#                     'disc_hidden_sizes': disc_hid
#                 }
#                 name = 'pol_num_layers_{}_disc_hid_{}_rew_scale_{}_grad_pen_{}'.format(pol_num_layers, disc_hid, rew_scale, grad_pen_weight)
#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/np-airl-single-task-with-zero-z',
#                     # '/scratch/gobi2/kamyar/oorl_rlkit/output/new-grad-pen-more-rew-scale-search-fully-specified-train-single-task-np-airl',
#                     # '/scratch/gobi2/kamyar/oorl_rlkit/output/rerun-more-rew-scale-search-fully-specified-train-single-task-np-airl',
#                     name,
#                     '/h/kamyar/oorl_rlkit/plots/single_task_zero_z/{}.png'.format(name),
#                     x_axis_lims=[0, 400],
#                     y_axis_lims=[-0.05, 1.05],
#                     plot_mean=False,
#                     column_name=[
#                         'Percent_Solved_meta_train',
#                         'Percent_Good_Reach_meta_train',
#                         'Disc_Acc',
#                         'Disc_Loss',
#                         # 'Avg_Min_Dist_to_Cor_meta_train',
#                         # 'Avg_Min_Cor_Z_meta_train'
#                         # 'Percent_Solved_meta_test',
#                         # 'Percent_Good_Reach_meta_test',
#                     ],
#                     constraints=constraints
#                 )


# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/zero-np-bc',
#     'zero-np-bc',
#     '/h/kamyar/oorl_rlkit/plots/zero_np_bc.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Disc_Acc',
#         # 'Disc_Loss'
#     ],
# )

# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/zero-single-task-dac-not-traj-based',
#     'zero_single_task_dac_not_traj_based',
#     '/h/kamyar/oorl_rlkit/plots/zero_single_task_dac_not_traj_based.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved'
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Disc_Acc',
#         # 'Disc_Loss'
#     ],
# )

# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/zero-single-task-dac',
#     'zero-single-task-dac-not-traj-based',
#     '/h/kamyar/oorl_rlkit/plots/zero_single_task_dac_traj_based.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved'
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Disc_Acc',
#         # 'Disc_Loss'
#     ],
# )


# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/color-is-zero-np-bc',
#     'fucking color is zero-np-bc',
#     '/h/kamyar/oorl_rlkit/plots/fucking_color_is_zero_np_bc.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Disc_Acc',
#         # 'Disc_Loss'
#     ],
# )

# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/fucking-correct-zero-single-task-dac-not-traj-based',
#     'fucking zero_single_task_dac_not_traj_based',
#     '/h/kamyar/oorl_rlkit/plots/fucking_zero_single_task_dac_not_traj_based.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved'
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Disc_Acc',
#         # 'Disc_Loss'
#     ],
# )

# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/fucking-correct-zero-single-task-dac',
#     'fucking zero-single-task-dac-not-traj-based',
#     '/h/kamyar/oorl_rlkit/plots/fucking_zero_single_task_dac_traj_based.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved'
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Disc_Acc',
#         # 'Disc_Loss'
#     ],
# )


# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/fucking-correct-zero-single-task-dac-not-traj-based-256-disc',
#     'fucking 256 disc zero_single_task_dac_not_traj_based',
#     '/h/kamyar/oorl_rlkit/plots/fucking_256_disc_zero_single_task_dac_not_traj_based.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved',
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Solved_meta_test',
#         'Disc_Acc',
#         'Disc_Loss'
#     ],
# )

# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/fucking-correct-zero-single-task-dac-256-disc',
#     'fucking 256 disc zero-single-task-dac traj based',
#     '/h/kamyar/oorl_rlkit/plots/fucking_256_disc_zero_single_task_dac_traj_based.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved',
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Solved_meta_test',
#         'Disc_Acc',
#         'Disc_Loss'
#     ],
# )

# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/truly-final-zero-I-beg-of-you',
#     'truly-final-zero-I-beg-of-you',
#     '/h/kamyar/oorl_rlkit/plots/truly_final_zero_I_beg_of-you.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved',
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Disc_Acc',
#         # 'Disc_Loss'
#     ],
# )

# for rew in [10.0, 15.0, 20.0, 40.0, 80.0, 120.0]:
#     for grad_pen_weight in [5.0, 10.0, 15.0, 20.0]:
#         for pol_size in [100, 256]:
#             constraints = {
#                 'policy_params.reward_scale': rew,
#                 'gail_params.grad_pen_weight': grad_pen_weight,
#                 'policy_net_size': pol_size
#             }
#             save_path = '/h/kamyar/oorl_rlkit/plots/truly_final_zero/truly_final_zero_I_beg_of_you_rew_{}_grad_pen_{}_pol_size_{}.png'.format(rew, grad_pen_weight, pol_size)
#             # print(save_path)
#             plot_experiment_returns(
#                 '/scratch/gobi2/kamyar/oorl_rlkit/output/truly-final-zero-I-beg-of-you',
#                 'truly-final-zero-I-beg-of-you {}'.format(str(constraints)),
#                 save_path,
#                 x_axis_lims=[0, 100],
#                 y_axis_lims=[-0.05, 1.05],
#                 plot_mean=False,
#                 column_name=[
#                     'Percent_Solved',
#                     'Percent_Good_Reach',
#                     # 'Percent_Solved_meta_train',
#                     # 'Percent_Solved_meta_test',
#                     'Disc_Acc',
#                     'Disc_Loss'
#                 ],
#                 constraints=constraints
#             )

# for rew in [2.0, 4.0, 6.0, 8.0]:
#     for grad_pen_weight in [1.0, 5.0, 10.0]:
#         constraints = {
#             'policy_params.reward_scale': rew,
#             'gail_params.grad_pen_weight': grad_pen_weight
#         }
        # plot_experiment_returns(
        #     '/scratch/gobi2/kamyar/oorl_rlkit/output/fucking-correct-zero-single-task-dac-256-disc',
        #     'rew % d grad pen %.4f fucking 256 disc zero-single-task-dac traj based' % (rew, grad_pen_weight),
        #     '/h/kamyar/oorl_rlkit/plots/fucking_256_disc_zero_traj_based/fucking_256_disc_zero_single_task_dac_traj_based_rew_%d_%.4f_grad_pen.png' % (rew, grad_pen_weight),
        #     # x_axis_lims=[0, 400],
        #     y_axis_lims=[-0.05, 1.05],
        #     plot_mean=False,
        #     column_name=[
        #         'Percent_Solved',
        #         'Percent_Good_Reach',
        #         # 'Percent_Solved_meta_train',
        #         # 'Percent_Solved_meta_test',
        #         'Disc_Acc',
        #         'Disc_Loss'
        #     ],
        #     constraints=constraints
        # )



# for n_rew in [8, 16, 32, 65, 130]:
# for n_rew in [1, 2, 16, 32, 65]:
#     for grad_pen in [5.0, 10.0, 15.0]:
#         # for rew in [4.0, 6.0, 8.0, 10.0]:
#         for rew in [6.0, 8.0, 10.0]:
#             for pol_size in [128, 256]:
#                 constraints = {
#                     'gail_params.num_reward_updates': n_rew,
#                     'gail_params.grad_pen_weight': grad_pen,
#                     'policy_params.reward_scale': rew,
#                     'policy_net_size': pol_size 
#                 }
#                 name = 'disc_iters_{}_rew_{}_grad_pen_{}_pol_size_{}'.format(n_rew, rew, grad_pen, pol_size)
                
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/rew-upd-16-final-disc-noise-0p1-to-0-over-25-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_disc_noise_0p1_to_0_over_25_epochs_grad_clip_10_linear_10K_demos/{}_16.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         # 'Percent_Solved_meta_train',
                #         # 'Percent_Solved_meta_test',
                #         'Disc_Acc',
                #         'Disc_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/rew-upd-16-final-disc-noise-0p1-to-0-over-25-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_disc_noise_0p1_to_0_over_25_epochs_grad_clip_10_linear_10K_demos/rewards_{}_16.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #         'Disc_Avg_Grad_Norm_this_epoch',
                #         'Disc_Max_Grad_Norm_this_epoch'
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/rew-upd-32-final-disc-noise-0p1-to-0-over-25-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_disc_noise_0p1_to_0_over_25_epochs_grad_clip_10_linear_10K_demos/{}_32.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         # 'Percent_Solved_meta_train',
                #         # 'Percent_Solved_meta_test',
                #         'Disc_Acc',
                #         'Disc_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/rew-upd-32-final-disc-noise-0p1-to-0-over-25-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_disc_noise_0p1_to_0_over_25_epochs_grad_clip_10_linear_10K_demos/rewards_{}_32.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #         'Disc_Avg_Grad_Norm_this_epoch',
                #         'Disc_Max_Grad_Norm_this_epoch'
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/rew-upd-65-final-disc-noise-0p1-to-0-over-25-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_disc_noise_0p1_to_0_over_25_epochs_grad_clip_10_linear_10K_demos/{}_65.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         # 'Percent_Solved_meta_train',
                #         # 'Percent_Solved_meta_test',
                #         'Disc_Acc',
                #         'Disc_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/rew-upd-65-final-disc-noise-0p1-to-0-over-25-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_disc_noise_0p1_to_0_over_25_epochs_grad_clip_10_linear_10K_demos/rewards_{}_65.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #         'Disc_Avg_Grad_Norm_this_epoch',
                #         'Disc_Max_Grad_Norm_this_epoch'
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         # 'Percent_Solved_meta_train',
                #         # 'Percent_Solved_meta_test',
                #         'Disc_Acc',
                #         'Disc_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #         'Disc_Avg_Grad_Norm_this_epoch',
                #         'Disc_Max_Grad_Norm_this_epoch'
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/disc-lr-5en5-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/disc_lr_5en5_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/disc-lr-5en5-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/disc_lr_5en5_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #         'Disc_Avg_Grad_Norm_this_epoch',
                #         'Disc_Max_Grad_Norm_this_epoch'
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-clip-100-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/clip_100_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-clip-100-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/clip_100_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-clip-100-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/clip_100_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         # 'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         # 'Disc_Avg_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-final-clip-10-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/clip_10_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-final-clip-10-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/clip_10_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-final-clip-10-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-grad-clip-10-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/clip_10_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_grad_clip_10_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         # 'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         # 'Disc_Avg_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/ce-grad-clip-1-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/ce_grad_clip_1_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/ce-grad-clip-1-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/ce_grad_clip_1_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/ce-grad-clip-1-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/ce_grad_clip_1_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/ce-grad-clip-0p25-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/ce_grad_clip_0p25_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/ce-grad-clip-0p25-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/ce_grad_clip_0p25_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/ce-grad-clip-0p25-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/ce_grad_clip_0p25_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/ce-grad-clip-0p5-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/ce_grad_clip_0p5_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/ce-grad-clip-0p5-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/ce_grad_clip_0p5_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/ce-grad-clip-0p5-disc-lr-3en4-disc-noise-0p1-to-0-over-50-epochs-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/ce_grad_clip_0p5_disc_lr_3en4_disc_noise_0p1_to_0_over_50_epochs_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-2-1-iters-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_2_1_iters_clip_0p5_with_input_noise_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-2-1-iters-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_2_1_iters_clip_0p5_with_input_noise_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-2-1-iters-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_2_1_iters_clip_0p5_with_input_noise_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-1-1-iters-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_1_1_iters_clip_0p5_with_input_noise_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-1-1-iters-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_1_1_iters_clip_0p5_with_input_noise_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-1-1-iters-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_1_1_iters_clip_0p5_with_input_noise_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_clip_0p5_with_input_noise_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_clip_0p5_with_input_noise_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_clip_0p5_with_input_noise_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-clip-10-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_clip_10_with_input_noise_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-clip-10-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_clip_10_with_input_noise_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/big-gan-hypers-clip-10-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/big_gan_hypers_clip_10_with_input_noise_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/better-gating-smaller-custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/better_gating_smaller_custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/better-gating-smaller-custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/better_gating_smaller_custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/better-gating-smaller-custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/better_gating_smaller_custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-ema-policy-better-gating-smaller-custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/with_ema_policy_better_gating_smaller_custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-ema-policy-better-gating-smaller-custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/with_ema_policy_better_gating_smaller_custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-ema-policy-better-gating-smaller-custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/with_ema_policy_better_gating_smaller_custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/second-version-custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/second_version_custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/second-version-custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/second_version_custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/second-version-custom-single-color-fetch-disc-big-gan-hypers-ce-grad-clip-0p5-with-input-noise-linear-10K-demos',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/second_version_custom_single_color_fetch_disc_big_gan_hypers_ce_grad_clip_0p5_with_input_noise_linear_10K_demos/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-first-version-larger-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/correct_first_version_larger_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-first-version-larger-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/correct_first_version_larger_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-first-version-larger-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/correct_first_version_larger_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/first-version-smaller-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/first_version_smaller_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/first-version-smaller-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/first_version_smaller_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/first-version-smaller-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/first_version_smaller_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-ema-policy-first-version-smaller-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/with_ema_policy_first_version_smaller_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-ema-policy-first-version-smaller-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/with_ema_policy_first_version_smaller_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-ema-policy-first-version-smaller-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/with_ema_policy_first_version_smaller_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-second-version-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_second_version_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         'Disc_Acc',
                #         'Disc_CE_Loss'
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-second-version-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_second_version_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #     ],
                #     constraints=constraints
                # )
                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-second-version-custom-disc-no-noise-0p5-ce-grad-clip-10-gp-grad-clip',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/final_second_version_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip/grads_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Avg_CE_Grad_Norm_this_epoch',
                #         'Disc_Max_CE_Grad_Norm_this_epoch',
                #         'Disc_Avg_Total_Grad_Norm_this_epoch',
                #         'Disc_Max_Total_Grad_Norm_this_epoch',
                #     ],
                #     constraints=constraints
                # )


def plot_exp_v0(exp_name, title, x_axis_lims=None, constraints=None):
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}.png'.format(name)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=False,
        column_name=[
            'Percent_Solved',
            'Percent_Good_Reach',
            'Disc_Acc',
            'Disc_CE_Loss'
        ],
        constraints=constraints
    )
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'rewards_{}.png'.format(name)),
        x_axis_lims=x_axis_lims,
        plot_mean=False,
        column_name=[
            'Disc_Rew_Max',
            'Disc_Rew_Min',
            'Disc_Rew_Mean', 
            'Disc_Rew_Mean+Disc_Rew_Std',
            'Disc_Rew_Mean-Disc_Rew_Std'
        ],
        constraints=constraints
    )
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'grads_{}.png'.format(name)),
        x_axis_lims=x_axis_lims,
        plot_mean=False,
        column_name=[
            'Disc_Avg_CE_Grad_Norm_this_epoch',
            'Disc_Max_CE_Grad_Norm_this_epoch',
            'Disc_Avg_GP_Grad_Norm_this_epoch',
            'Disc_Max_GP_Grad_Norm_this_epoch',
        ],
        constraints=constraints
    )


def plot_meta_exp_v0(exp_name, title, x_axis_lims=None, constraints=None):
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}.png'.format(name)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=False,
        column_name=[
            'Percent_Solved_meta_train',
            'Percent_Good_Reach_meta_train',
            'Percent_Solved_meta_test',
            'Percent_Good_Reach_meta_test',
            'Disc_Acc',
            'Disc_CE_Loss'
        ],
        constraints=constraints
    )
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'rewards_{}.png'.format(name)),
        x_axis_lims=x_axis_lims,
        plot_mean=False,
        column_name=[
            'Disc_Rew_Max',
            'Disc_Rew_Min',
            'Disc_Rew_Mean', 
            'Disc_Rew_Mean+Disc_Rew_Std',
            'Disc_Rew_Mean-Disc_Rew_Std'
        ],
        constraints=constraints
    )
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'grads_{}.png'.format(name)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[0,4],
        plot_mean=False,
        column_name=[
            'Disc_Avg_CE_Grad_Norm_this_epoch',
            # 'Disc_Max_CE_Grad_Norm_this_epoch',
            'Enc_Avg_CE_Grad_Norm_this_epoch',
            # 'Enc_Max_CE_Grad_Norm_this_epoch',
            'Disc_Avg_GP_Grad_Norm_this_epoch',
            # 'Disc_Max_GP_Grad_Norm_this_epoch',
        ],
        constraints=constraints
    )


# for n_rew in [1, 2, 16, 32, 65]:
#     for grad_pen in [5.0, 10.0, 15.0]:
#         # for rew in [4.0, 6.0, 8.0, 10.0]:
#         for rew in [6.0, 8.0, 10.0]:
#             for pol_size in [128, 256]:
#                 # for ema_pol in [True, False]:
#                 constraints = {
#                     'gail_params.num_reward_updates': n_rew,
#                     'gail_params.grad_pen_weight': grad_pen,
#                     'policy_params.reward_scale': rew,
#                     'policy_net_size': pol_size,
#                     # 'policy_params.use_policy_as_ema_policy': ema_pol
#                 }
#                 name = 'disc_iters_{}_rew_{}_grad_pen_{}_pol_size_{}'.format(n_rew, rew, grad_pen, pol_size)
#                 # name = 'ema_policy_{}_disc_iters_{}_rew_{}_grad_pen_{}_pol_size_{}'.format(ema_pol, n_rew, rew, grad_pen, pol_size)
                
#                 # plot_exp_v0(
#                 #     'correct_first_version_larger_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                 #     name,
#                 #     x_axis_lims=[0,400],
#                 #     constraints=constraints
#                 # )
#                 # plot_exp_v0(
#                 #     'first_version_smaller_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                 #     name,
#                 #     x_axis_lims=[0,400],
#                 #     constraints=constraints
#                 # )
#                 # plot_exp_v0(
#                 #     'with_ema_policy_first_version_smaller_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                 #     name,
#                 #     x_axis_lims=[0,400],
#                 #     constraints=constraints
#                 # )
#                 # plot_exp_v0(
#                 #     'final_second_version_custom_disc_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                 #     name,
#                 #     x_axis_lims=[0,400],
#                 #     constraints=constraints
#                 # )

#                 plot_exp_v0(
#                     'correct_with_ema_policy_policy_uses_disc_gating_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                     name,
#                     x_axis_lims=[0,400],
#                     constraints=constraints
#                 )
#                 plot_exp_v0(
#                     'without_ema_policy_1_1_policy_uses_disc_gating_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                     name,
#                     x_axis_lims=[0,400],
#                     constraints=constraints
#                 )
#                 plot_exp_v0(
#                     'without_ema_policy_2_1_policy_uses_disc_gating_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                     name,
#                     x_axis_lims=[0,400],
#                     constraints=constraints
#                 )

# for gp in [0.5, 2.5, 10.0]:
#     for rew in [8.0, 10.0]:
#         for pol_size in [128, 256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'final_correct_other_gp_loss_with_ema_policy_1_1_policy_uses_disc_gating_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                 name,
#                 x_axis_lims=[0,400],
#                 constraints=constraints
#             )

# for gp in [5.0, 10.0, 15.0]:
#     for rew in [15.0]:
#         for pol_size in [128, 256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'SAC_temp_15_temps_with_ema_policy_1_1_policy_uses_disc_gating_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                 name,
#                 x_axis_lims=[0,400],
#                 constraints=constraints
#             )

# for gp in [5.0, 10.0, 15.0]:
#     for rew in [2.0]:
#         for pol_size in [128, 256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'SAC_temp_2_temps_with_ema_policy_1_1_policy_uses_disc_gating_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                 name,
#                 x_axis_lims=[0,400],
#                 constraints=constraints
#             )

# for gp in [5.0, 10.0, 15.0]:
#     for rew in [0.5, 1.0]:
#         for pol_size in [128, 256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'correct_lower_SAC_temp_with_ema_policy_1_1_policy_uses_disc_gating_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                 name,
#                 x_axis_lims=[0,400],
#                 constraints=constraints
#             )

# for gp in [5.0, 10.0, 15.0]:
#     for rew in [8.0, 10.0]:
#         for pol_size in [128, 256]:
#             for disc_mom in [-0.1, -0.25]:
#                 constraints = {
#                     'gail_params.grad_pen_weight': gp,
#                     'policy_params.reward_scale': rew,
#                     'policy_net_size': pol_size,
#                     'gail_params.disc_momentum': disc_mom
#                 }
#                 name = 'disc_mom_{}_rew_{}_gp_{}_pol_size_{}'.format(disc_mom, rew, gp, pol_size)
#                 plot_exp_v0(
#                     'disc_neg_mom_disc_lr_2en4_with_ema_policy_1_1_policy_uses_disc_gating_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                     name,
#                     x_axis_lims=[0,400],
#                     constraints=constraints
#                 )

# for gp in [5.0, 10.0, 15.0]:
#     for rew in [8.0, 10.0]:
#         for pol_size in [128, 256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'final_disc_lr_5en5_with_ema_policy_1_1_policy_uses_disc_gating_no_noise_0p5_ce_grad_clip_10_gp_grad_clip',
#                 name,
#                 x_axis_lims=[0,400],
#                 constraints=constraints
#             )


# for gp in [2.5, 5.0, 10.0]:
#     for rew in [2.0, 4.0, 6.0, 10.0]:
#         for pol_size in [256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'state_only_zero_fetch',
#                 name,
#                 x_axis_lims=[0,300],
#                 constraints=constraints
#             )


# for gp in [2.5, 5.0, 10.0]:
#     for rew in [15.0, 20.0, 25.0, 30.0]:
#         for pol_size in [256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'more_state_only_zero_fetch',
#                 name,
#                 x_axis_lims=[0,300],
#                 constraints=constraints
#             )


# for gp in [2.5, 5.0, 10.0]:
#     for rew in [2.0, 4.0, 6.0, 8.0]:
#         for pol_size in [256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'traj_based_zero_fetch',
#                 name,
#                 x_axis_lims=[0,300],
#                 constraints=constraints
#             )
#             plot_exp_v0(
#                 'correct_fully_traj_based_disc_zero_fetch',
#                 name,
#                 x_axis_lims=[0,300],
#                 constraints=constraints
#             )
#             plot_exp_v0(
#                 'subsampled_8_128_trajs_based_zero_fetch',
#                 name,
#                 x_axis_lims=[0,300],
#                 constraints=constraints
#             )
#             plot_exp_v0(
#                 'subsampled_16_64_trajs_based_zero_fetch',
#                 name,
#                 x_axis_lims=[0,300],
#                 constraints=constraints
#             )

# for gp in [2.5, 5.0, 10.0]:
#     for rew in [2.0, 4.0, 8.0, 10.0]:
#         constraints = {
#             'algo_params.grad_pen_weight': gp,
#             'policy_params.reward_scale': rew,
#         }
#         name = 'rew_{}_gp_{}'.format(rew, gp)
#         plot_exp_v0(
#             'test_np_airl_first_try',
#             name,
#             x_axis_lims=[0,100],
#             constraints=constraints
#         )d

# for gp in [2.5, 5.0, 10.0]:
#     for rew in [2.0, 4.0, 8.0, 10.0]:
#         for pol_size in [256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'sanity_check_standard',
#                 name,
#                 x_axis_lims=[0,400],
#                 constraints=constraints
#             )

# for gp in [0.5, 1.0]:
#     for rew in [2.0, 4.0, 8.0, 10.0]:
#         for pol_size in [256]:
#             constraints = {
#                 'gail_params.grad_pen_weight': gp,
#                 'policy_params.reward_scale': rew,
#                 'policy_net_size': pol_size
#             }
#             name = 'rew_{}_gp_{}_pol_size_{}'.format(rew, gp, pol_size)
#             plot_exp_v0(
#                 'sanity_check_standard_less_gp',
#                 name,
#                 x_axis_lims=[0,400],
#                 constraints=constraints
#             )

# for exp_batch_size in [32, 64, 128, 256]:
#     for gp in [2.5, 5.0]:
#         for rew in [2.0, 8.0]:
#             for pol_size in [256]:
#                 constraints = {
#                     'gail_params.grad_pen_weight': gp,
#                     'policy_params.reward_scale': rew,
#                     'policy_net_size': pol_size,
#                     'gail_params.policy_optim_batch_size_from_expert': exp_batch_size
#                 }
#                 name = 'rew_{}_gp_{}_pol_size_{}_exp_batch_size_{}'.format(rew, gp, pol_size, exp_batch_size)
#                 plot_exp_v0(
#                     'using_expert_demos_for_policy_traning',
#                     name,
#                     x_axis_lims=[0,400],
#                     constraints=constraints
#                 )

# for seed in [9783, 5914]:
#     for rew in [2.0, 4.0, 8.0, 10.0]:
#         for grad_pen in [1.0, 2.5, 5.0]:
#             constraints = {
#                 'algo_params.grad_pen_weight': grad_pen,
#                 'policy_params.reward_scale': rew,
#                 'seed': seed
#             }
#             name = 'rew_{}_gp_{}_seed_{}'.format(rew, grad_pen, seed)
#             plot_meta_exp_v0(
#                 'final_correct_np_airl_first_try_more_varied',
#                 name,
#                 x_axis_lims=[0,400],
#                 constraints=constraints
#             )

# for seed in [9783, 5914]:
#     constraints = {'seed': seed}
#     plot_meta_exp_v0(
#         'final_correct_np_airl_first_try_rew_10_gp_5',
#         'meta_rew_10_gp_5_seed_' + str(seed),
#         x_axis_lims=[0,300],
#         constraints=constraints
#     )

#     plot_meta_exp_v0(
#         'final_correct_np_airl_first_try_rew_5_gp_5',
#         'meta_rew_5_gp_5_seed_' + str(seed),
#         x_axis_lims=[0,300],
#         constraints=constraints
#     )

#     for z_dim in [25, 50]:
#         constraints['algo_params.np_params.z_dim'] = z_dim
#         name = 'meta_rew_5_gp_5_z_dim_%d_seed_%d' % (z_dim, seed)

#         plot_meta_exp_v0(
#             'env_arch_v1_128_conv_channels',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )

#         plot_meta_exp_v0(
#             'np_airl_gating_hid_size_64_z_dim_varied',
#             name,
#             x_axis_lims=[0,500],
#             constraints=constraints
#         )

#         plot_meta_exp_v0(
#             'env_arch_v1_64_conv_channels',
#             name,
#             x_axis_lims=[0,500],
#             constraints=constraints
#         )


# for seed in [9783, 5914]:
#     for rew in [2.0, 4.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)

        # plot_meta_exp_v0(
        #     'np_airl_enc_adam_0p9_gp_0p5',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'np_airl_enc_adam_0p9_gp_1',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'np_airl_enc_adam_0p9_gp_2p5',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'correct_np_airl_enc_adam_0p9_gp_1_gp_clip_100',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'np_airl_enc_adam_0p9_gp_1_pol_adam_0p5',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'np_airl_enc_adam_0p9_gp_1_pol_adam_0p9',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )



        # ------
# for seed in [9783, 5914]:
#     for rew in [2.0, 4.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
        # plot_meta_exp_v0(
        #     'np_airl_new_12_demos_10_each_sub_8_pol_adam_0_gp_0p5_ce_clip_5_gp_clip_10_z_dim_25',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'np_airl_new_12_demos_10_each_sub_8_pol_adam_0_gp_0p5_ce_clip_5_gp_clip_10_z_dim_50',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'np_airl_new_12_demos_10_each_sub_8_pol_adam_0p5_gp_0p5_ce_clip_5_gp_clip_10_z_dim_25',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'np_airl_new_12_demos_10_each_sub_8_pol_adam_0p5_gp_0p5_ce_clip_5_gp_clip_10_z_dim_50',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'correct_with_8_exp_samples_np_airl_new_12_demos_10_each_sub_8_pol_adam_0_gp_0p5_ce_clip_5_gp_clip_10_z_dim_50',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'correct_with_4_exp_samples_np_airl_new_12_demos_10_each_sub_8_pol_adam_0_gp_0p5_ce_clip_5_gp_clip_10_z_dim_50',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )

        # plot_meta_exp_v0(
        #     'np_airl_new_12_demos_10_each_sub_8_pol_adam_0p25_gp_0p5_all_clip_100_z_dim_50',
        #     name,
        #     x_axis_lims=[0,400],
        #     constraints=constraints
        # )


def plot_np_bc_results(exp_name, title, x_axis_lims=None, constraints=None):
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'meta_train_solved_{}.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=True,
        column_name=[
            'Percent_Solved_meta_train',
            # 'Percent_Good_Reach_meta_train',
            # 'Percent_Solved_meta_test',
            # 'Percent_Good_Reach_meta_test',
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'meta_test_solved_{}.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=True,
        column_name=[
            # 'Percent_Solved_meta_train',
            # 'Percent_Good_Reach_meta_train',
            'Percent_Solved_meta_test',
            # 'Percent_Good_Reach_meta_test',
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'good_reach_meta_train_{}.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=True,
        column_name=[
            # 'Percent_Solved_meta_train',
            'Percent_Good_Reach_meta_train',
            # 'Percent_Solved_meta_test',
            # 'Percent_Good_Reach_meta_test',
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'good_reach_meta_test_{}.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=True,
        column_name=[
            # 'Percent_Solved_meta_train',
            # 'Percent_Good_Reach_meta_train',
            # 'Percent_Solved_meta_test',
            'Percent_Good_Reach_meta_test',
        ],
        constraints=constraints
    )
    
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'all_meta_train_solved_{}.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=False,
        column_name=[
            'Percent_Solved_meta_train',
            # 'Percent_Good_Reach_meta_train',
            # 'Percent_Solved_meta_test',
            # 'Percent_Good_Reach_meta_test',
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'all_meta_test_solved_{}.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=False,
        column_name=[
            # 'Percent_Solved_meta_train',
            # 'Percent_Good_Reach_meta_train',
            'Percent_Solved_meta_test',
            # 'Percent_Good_Reach_meta_test',
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'all_good_reach_meta_train_{}.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=False,
        column_name=[
            # 'Percent_Solved_meta_train',
            'Percent_Good_Reach_meta_train',
            # 'Percent_Solved_meta_test',
            # 'Percent_Good_Reach_meta_test',
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, 'all_good_reach_meta_test_{}.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=False,
        column_name=[
            # 'Percent_Solved_meta_train',
            # 'Percent_Good_Reach_meta_train',
            # 'Percent_Solved_meta_test',
            'Percent_Good_Reach_meta_test',
        ],
        constraints=constraints
    )

# plot_np_bc_results(
#     'more_eval_z_dim_25_np_bc_new_gen_16_tasks_10_total_each_subsample_8',
#     '16 tasks'
# )

# plot_np_bc_results(
#     'more_seeds_0_eval_z_dim_25_np_bc_new_gen_16_tasks_10_total_each_subsample_8',
#     '16 tasks'
# )

# plot_np_bc_results(
#     'seeds_1_eval_z_dim_25_np_bc_new_gen_16_tasks_20_total_each_subsample_8',
#     '16 tasks 20 trajs each'
# )

# plot_np_bc_results(
#     'seeds_0_eval_z_dim_25_np_bc_new_gen_16_tasks_15_total_each_subsample_8',
#     '16 tasks 15 trajs each'
# )

# plot_np_bc_results(
#     'seeds_1_eval_z_dim_25_np_bc_new_gen_16_tasks_15_total_each_subsample_8',
#     '16 tasks 15 trajs each'
# )

# plot_np_bc_results(
#     'seeds_2_eval_z_dim_25_np_bc_new_gen_16_tasks_20_total_each_subsample_8',
#     '16 tasks 20 trajs each'
# )

# plot_np_bc_results(
#     'seeds_3_eval_z_dim_25_np_bc_new_gen_16_tasks_20_total_each_subsample_8',
#     '16 tasks 20 trajs each'
# )

# plot_np_bc_results(
#     'many_seeds_eval_z_dim_25_np_bc_new_gen_16_tasks_16_total_each_subsample_8',
#     '16 tasks 16 trajs each'
# )

# plot_np_bc_results(
#     'many_seeds_eval_z_dim_25_np_bc_new_gen_32_tasks_16_total_each_subsample_8',
#     '32 tasks 16 trajs each z 25'
# )

# plot_np_bc_results(
#     'many_seeds_eval_z_dim_50_np_bc_new_gen_32_tasks_16_total_each_subsample_8',
#     '32 tasks 16 trajs each z 50'
# )

# plot_np_bc_results(
#     'another_many_seeds_eval_z_dim_50_np_bc_new_gen_32_tasks_16_total_each_subsample_8',
#     '32 tasks 16 trajs each z 50'
# )

# plot_np_bc_results(
#     'another_many_seeds_eval_z_dim_25_np_bc_new_gen_32_tasks_16_total_each_subsample_8',
#     '32 tasks 16 trajs each z 25'
# )

# plot_np_bc_results(
#     'another_many_seeds_eval_z_dim_25_np_bc_new_gen_24_tasks_16_total_each_subsample_8',
#     '24 tasks 16 trajs each z 25'
# )

# plot_np_bc_results(
#     'another_many_seeds_eval_z_dim_25_np_bc_new_gen_16_tasks_16_total_each_subsample_8',
#     '16 tasks 16 trajs each z 25'
# )

# plot_np_bc_results(
#     'another_many_seeds_eval_z_dim_25_np_bc_new_gen_16_tasks_16_total_each_subsample_8',
#     '16 tasks 16 trajs each z 25'
# )

# plot_np_bc_results(
#     'another_many_seeds_eval_z_dim_25_np_bc_new_gen_20_tasks_16_total_each_subsample_8',
#     '20 tasks 16 trajs each z 25',
#     x_axis_lims=[0,25]
# )





# plot_np_bc_results(
#     'test_KL_0',
#     'testing_KL',
#     x_axis_lims=[0,40]
# )
# plot_np_bc_results(
#     'test_KL_2',
#     'testing_KL',
#     x_axis_lims=[0,40]
# )
# plot_np_bc_results(
#     'test_KL_4',
#     'testing_KL',
#     x_axis_lims=[0,70]
# )





# JACKFRUIT KOFTA

# for seed in [9783, 5914]:
#     for rew in [14.0, 16.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_14_16_from_expert_8',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_14_16_from_expert_8_disc_size_192',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_14_16_from_expert_8_disc_updates_2',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'not_state_only_actually_correct_reparam_sanity_check_rew_14_16_from_expert_8_disc_size_128',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         # plot_meta_exp_v0(
#         #     'reparam_sanity_check_disc_128_from_expert_0_rew_14_16',
#         #     name,
#         #     x_axis_lims=[0,100],
#         #     constraints=constraints
#         # )

# for seed in [9783, 5914]:
#     for rew in [10.0, 12.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_10_12_from_expert_8',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_10_12_from_expert_8_disc_size_192',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_10_12_from_expert_8_disc_updates_2',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'not_state_only_actually_correct_reparam_sanity_check_rew_10_12_from_expert_8_disc_size_128',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         # plot_meta_exp_v0(
#         #     'reparam_sanity_check_disc_128_from_expert_0_rew_10_12',
#         #     name,
#         #     x_axis_lims=[0,100],
#         #     constraints=constraints
#         # )

# for seed in [9783, 5914]:
#     for rew in [6.0, 8.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
#         plot_meta_exp_v0(
#             'np_airl_32_16_1_rew_6_8_pol_momentum_0p5',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_32_16_1_rew_6_8_disc_momentum_0p25',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_32_16_1_rew_6_8',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_32_16_1_rew_6_8_disc_192',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_6_8_from_expert_8',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_6_8_from_expert_8_disc_size_192',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_6_8_from_expert_8_disc_updates_2',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'not_state_only_actually_correct_reparam_sanity_check_rew_6_8_from_expert_8_disc_size_128',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'not_state_only_actually_correct_reparam_sanity_check_rew_6_8_from_expert_8_disc_size_128_disc_updates_2',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         # plot_meta_exp_v0(
#         #     'reparam_sanity_check_disc_128_from_expert_0_rew_6_8',
#         #     name,
#         #     x_axis_lims=[0,100],
#         #     constraints=constraints
#         # )

# for seed in [1553, 7972, 9361, 1901]:
#     constraints = {
#         'seed': seed,
#     }
#     name = 'seed_{}'.format(seed)
#     plot_np_bc_results(
#         'np_bc_KL_0_FINAL',
#         'np bc'
#     )
    # plot_np_bc_results(
    #     'np_bc_KL_0_FINAL',
    #     'np bc'
    # )
    # plot_np_bc_results(
    #     'np_bc_KL_0p05_FINAL',
    #     'np bc'
    # )
    # plot_np_bc_results(
    #     'np_bc_KL_0p1_FINAL',
    #     'np bc'
    # )
    # plot_np_bc_results(
    #     'np_bc_KL_0p15_FINAL',
    #     'np bc'
    # )
    # plot_np_bc_results(
    #     'np_bc_KL_0p2_FINAL',
    #     'np bc'
    # )
    # plot_np_bc_results(
    #     'np_bc_KL_0_FINAL_WITH_SAVING',
    #     'np bc'
    # )
    # plot_np_bc_results(
    #     'np_bc_KL_0p05_FINAL_WITH_SAVING',
    #     'np bc'
    # )
    # plot_np_bc_results(
    #     'np_bc_KL_0p1_FINAL_WITH_SAVING',
    #     'np bc'
    # )
    # plot_np_bc_results(
    #     'np_bc_KL_0p15_FINAL_WITH_SAVING',
    #     'np bc'
    # )
    # plot_np_bc_results(
    #     'np_bc_KL_0p2_FINAL_WITH_SAVING',
    #     'np bc'
    # )

for seed in [1553, 7972, 9361, 1901]:
    constraints = {
        'seed': seed,
    }
    name = 'seed_{}'.format(seed)
    
    plot_meta_exp_v0(
        'final_correct_state_only_np_airl_KL_0_disc_512_dim_rew_2_over_10_epochs',
        name,
        x_axis_lims=[0,200],
        constraints=constraints
    )
    plot_meta_exp_v0(
        'final_correct_state_only_np_airl_KL_0_disc_512_dim_rew_4_over_10_epochs',
        name,
        x_axis_lims=[0,200],
        constraints=constraints
    )
    plot_meta_exp_v0(
        'final_correct_state_only_np_airl_KL_0_disc_512_dim_rew_6_over_10_epochs',
        name,
        x_axis_lims=[0,200],
        constraints=constraints
    )
    plot_meta_exp_v0(
        'final_correct_state_only_np_airl_KL_0_disc_512_dim_rew_8_over_10_epochs',
        name,
        x_axis_lims=[0,200],
        constraints=constraints
    )

    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0p2_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_5_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0p15_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_5_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0p1_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_5_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0p05_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_5_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_5_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    
    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0p2_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_10_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0p15_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_10_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0p1_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_10_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0p05_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_10_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'correct_saving_np_airl_KL_0_disc_512_dim_rew_2_NO_TARGET_ANYTHING_over_10_epochs',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'np-airl-KL-0-disc-512-dim-rew-1-NO-TARGET-ANYTHING',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'np-airl-KL-0-disc-512-dim-rew-2-NO-TARGET-ANYTHING',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'np-airl-KL-0-disc-512-dim-rew-3-NO-TARGET-ANYTHING',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
    # plot_meta_exp_v0(
    #     'np_airl_KL_4_disc_512_dim_rew_4_NO_TARGET_ANYTHING',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )



# for seed in [1553, 7972, 9361, 1901]:
#     for rew in [4.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
#         plot_meta_exp_v0(
#             'np_airl_KL_0_disc_512_dim_rew_4_FINAL',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_KL_0_disc_256_dim_rew_4_FINAL',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
# for seed in [1553, 7972, 9361, 1901]:
#     for rew in [6.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
#         plot_meta_exp_v0(
#             'np_airl_KL_0_disc_512_dim_rew_6_FINAL',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_KL_0_disc_256_dim_rew_6_FINAL',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )


#         plot_meta_exp_v0(
#             'np_airl_32_16_1_rew_2_4_pol_momentum_0p5',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_32_16_1_rew_2_4_disc_momentum_0p25',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_32_16_1_rew_2_4',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_32_16_1_rew_2_4_disc_192',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )

#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_2_4_from_expert_8',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_2_4_from_expert_8_disc_size_192',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'actually_correct_reparam_sanity_check_rew_2_4_from_expert_8_disc_updates_2',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'not_state_only_actually_correct_reparam_sanity_check_rew_2_4_from_expert_8_disc_size_128',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'not_state_only_actually_correct_reparam_sanity_check_rew_2_4_from_expert_8_disc_size_128_disc_updates_2',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         # plot_meta_exp_v0(
#         #     'reparam_sanity_check_disc_128_from_expert_8',
#         #     name,
#         #     x_axis_lims=[0,100],
#         #     constraints=constraints
#         # )




# MASALA

def plot_transfer_disc_training_results(exp_name, title, x_axis_lims=None, constraints=None):
    # constraints = None
    
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_disc_stats.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=False,
        column_name=[
            'Disc_CE_Loss',
            'Disc_Acc',
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_r_stats.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=None,
        plot_mean=False,
        column_name=[
            'Avg_R',
            'Avg_R+Std_R',
            'Avg_R-Std_R',
            'Max_R',
            'Min_R'
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_V_s_stats.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=None,
        plot_mean=False,
        column_name=[
            'Avg_V_s',
            'Avg_V_s+Std_V_s',
            'Avg_V_s-Std_V_s',
            'Max_V_s',
            'Min_V_s'
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_shaping_stats.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=None,
        plot_mean=False,
        column_name=[
            'Avg_Shaping',
            'Avg_Shaping+Std_Shaping',
            'Avg_Shaping-Std_Shaping',
            'Max_Shaping',
            'Min_Shaping'
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_f_stats.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=None,
        plot_mean=False,
        column_name=[
            'Avg_f',
            'Avg_f+Std_f',
            'Avg_f-Std_f',
            'Max_f',
            'Min_f'
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_rew_stats.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=None,
        plot_mean=False,
        column_name=[
            'Disc_Rew_Mean',
            'Disc_Rew_Mean+Disc_Rew_Std',
            'Disc_Rew_Mean-Disc_Rew_Std',
            'Disc_Rew_Max',
            'Disc_Rew_Min'
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_logprob_for_exp.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=None,
        plot_mean=False,
        column_name=[
            'Avg_Pol_LogProb_for_Exp',
            'Avg_Pol_LogProb_for_Exp+Std_Pol_LogProb_for_Exp',
            'Avg_Pol_LogProb_for_Exp-Std_Pol_LogProb_for_Exp',
            'Max_Pol_LogProb_for_Exp',
            'Min_Pol_LogProb_for_Exp'
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_logprob_for_pol.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=None,
        plot_mean=False,
        column_name=[
            'Avg_Pol_LogProb_for_Pol',
            'Avg_Pol_LogProb_for_Pol+Std_Pol_LogProb_for_Pol',
            'Avg_Pol_LogProb_for_Pol-Std_Pol_LogProb_for_Pol',
            'Max_Pol_LogProb_for_Pol',
            'Min_Pol_LogProb_for_Pol'
        ],
        constraints=constraints
    )
    
    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_solve_stats.png'.format(title)),
        x_axis_lims=x_axis_lims,
        y_axis_lims=[-0.05, 1.05],
        plot_mean=False,
        column_name=[
            'Percent_Solved_meta_train',
            'Percent_Good_Reach_meta_train',
            'Percent_Solved_meta_test',
            'Percent_Good_Reach_meta_test',
        ],
        constraints=constraints
    )

    plot_experiment_returns(
        os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', exp_name.replace('_', '-')),
        title,
        os.path.join('/h/kamyar/oorl_rlkit/plots', exp_name, '{}_grads.png'.format(title)),
        x_axis_lims=x_axis_lims,
        plot_mean=False,
        column_name=[
            'Disc_Avg_CE_Grad_Norm_this_epoch',
            'Disc_Max_CE_Grad_Norm_this_epoch',
            'Enc_Avg_CE_Grad_Norm_this_epoch',
            'Enc_Max_CE_Grad_Norm_this_epoch',
            'Disc_Avg_GP_Grad_Norm_this_epoch',
            'Disc_Max_GP_Grad_Norm_this_epoch'
        ],
        constraints=constraints
    )



# for rew in [2.0, 4.0]:
#     constraints = {
#         'policy_params.reward_scale': rew
#     }
#     name = 'rew_{}'.format(rew)
#     plot_transfer_disc_training_results(
#         'final_correct_pol_log_prob_clipped_Adam_0_L2_reg_logits_0p005_rew_2_4',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )
#     plot_transfer_disc_training_results(
#         'final_correct_pol_log_prob_clipped_Adam_0_L2_reg_logits_0_rew_2_4',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )
# for rew in [6.0, 8.0]:
#     constraints = {
#         'policy_params.reward_scale': rew
#     }
#     name = 'rew_{}'.format(rew)
#     plot_transfer_disc_training_results(
#         'final_correct_pol_log_prob_clipped_Adam_0_L2_reg_logits_0p005_rew_6_8',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )
#     plot_transfer_disc_training_results(
#         'final_correct_pol_log_prob_clipped_Adam_0_L2_reg_logits_0_rew_6_8',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )


# for seed in [9783, 5914]:
#     for rew in [2.0, 4.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
#         # name = 'rew_{}'.format(rew)
#         plot_meta_exp_v0(
#             'correct_np_airl_32_16_1_rew_2_4_disc_3_layer_256',
#             name,
#             x_axis_lims=[0,200],
#             constraints=constraints
#         )

# for seed in [9783, 5914]:
# for rew in [2.0, 4.0]:
#     constraints = {
#         'policy_params.reward_scale': rew
#     }
#     name = 'rew_{}'.format(rew)
#     plot_transfer_disc_training_results(
#         'r_3_512_V_3_512_disc_Adam_0p9_L2_reg_logits_0p005_rew_2_4',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )
    # plot_transfer_disc_training_results(
    #     'r_3_512_V_3_512_disc_Adam_0p9_L2_reg_logits_0p01_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_512_V_3_512_disc_Adam_0p9_L2_reg_logits_0p05_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )

    # GOOD ONES!!!!!
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0p5_L2_reg_logits_0p005_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0p25_L2_reg_logits_0p005_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0_L2_reg_logits_0_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0_L2_reg_logits_0p005_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0_L2_reg_logits_0p01_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0_L2_reg_logits_0p05_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0p05_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0p01_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0p005_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'fixed_grad_pen_r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0_rew_2_4',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    
    


# for rew in [8.0, 10.0]:
#     constraints = {
#         'policy_params.reward_scale': rew
#     }
#     name = 'rew_{}'.format(rew)
#     plot_transfer_disc_training_results(
#         'r_3_512_V_3_512_disc_Adam_0p9_L2_reg_logits_0p005_rew_8_10',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )
#     plot_transfer_disc_training_results(
#         'r_3_512_V_3_512_disc_Adam_0p9_L2_reg_logits_0p01_rew_8_10',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )
#     plot_transfer_disc_training_results(
#         'r_3_512_V_3_512_disc_Adam_0p9_L2_reg_logits_0p05_rew_8_10',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )

# for rew in [14.0, 16.0]:
#     constraints = {
#         'policy_params.reward_scale': rew
#     }
#     name = 'rew_{}'.format(rew)
#     plot_transfer_disc_training_results(
#         'r_3_512_V_3_512_disc_Adam_0p9_L2_reg_logits_0p005_rew_14_16',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )
#     plot_transfer_disc_training_results(
#         'r_3_512_V_3_512_disc_Adam_0p9_L2_reg_logits_0p01_rew_14_16',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )
#     plot_transfer_disc_training_results(
#         'r_3_512_V_3_512_disc_Adam_0p9_L2_reg_logits_0p05_rew_14_16',
#         name,
#         x_axis_lims=[0,100],
#         constraints=constraints
#     )
    # plot_transfer_disc_training_results(
    #     'r_3_128_V_3_256_disc_Adam_0p9_L2_reg_logits_0p1',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0p005',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_128_V_3_256_disc_Adam_0p9_L2_reg_logits_0p05',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0p001',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_128_V_3_256_disc_Adam_0p9_L2_reg_logits_0p01',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_128_V_3_256_disc_Adam_0p9_L2_reg_logits_0p5',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0p5',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0p1',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0p05',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )
    # plot_transfer_disc_training_results(
    #     'r_3_256_V_3_256_disc_Adam_0p9_L2_reg_logits_0p01',
    #     name,
    #     x_axis_lims=[0,100],
    #     constraints=constraints
    # )

        # np_airl stuff
    # plot_meta_exp_v0(
    #     'correct_np_airl_32_16_1_rew_2_4_disc_3_layer_128',
    #     name,
    #     x_axis_lims=[0,200],
    #     constraints=constraints
    # )
        # plot_meta_exp_v0(
        #     'correct_np_airl_32_16_1_rew_2_4_disc_3_layer_192',
        #     name,
        #     x_axis_lims=[0,200],
        #     constraints=constraints
        # )
        # plot_meta_exp_v0(
        #     'correct_np_airl_32_16_1_rew_2_4_disc_2_layer_256',
        #     name,
        #     x_axis_lims=[0,200],
        #     constraints=constraints
        # )

        # # transfer stuff
        # plot_transfer_disc_training_results(
        #     'r_64_V_128_disc_Adam_0p9_L2_reg_weight_1_ratio_1_to_10',
        #     name,
        #     x_axis_lims=[0,100],
        #     constraints=constraints
        # )
        # plot_transfer_disc_training_results(
        #     'r_64_V_128_disc_Adam_0p9_L2_reg_weight_0p5_ratio_1_to_10',
        #     name,
        #     x_axis_lims=[0,100],
        #     constraints=constraints
        # )
        # plot_transfer_disc_training_results(
        #     'r_64_V_128_disc_Adam_0p9_L2_reg_weight_0p1_ratio_1_to_10',
        #     name,
        #     x_axis_lims=[0,100],
        #     constraints=constraints
        # )
        # plot_transfer_disc_training_results(
        #     'r_64_V_128_disc_Adam_0p9_L2_reg_weight_5_ratio_1_to_10',
        #     name,
        #     x_axis_lims=[0,100],
        #     constraints=constraints
        # )
        # plot_transfer_disc_training_results(
        #     'r_64_V_128_disc_Adam_0p9_L2_reg_weight_1_ratio_1_to_1',
        #     name,
        #     x_axis_lims=[0,100],
        #     constraints=constraints
        # )
        # plot_transfer_disc_training_results(
        #     'r_64_V_128_disc_Adam_0p9_L2_reg_logits_0p1',
        #     name,
        #     x_axis_lims=[0,100],
        #     constraints=constraints
        # )
        # plot_transfer_disc_training_results(
        #     'r_64_V_128_disc_Adam_0p9_L2_reg_logits_0p5',
        #     name,
        #     x_axis_lims=[0,100],
        #     constraints=constraints
        # )
        # plot_transfer_disc_training_results(
        #     'r_64_V_128_disc_Adam_0p9_L2_reg_logits_1',
        #     name,
        #     x_axis_lims=[0,100],
        #     constraints=constraints
        # )
        # plot_transfer_disc_training_results(
        #     'r_64_V_128_disc_Adam_0p9_L2_reg_logits_5',
        #     name,
        #     x_axis_lims=[0,100],
        #     constraints=constraints
        # )


# PHO


# plot_np_bc_results(
#     'test_few_shot_on_subsample_1_no_KL',
#     'np_bc few shot no KL'
# )

# for kl in [0.0, 0.25]:
#     constraints = {'algo_params.max_KL_beta': kl}
#     # plot_np_bc_results(
#     #     'test_few_shot_with_bunch_of_KLs_with_cont_1_eval',
#     #     'np_bc 1 context eval KL %f' % kl,
#     #     constraints=constraints
#     # )
#     # plot_np_bc_results(
#     #     'test_few_shot_with_bunch_of_KLs_with_cont_3_eval',
#     #     'np_bc 3 context eval KL %f' % kl,
#     #     constraints=constraints
#     # )
#     plot_np_bc_results(
#         'test_np_bc_on_fixed_colors',
#         'np_bc 1 context eval KL %f' % kl,
#         constraints=constraints
#     )
#     plot_np_bc_results(
#         'test_np_bc_on_fixed_colors_0p5_eval_on_1_context',
#         'np_bc 1 context eval KL %f' % kl,
#         constraints=constraints
#     )
#     plot_np_bc_results(
#         'test_np_bc_on_fixed_colors_0p5_eval_on_3_context',
#         'np_bc 3 context eval KL %f' % kl,
#         constraints=constraints
#     )


# for seed in [9783, 5914]:
#     for rew in [2.0, 4.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
#         plot_meta_exp_v0(
#             'reparam_sanity_check_disc_128_from_expert_0_rew_2_4_slower_disc',
#             name,
#             x_axis_lims=[0,100],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'reparam_sanity_check_disc_128_from_expert_8_rew_2_4_slower_disc',
#             name,
#             x_axis_lims=[0,100],
#             constraints=constraints
#         )

# for seed in [9783, 5914]:
#     for rew in [6.0, 8.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
#         plot_meta_exp_v0(
#             'reparam_sanity_check_disc_128_from_expert_0_rew_6_8_slower_disc',
#             name,
#             x_axis_lims=[0,100],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'reparam_sanity_check_disc_128_from_expert_8_rew_6_8_slower_disc',
#             name,
#             x_axis_lims=[0,100],
#             constraints=constraints
#         )









# for seed in [9783, 5914]:
#     for rew in [2.0, 4.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)

#         plot_meta_exp_v0(
#             'demos_20_0_expert_disc_128_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_20_4_expert_disc_128_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_20_8_expert_disc_128_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_20_0_expert_disc_96_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_20_4_expert_disc_96_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'correct_state_only_disc_128_from_expert_8_pol_beta_0p25_demos_32_16_1',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_20_8_expert_disc_96_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_16_0_expert_disc_128_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_16_4_expert_disc_128_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_16_8_expert_disc_128_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_16_0_expert_disc_96_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_16_4_expert_disc_96_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'demos_16_8_expert_disc_96_pol_beta_0p25_np_airl',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'correct_state_only_disc_128_from_expert_0_pol_beta_0p25_demos_32_16_1',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'correct_state_only_disc_128_from_expert_8_pol_beta_0p25_demos_32_16_1',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'correct_state_only_disc_128_from_expert_4_pol_beta_0p25_demos_32_16_1',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )

#         plot_meta_exp_v0(
#             'no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_64',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'lower_disc_lr_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_64',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'lower_disc_lr_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_96',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_96',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_128',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'lower_disc_lr_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_128',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )

#         plot_meta_exp_v0(
#             'no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_160',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'lower_disc_lr_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_160',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'final_lower_disc_lr_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_192',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'final_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_192',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p9_z_dim_25_rew_2_and_4_disc_128',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'lower_disc_lr_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p9_z_dim_25_rew_2_and_4_disc_128',
#             name,
#             x_axis_lims=[0,300],
#             constraints=constraints
#         )

        # plot_meta_exp_v0(
        #     'no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_160',
        #     name,
        #     x_axis_lims=[0,300],
        #     constraints=constraints
        # )
        # plot_meta_exp_v0(
        #     'lower_disc_lr_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_160',
        #     name,
        #     x_axis_lims=[0,300],
        #     constraints=constraints
        # )
        # plot_meta_exp_v0(
        #     'final_lower_disc_lr_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_192',
        #     name,
        #     x_axis_lims=[0,300],
        #     constraints=constraints
        # )
        # plot_meta_exp_v0(
        #     'final_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_192',
        #     name,
        #     x_axis_lims=[0,300],
        #     constraints=constraints
        # )
        # plot_meta_exp_v0(
        #     'no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p9_z_dim_25_rew_2_and_4_disc_128',
        #     name,
        #     x_axis_lims=[0,300],
        #     constraints=constraints
        # )
        # plot_meta_exp_v0(
        #     'lower_disc_lr_no_clip_np_airl_another_32_demos_16_each_sub_8_pol_adam_0p9_z_dim_25_rew_2_and_4_disc_128',
        #     name,
        #     x_axis_lims=[0,300],
        #     constraints=constraints
        # )

        
#         plot_meta_exp_v0(
#             'np_airl_new_12_demos_10_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_64',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_new_12_demos_10_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_96',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'no_clip_np_airl_new_12_demos_10_each_sub_8_pol_adam_0p25_z_dim_25_rew_2_and_4_disc_96',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )

# for seed in [9783, 5914]:
#     for rew in [0.5, 1.0]:
#         constraints = {
#             'seed': seed,
#             'policy_params.reward_scale': rew
#         }
#         name = 'rew_{}_seed_{}'.format(rew, seed)
#         plot_meta_exp_v0(
#             'np_airl_new_12_demos_10_each_sub_8_pol_adam_0p25_z_dim_25_rew_0p5_and_1_disc_64',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'np_airl_new_12_demos_10_each_sub_8_pol_adam_0p25_z_dim_25_rew_0p5_and_1_disc_96',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )
#         plot_meta_exp_v0(
#             'no_clip_np_airl_new_12_demos_10_each_sub_8_pol_adam_0p25_z_dim_25_rew_0p5_and_1_disc_96',
#             name,
#             x_axis_lims=[0,400],
#             constraints=constraints
#         )


# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'anoter_z_dim_50_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'z_dim_50_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'z_dim_50_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_train_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'anoter_z_dim_50_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'z_dim_50_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'z_dim_50_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_test_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_train_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_test_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'anoter_z_dim_12_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'anoter_z_dim_12_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'anoter_z_dim_12_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_train_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'anoter_z_dim_12_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'anoter_z_dim_12_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'anoter_z_dim_12_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_test_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'anoter_z_dim_12_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'anoter_z_dim_12_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'anoter_z_dim_12_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'all_plots_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_train_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_test_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'all_plots_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'gate_64_conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'gate_64_conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'gate_64_conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_train_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'gate_64_conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'gate_64_conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'gate_64_conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_test_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'gate_64_conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'gate_64_conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'gate_64_conv_64_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'all_plots_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'gate_32_conv_32_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'gate_32_conv_32_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'gate_32_conv_32_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_train_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'gate_32_conv_32_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'gate_32_conv_32_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'gate_32_conv_32_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'meta_test_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=True,
#     column_name=[
#         # 'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

# plot_experiment_returns(
#     os.path.join('/scratch/gobi2/kamyar/oorl_rlkit/output', 'gate_32_conv_32_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8'.replace('_', '-')),
#     'gate_32_conv_32_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8',
#     os.path.join('/h/kamyar/oorl_rlkit/plots', 'gate_32_conv_32_anoter_z_dim_25_np_bc_new_gen_12_tasks_10_total_each_subsample_8', 'all_plots_{}.png'.format(name)),
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved_meta_train',
#         # 'Percent_Good_Reach_meta_train',
#         'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )

            # plot_exp_v0(
            #     'final_wrap_absorbing_standard',
            #     name,
            #     x_axis_lims=[0,100],
            #     constraints=constraints
            # )
            # plot_exp_v0(
            #     'correct_final_wrap_absorbing_standard_traj_based',
            #     name,
            #     x_axis_lims=[0,100],
            #     constraints=constraints
            # )
            # plot_exp_v0(
            #     'final_wrap_absorbing_state_only',
            #     name,
            #     x_axis_lims=[0,100],
            #     constraints=constraints
            # )
            # plot_exp_v0(
            #     'final_wrap_absorbing_state_only_traj_based',
            #     name,
            #     x_axis_lims=[0,100],
            #     constraints=constraints
            # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-correct-relu-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/relu_grind_zero/{}.png'.format(name),
#                 #     # x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-unscaled-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/unscaled_grind_zero/{}.png'.format(name),
#                 #     # x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/grind_zero/{}.png'.format(name),
#                 #     # x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/resnet-disc-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/resnet_disc_grind_zero/{}.png'.format(name),
#                 #     # x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/batchnorm_disc_grind_zero/{}.png'.format(name),
#                 #     # x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/cpu_relu_batchnorm_disc_grind_zero/{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/cpu_relu_batchnorm_disc_grind_zero/disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/with_target_disc_cpu_relu_batchnorm_disc_grind_zero/{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-correct-scale-0p9-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/scale_0p9_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/final-correct-scale-0p9-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/scale_0p9_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/scale_0p9_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     name,
#                 #     '/h/kamyar/oorl_rlkit/plots/scale_0p9_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/scale_0p9_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/scale_0p9_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/less-rew-iters-disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/less-rew-iters-disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/clamped-rew-less-rew-iters-disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/clamped_4_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/clamped-rew-less-rew-iters-disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/clamped_4_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-lr-1en4-clamped-rew-less-rew-iters-disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/lr_1en4_clamped_4_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-lr-1en4-clamped-rew-less-rew-iters-disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/lr_1en4_clamped_4_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/omfg-correct-lr-1en4-clamped-rew-less-rew-iters-disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/omfg_lr_1en4_clamped_4_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/omfg-correct-lr-1en4-clamped-rew-less-rew-iters-disc-64-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 64 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/omfg_lr_1en4_clamped_4_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_64_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/disc-128-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/disc-128-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/disc-256-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 256 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/disc_256_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/disc-256-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 256 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/disc_256_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-grad-log-rew-updates-65-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_65_disc_128_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-grad-log-rew-updates-65-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_65_disc_128_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min',
#                 #         'Disc_Avg_Grad_Norm_this_epoch',
#                 #         'Disc_Max_Grad_Norm_this_epoch'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-grad-log-rew-updates-32-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_32_disc_128_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-grad-log-rew-updates-32-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_32_disc_128_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min',
#                 #         'Disc_Avg_Grad_Norm_this_epoch',
#                 #         'Disc_Max_Grad_Norm_this_epoch'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-grad-log-rew-updates-16-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_16_disc_128_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/with-grad-log-rew-updates-16-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_16_disc_128_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min',
#                 #         'Disc_Avg_Grad_Norm_this_epoch',
#                 #         'Disc_Max_Grad_Norm_this_epoch'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-5-rew-updates-65-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/grad_clip_5_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_65_disc_128_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-5-rew-updates-65-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/grad_clip_5_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_65_disc_128_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min',
#                 #         'Disc_Avg_Grad_Norm_this_epoch',
#                 #         'Disc_Max_Grad_Norm_this_epoch'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-5-rew-updates-32-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/grad_clip_5_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_32_disc_128_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-5-rew-updates-32-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/grad_clip_5_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_32_disc_128_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min',
#                 #         'Disc_Avg_Grad_Norm_this_epoch',
#                 #         'Disc_Max_Grad_Norm_this_epoch'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-5-rew-updates-16-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/grad_clip_5_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_16_disc_128_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-5-rew-updates-16-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/grad_clip_5_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_16_disc_128_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min',
#                 #         'Disc_Avg_Grad_Norm_this_epoch',
#                 #         'Disc_Max_Grad_Norm_this_epoch'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-10-rew-updates-all-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/grad_clip_10_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     y_axis_lims=[-0.05, 1.05],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Percent_Solved',
#                 #         'Percent_Good_Reach',
#                 #         # 'Percent_Solved_meta_train',
#                 #         # 'Percent_Solved_meta_test',
#                 #         'Disc_Acc',
#                 #         'Disc_Loss'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 # plot_experiment_returns(
#                 #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-10-rew-updates-all-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                 #     'disc 128 '+name,
#                 #     '/h/kamyar/oorl_rlkit/plots/grad_clip_10_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_disc_rewards_{}.png'.format(name),
#                 #     x_axis_lims=[0, 100],
#                 #     plot_mean=False,
#                 #     column_name=[
#                 #         'Disc_Rew_Max',
#                 #         'Disc_Rew_Min',
#                 #         'Disc_Avg_Grad_Norm_this_epoch',
#                 #         'Disc_Max_Grad_Norm_this_epoch'
#                 #     ],
#                 #     constraints=constraints
#                 # )

#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/no-bn-grad-clip-10-rew-updates-65-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                     'disc 128 '+name,
#                     '/h/kamyar/oorl_rlkit/plots/no_bn_grad_clip_10_rew_updates_65_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_{}.png'.format(name),
#                     x_axis_lims=[0, 100],
#                     y_axis_lims=[-0.05, 1.05],
#                     plot_mean=False,
#                     column_name=[
#                         'Percent_Solved',
#                         'Percent_Good_Reach',
#                         # 'Percent_Solved_meta_train',
#                         # 'Percent_Solved_meta_test',
#                         'Disc_Acc',
#                         'Disc_Loss'
#                     ],
#                     constraints=constraints
#                 )

#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/no-bn-grad-clip-10-rew-updates-65-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                     'disc 128 '+name,
#                     '/h/kamyar/oorl_rlkit/plots/no_bn_grad_clip_10_rew_updates_65_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_disc_rewards_{}.png'.format(name),
#                     x_axis_lims=[0, 100],
#                     plot_mean=False,
#                     column_name=[
#                         'Disc_Rew_Max',
#                         'Disc_Rew_Min',
#                         'Disc_Avg_Grad_Norm_this_epoch',
#                         'Disc_Max_Grad_Norm_this_epoch'
#                     ],
#                     constraints=constraints
#                 )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-1-rew-updates-all-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/grad_clip_1_rew_updates_all_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     y_axis_lims=[-0.05, 1.05],
                #     plot_mean=False,
                #     column_name=[
                #         'Percent_Solved',
                #         'Percent_Good_Reach',
                #         # 'Percent_Solved_meta_train',
                #         # 'Percent_Solved_meta_test',
                #         'Disc_Acc',
                #         'Disc_Loss'
                #     ],
                #     constraints=constraints
                # )

                # plot_experiment_returns(
                #     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-1-rew-updates-all-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
                #     'disc 128 '+name,
                #     '/h/kamyar/oorl_rlkit/plots/grad_clip_1_rew_updates_all_10K_demos_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/disc_128_disc_rewards_{}.png'.format(name),
                #     x_axis_lims=[0, 100],
                #     plot_mean=False,
                #     column_name=[
                #         'Disc_Rew_Max',
                #         'Disc_Rew_Min',
                #         'Disc_Avg_Grad_Norm_this_epoch',
                #         'Disc_Max_Grad_Norm_this_epoch'
                #     ],
                #     constraints=constraints
                # )

# for n_rew in [1, 2]:
#     for grad_pen in [5.0, 10.0, 15.0]:
#         for rew in [4.0, 6.0, 8.0, 10.0]:
#             for pol_size in [128, 256, 512]:
#                 constraints = {
#                     'gail_params.num_reward_updates': n_rew,
#                     'gail_params.grad_pen_weight': grad_pen,
#                     'policy_params.reward_scale': rew,
#                     'policy_net_size': pol_size 
#                 }
#                 name = 'disc_iters_{}_rew_{}_grad_pen_{}_pol_size_{}'.format(n_rew, rew, grad_pen, pol_size)
#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-10-updates-1-1-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                     'disc 128 '+name,
#                     '/h/kamyar/oorl_rlkit/plots/grad_clip_10_updates_1_1_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_65_disc_128_{}.png'.format(name),
#                     x_axis_lims=[0, 400],
#                     y_axis_lims=[-0.05, 1.05],
#                     plot_mean=False,
#                     column_name=[
#                         'Percent_Solved',
#                         'Percent_Good_Reach',
#                         # 'Percent_Solved_meta_train',
#                         # 'Percent_Solved_meta_test',
#                         'Disc_Acc',
#                         'Disc_Loss'
#                     ],
#                     constraints=constraints
#                 )

#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-10-updates-1-1-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                     'disc 128 '+name,
#                     '/h/kamyar/oorl_rlkit/plots/grad_clip_10_updates_1_1_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_65_disc_128_disc_rewards_{}.png'.format(name),
#                     x_axis_lims=[0, 400],
#                     plot_mean=False,
#                     column_name=[
#                         'Disc_Rew_Max',
#                         'Disc_Rew_Min',
#                         'Disc_Avg_Grad_Norm_this_epoch',
#                         'Disc_Max_Grad_Norm_this_epoch'
#                     ],
#                     constraints=constraints
#                 )

#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-10-updates-2-1-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                     'disc 128 '+name,
#                     '/h/kamyar/oorl_rlkit/plots/grad_clip_10_updates_2_1_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_32_disc_128_{}.png'.format(name),
#                     x_axis_lims=[0, 100],
#                     y_axis_lims=[-0.05, 1.05],
#                     plot_mean=False,
#                     column_name=[
#                         'Percent_Solved',
#                         'Percent_Good_Reach',
#                         # 'Percent_Solved_meta_train',
#                         # 'Percent_Solved_meta_test',
#                         'Disc_Acc',
#                         'Disc_Loss'
#                     ],
#                     constraints=constraints
#                 )

#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-10-updates-2-1-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                     'disc 128 '+name,
#                     '/h/kamyar/oorl_rlkit/plots/grad_clip_10_updates_2_1_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_32_disc_128_disc_rewards_{}.png'.format(name),
#                     x_axis_lims=[0, 100],
#                     plot_mean=False,
#                     column_name=[
#                         'Disc_Rew_Max',
#                         'Disc_Rew_Min',
#                         'Disc_Avg_Grad_Norm_this_epoch',
#                         'Disc_Max_Grad_Norm_this_epoch'
#                     ],
#                     constraints=constraints
#                 )

#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-10-updates-1-2-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                     'disc 128 '+name,
#                     '/h/kamyar/oorl_rlkit/plots/grad_clip_10_updates_1_2_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_16_disc_128_{}.png'.format(name),
#                     x_axis_lims=[0, 100],
#                     y_axis_lims=[-0.05, 1.05],
#                     plot_mean=False,
#                     column_name=[
#                         'Percent_Solved',
#                         'Percent_Good_Reach',
#                         # 'Percent_Solved_meta_train',
#                         # 'Percent_Solved_meta_test',
#                         'Disc_Acc',
#                         'Disc_Loss'
#                     ],
#                     constraints=constraints
#                 )

#                 plot_experiment_returns(
#                     '/scratch/gobi2/kamyar/oorl_rlkit/output/grad-clip-10-updates-1-2-disc-128-10K-demos-lr-3en4-clamped-rew-less-rew-iters-scale-0p9-linear-demos-with-target-disc-cpu-relu-batch-norm-disc-65-iters-correct-grind-zero-I-beg-of-you',
#                     'disc 128 '+name,
#                     '/h/kamyar/oorl_rlkit/plots/grad_clip_10_updates_1_2_disc_128_lr_3en4_clamped_10_scale_0p9_less_rew_iters_linear_demos_with_target_disc_cpu_relu_batchnorm_disc_grind_zero/rew_upd_16_disc_128_disc_rewards_{}.png'.format(name),
#                     x_axis_lims=[0, 100],
#                     plot_mean=False,
#                     column_name=[
#                         'Disc_Rew_Max',
#                         'Disc_Rew_Min',
#                         'Disc_Avg_Grad_Norm_this_epoch',
#                         'Disc_Max_Grad_Norm_this_epoch'
#                     ],
#                     constraints=constraints
#                 )






#     plot_experiment_returns(
#         '/scratch/gobi2/kamyar/oorl_rlkit/output/fucking-correct-zero-single-task-dac',
#         'rew %d fucking zero-single-task-dac-not-traj-based' % rew,
#         '/h/kamyar/oorl_rlkit/plots/fucking_zero_single_task_dac_traj_based_rew_%d.png' % rew,
#         # x_axis_lims=[0, 400],
#         y_axis_lims=[-0.05, 1.05],
#         plot_mean=False,
#         column_name=[
#             'Percent_Solved'
#             # 'Percent_Solved_meta_train',
#             # 'Percent_Solved_meta_test',
#             # 'Disc_Acc',
#             # 'Disc_Loss'
#         ],
#         constraints=constraints
#     )



# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-traj-based-gail-8-trajs-per-update',
#     'dac 8 trajs',
#     '/h/kamyar/oorl_rlkit/plots/dac_8_trajs.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved',
#         # 'Disc_Acc',
#         # 'Disc_Loss'
#     ],
# )

# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/traj-based-gail-32-trajs-per-update-subsample-4',
#     'dac 32 trajs subsample 4',
#     '/h/kamyar/oorl_rlkit/plots/dac_32_trajs_sub_4.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved',
#         # 'Disc_Acc',
#         # 'Disc_Loss'
#     ],
# )

# plot_experiment_returns(
#     '/scratch/gobi2/kamyar/oorl_rlkit/output/single-task-np-bc',
#     'single task np bc',
#     '/h/kamyar/oorl_rlkit/plots/single_task_np_bc.png',
#     # x_axis_lims=[0, 400],
#     y_axis_lims=[-0.05, 1.05],
#     plot_mean=False,
#     column_name=[
#         'Percent_Solved_meta_train',
#         'Percent_Good_Reach_meta_train',
#         # 'Percent_Solved_meta_test',
#         # 'Percent_Good_Reach_meta_test',
#     ],
# )
# ----------------------------------------------------------------------------------------------


# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/with-logging-reacher-bc-100-trajs',
#     'reacher behaviour cloning',
#     '/u/kamyar/oorl_rlkit/plots/reacher_bc.png',
#     y_axis_lims=[0, 80],
#     plot_mean=False,
#     # plot_horizontal_lines_at=[8191.5064, 8191.5064 + 629.3840, 8191.5064 - 629.3840],
#     # horizontal_lines_names=['expert_mean', 'expert_high_std', 'expert_low_std']
# )


# # dac-dmcs-simple-meta-reacher-2-layer-disc-batch-norm-1-subsampling
# # dac-dmcs-simple-meta-reacher-2-layer-disc-batch-norm-20-subsampling
# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/dac-dmcs-simple-meta-reacher-2-layer-disc-batch-norm-1-subsampling',
#     'dac-dmcs-simple-meta-reacher-2-layer-disc-batch-norm-1-subsampling',
#     '/u/kamyar/oorl_rlkit/plots/dac-dmcs-simple-meta-reacher-2-layer-disc-batch-norm-1-subsampling.png',
#     y_axis_lims=[0, 85],
#     plot_mean=False,
#     plot_horizontal_lines_at=[66.0388, 66.0388 + 16.3005, 66.0388 - 16.3005], # for the 1 subsampling expert
#     # plot_horizontal_lines_at=[64.3771, 64.3771 + 18.0769, 64.3771 - 18.0769], # for the 20 subsampling expert
#     horizontal_lines_names=['expert_mean', 'expert_high_std', 'expert_low_std']
# )


# for num_exp_traj in [100, 25]:
#     for rew in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
#         constraints = {
#             'policy_params.reward_scale': rew,
#             'num_expert_trajs': num_exp_traj
#         }
#         for exp_name in [
#             'temp-dac-dmcs-simple-meta-reacher-2-layer-disc-batch-norm',
#             'temp-dac-dmcs-simple-meta-reacher-2-layer-disc'
#         ]:
#             plot_experiment_returns(
#                 '/ais/gobi6/kamyar/oorl_rlkit/output/{}'.format(exp_name),
#                 '{} trajs {} rew_scale {}'.format(exp_name, num_exp_traj, rew),
#                 '/u/kamyar/oorl_rlkit/plots/{}_{}_trajs_{}_rew.png'.format(exp_name, num_exp_traj, rew),
#                 y_axis_lims=[0, 70],
#                 plot_mean=False,
#                 constraints=constraints
#             )



# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/test-something',
#     'test-something',
#     '/u/kamyar/oorl_rlkit/plots/test_something.png',
#     y_axis_lims=[-100, 7000],
#     plot_mean=False,
# )


# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/test-gail-simple-meta-reacher',
#     'test_gail_simple_meta_reacher',
#     '/u/kamyar/oorl_rlkit/plots/test_gail_simple_meta_reacher.png',
#     # y_axis_lims=[-100, 6000],
#     plot_mean=False,
# )


# for rb_size in [1000, 10000, 100000]:
#     for rew_scale in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
#         constraints = {
#             'gail_params.replay_buffer_size': rb_size,
#             'policy_params.reward_scale': rew_scale
#         }

#         for num_p_upd in [16, 32]:
#             constraints['gail_params.num_policy_updates'] = num_p_upd
#             name = 'rb_size_{}_rew_scale_{}_num_p_upd_{}'.format(rb_size, rew_scale, num_p_upd)
#             plot_experiment_returns(
#                 '/ais/gobi6/kamyar/oorl_rlkit/output/gail-simple-meta-reacher-p-updates-16-32',
#                 name,
#                 '/u/kamyar/oorl_rlkit/plots/gail_simple_meta_reacher_{}.png'.format(name),
#                 y_axis_lims=[0, 40],
#                 constraints=constraints,
#                 plot_mean=False,
#             )

#         for num_p_upd in [4, 8]:
#             constraints['gail_params.num_policy_updates'] = num_p_upd
#             name = 'rb_size_{}_rew_scale_{}_num_p_upd_{}'.format(rb_size, rew_scale, num_p_upd)
#             plot_experiment_returns(
#                 '/ais/gobi6/kamyar/oorl_rlkit/output/gail-simple-meta-reacher-p-updates-4-8',
#                 name,
#                 '/u/kamyar/oorl_rlkit/plots/gail_simple_meta_reacher_{}.png'.format(name),
#                 y_axis_lims=[0, 40],
#                 constraints=constraints,
#                 plot_mean=False,
#             )

#         for num_p_upd in [1, 2]:
#             constraints['gail_params.num_policy_updates'] = num_p_upd
#             name = 'rb_size_{}_rew_scale_{}_num_p_upd_{}'.format(rb_size, rew_scale, num_p_upd)
#             plot_experiment_returns(
#                 '/ais/gobi6/kamyar/oorl_rlkit/output/gail-simple-meta-reacher-p-updates-1-2',
#                 name,
#                 '/u/kamyar/oorl_rlkit/plots/gail_simple_meta_reacher_{}.png'.format(name),
#                 y_axis_lims=[0, 40],
#                 constraints=constraints,
#                 plot_mean=False,
#             )

# plot_experiment_returns(
#     '/ais/gobi6/kamyar/oorl_rlkit/output/test-something',
#     'DAC but no wrapping (but halfcheetah does not need wrapping anyways)',
#     '/u/kamyar/oorl_rlkit/plots/dac_halfcheetah.png',
#     y_axis_lims=[-100, 6000],
#     plot_mean=False,
# )

# plot_results(
#     'fixing-sac-meta-maze',
#     [
#         'algo_params.reward_scale',
#         'algo_params.soft_target_tau',
#         'env_specs.timestep_cost'
#     ],
#     plot_mean=False,
#     # y_axis_lims=[-3,3]
# )


# plot_results(
#     'on-the-fly-new-sac-ant',
#     [
#         'algo_params.epoch_to_start_training',
#         'algo_params.soft_target_tau',
#         'env_specs.normalized'
#     ],
#     plot_mean=False,
#     y_axis_lims=[0,4000]
# )





















# -------------------------------------------------------------------------------------------------------------------
# for reward_scale in [5.0]:
#     for epoch_to_start in [50]:
#         for seed in [9783, 5914, 4865, 2135, 2349]:
#             constraints = {
#                 # 'algo_params.reward_scale': reward_scale,
#                 # 'epoch_to_start_training': epoch_to_start,
#                 'seed': seed
#             }

#             try:
#                 plot_experiment_returns(
#                     '/u/kamyar/oorl_rlkit/output/4-layers-unnorm-meta-reacher-robustness-check',
#                     '4 layers unnormalized robustness check',
#                     '/u/kamyar/oorl_rlkit/plots/4_layers_unnorm_meta_reacher_robustness_check_seed_{}.png'.format(seed),
#                     y_axis_lims=[-300, 0],
#                     constraints=constraints
#                 )
#             except:
#                 print('failed ')




# FOR PLOTTING GEARS SEARCH
# for gear_0 in [[50], [100], [150], [400], [250], [300]]:
#     for gear_1 in [[50], [100], [150], [400], [250], [300]]:
#         constraints = {
#             'env_specs.gear_0': gear_0,
#             'env_specs.gear_1': gear_1,
#         }

#         try:
#             plot_experiment_returns(
#                 '/u/kamyar/oorl_rlkit/output/unnom-valid-gears-search',
#                 'unnom-valid-gears-search gear_0 {} gear_1 {}'.format(gear_0, gear_1),
#                 '/u/kamyar/oorl_rlkit/plots/reacher_gear_search/unnom_valid_gears_search_gear_0_{}_gear_1_{}.png'.format(gear_0, gear_1),
#                 y_axis_lims=[-100, 0],
#                 constraints=constraints
#             )
#         except:
#             print('failed ')




# FOR PLOTTING ROBUSTNESS
# for reward_scale in [5.0]:
#     for epoch_to_start in [50]:
#         for seed in [9783, 5914, 4865, 2135, 2349]:
#             constraints = {
#                 'seed': seed,
#                 'algo_params.reward_scale': reward_scale,
#                 'algo_params.num_updates_per_env_step': 4,
#                 'epoch_to_start_training': epoch_to_start,
#             }

#             try:
#                 plot_experiment_returns(
#                     '/u/kamyar/oorl_rlkit/output/meta-reacher-robustness-check',
#                     'meta-reacher-robustness-check seed {}'.format(seed),
#                     '/u/kamyar/oorl_rlkit/plots/meta-reacher-robustness-check-seed-{}.png'.format(seed),
#                     y_axis_lims=[-300, 0],
#                     constraints=constraints
#                 )
#             except:
#                 print('failed ')



# FOR PLOTTING REGRESSION RESULTS
# for train_set_size in [16000, 64000, 256000]:
#     for num_hidden_layers in [2, 4, 6, 8]:
#             for from_begin in [True, False]:
#                 # for seed in [9783, 5914, 4865, 2135, 2349]:
#                 constraints = {
#                     'train_set_size': train_set_size,
#                     'num_hidden_layers': num_hidden_layers,
#                     'train_from_beginning_transitions': from_begin
#                 }


#                 try:
#                     plot_experiment_returns(
#                         '/u/kamyar/oorl_rlkit/output/transition-hype-search-for-good-2-layer-small-range-meta-reacher',
#                         '{} layer MLP, from begin {}, train set size {}'.format(num_hidden_layers, from_begin, train_set_size),
#                         '/u/kamyar/oorl_rlkit/plots/regr_for_good_2_layer_small_range_meta_reacher_{}_layer_MLP_from_begin_{}_train_set_size_{}.png'.format(num_hidden_layers, from_begin, train_set_size),
#                         y_axis_lims=[0, 5],
#                         constraints=constraints,
#                         column_name='Obs_Loss'
#                     )
#                 except:
#                     print('failed')


# # FOR PLOTTING NPV1 regression results
# for val_context_size in [1, 10, 25, 50, 100, 250, 300, 400, 500, 750, 890]:
#     for train_batch_size in [16]:
#         for context_size_range in [[1,101], [1,251]]:
#         # for context_size_range in [[1, 21], [1, 51], [1, 101], [50, 101]]:
#             # for aggregator in ['mean_aggregator']:
#             for aggregator in ['sum_aggregator', 'mean_aggregator', 'tanh_sum_aggregator']:
#             # for num_train in: [50, 100, 400]:
#                 for num_train in [50]:
#                     for num_encoder_hidden_layers in [2]:
#                         for z_dim in [2, 5, 10, 20]:
#                             constraints = {
#                                 'train_batch_size': train_batch_size,
#                                 'context_size_range': context_size_range,
#                                 'aggregator_mode': aggregator,
#                                 'data_prep_specs.num_train': num_train,
#                                 'num_encoder_hidden_layers': num_encoder_hidden_layers,
#                                 'z_dim': z_dim
#                                 # 'num_train': num_train
#                             }

#                             try:
#                                 plot_experiment_returns(
#                                     '/u/kamyar/oorl_rlkit/output/more-fixed-train-modified-npv3',
#                                     'val_context_size_{} agg {} context_size_range {} num_train {} enc hidden layers {} z_dim {}'.format(val_context_size, aggregator, context_size_range, num_train, num_encoder_hidden_layers, z_dim),
#                                     '/u/kamyar/oorl_rlkit/plots/npv3_on_the_fly_reg_results/Test_MSE_val_context_size_{}_agg_{}_context_size_range_{}_num_train_{}_enc_hidden_layers_{}_z_dim_{}.png'.format(val_context_size, aggregator, context_size_range, num_train, num_encoder_hidden_layers, z_dim),
#                                     y_axis_lims=[0, 2],
#                                     constraints=constraints,
#                                     column_name='con_size_%d_Context_MSE'%val_context_size
#                                 )
#                             except Exception as e:
#                                 print('failed')
#                                 print(e)



# # FOR PLOTTING NO ENV INFO TRANSITION REGRESSION
# for num_hidden_layers in [2, 4]:
#     for train_from_beginning_transitions in [True, False]:
#         for remove_env_info in [True, False]:
#             constraints = {
#                 'num_hidden_layers': num_hidden_layers,
#                 'train_from_beginning_transitions': train_from_beginning_transitions,
#                 'remove_env_info': remove_env_info
#             }
#             try:
#                 plot_experiment_returns(
#                     '/u/kamyar/oorl_rlkit/output/on-the-fly-trans-regression',
#                     'on the fly transition regression num_hidden_{}_from_begin_{}_remove_info_{}'.format(num_hidden_layers, train_from_beginning_transitions, remove_env_info),
#                     '/u/kamyar/oorl_rlkit/plots/on_the_fly_transition_regression/num_hidden_{}_from_begin_{}_remove_info_{}.png'.format(num_hidden_layers, train_from_beginning_transitions, remove_env_info),
#                     y_axis_lims=[0, 2],
#                     column_name='Loss',
#                     constraints=constraints
#                 )
#                 print('worked')
#             except Exception as e:
#                 print('failed')
#                 print(e)





# META REACHER ON THE FLY ENV SAMPLING
# for seed in [9783, 5914, 4865, 2135, 2349]:
#     constraints = {'seed': seed}
#     try:
#         plot_experiment_returns(
#             '/u/kamyar/oorl_rlkit/output/larger-range-on-the-fly-unnorm-meta-reacher',
#             'larger_range_on_the_fly_unnorm_meta_reacher',
#             '/u/kamyar/oorl_rlkit/plots/larger_range_on_the_fly_unnorm_meta_reacher_seed_{}.png'.format(seed),
#             y_axis_lims=[-100, 0],
#             plot_mean=False,
#             constraints=constraints
#         )
#     except:
#         print('failed ')




# # FOR WTF
# for epoch_to_start in [50, 250]:
#     for num_updates in [1,4]:
#         for net_size in [64, 128, 256]:
#             constraints = {
#                 'algo_params.epoch_to_start_training': epoch_to_start,
#                 'algo_params.num_updates_per_env_step': num_updates,
#                 'net_size': net_size
#             }
#             try:
#                 plot_experiment_returns(
#                     '/u/kamyar/oorl_rlkit/output/wtf-norm-env-params',
#                     'normalized_env_info epoch_to_start_{}_num_updates_{}_net_size_{}'.format(epoch_to_start, num_updates, net_size),
#                     '/u/kamyar/oorl_rlkit/plots/wtf-norm/epoch_to_start_{}_num_updates_{}_net_size_{}.png'.format(epoch_to_start, num_updates, net_size),
#                     # y_axis_lims=[-300, 0],
#                     plot_mean=False,
#                     constraints=constraints
#                 )
#             except Exception as e:
#                 # raise Exception(e)
#                 print('failed ')






#     # constraints = {'seed': seed}
# for reward_scale in [1.0, 5.0, 10.0]:
#     for num_updates_per_env_step in [1, 4]:
#         for soft_target in [0.01, 0.005]:
#             for gear_0 in [[50], [100], [400]]:
#                 for gear_1 in [[50], [100], [400]]:
#                     for gear_2 in [[50], [100], [400]]:
#                         constraints = {
#                             'algo_params.reward_scale': reward_scale,
#                             'algo_params.num_updates_per_env_step': num_updates_per_env_step,
#                             'algo_params.soft_target_tau': soft_target,
#                             'env_specs.gear_0': gear_0,
#                             'env_specs.gear_1': gear_1,
#                             'env_specs.gear_2': gear_2
#                         }
#                         try:
#                             plot_experiment_returns(
#                                 '/u/kamyar/oorl_rlkit/output/unnorm-meta-hopper-hype-search',
#                                 'unnorm-meta-hopper-hype-search rew {} num upd {} soft-targ {} gear_0_{}_gear_1_{}_gear_2_{}'.format(reward_scale, num_updates_per_env_step, soft_target, gear_0, gear_1, gear_2),
#                                 '/u/kamyar/oorl_rlkit/plots/unnorm_meta_hopper_gears_search/unnorm-meta-hopper-hype-search_rew_{}_num_upd_{}_soft-targ_{}_gear_0_{}_gear_1_{}_gear_2_{}.png'.format(reward_scale, num_updates_per_env_step, soft_target, gear_0, gear_1, gear_2),
#                                 y_axis_lims=[0, 3000],
#                                 plot_mean=False,
#                                 constraints=constraints
#                             )
#                         except Exception as e:
#                             # raise(e)
#                             print('failed ')

















# for num_updates in [1,4]:
#     for reward_scale in [1,5]:
#         for soft in [0.005, 0.01]:
#             for normalized in [True, False]:
#                 constraints = {
#                     'algo_params.num_updates_per_env_step': num_updates,
#                     'algo_params.reward_scale': reward_scale,
#                     'algo_params.soft_target_tau': soft,
#                     'env_specs.normalized': normalized,
#                 }
#                 name = '_'.join(
#                     [
#                         '{}_{}'.format(k, constraints[k]) for
#                         k in sorted(constraints)
#                     ]
#                 )
#                 try:
#                     plot_experiment_returns(
#                         '/u/kamyar/oorl_rlkit/output/new-sac-ant',
#                         name,
#                         '/u/kamyar/oorl_rlkit/plots/new-sac-ant/{}.png'.format(name),
#                         y_axis_lims=[0, 3000],
#                         plot_mean=False,
#                         constraints=constraints
#                     )
#                 except Exception as e:
#                     # raise(e)
#                     print('failed ')




# # PLOT EVERYTHING
# try:
#     plot_experiment_returns(
#         '/u/kamyar/oorl_rlkit/output/new-sac-ant',
#         'new-sac-ant',
#         '/u/kamyar/oorl_rlkit/plots/new-sac-ant.png',
#         # y_axis_lims=[0, 3000],
#         plot_mean=False,
#         # constraints=constraints
#     )
# except Exception as e:
#     raise(e)
#     print('failed ')


# Hopper gears
# for gear_0 in [[50.], [100.], [400.]]:
#     for gear_1 in [[50.], [100.], [400.]]:
#         for gear_2 in [[50.], [100.], [400.]]:
#             constraints = {
#                 'env_specs.gear_0': gear_0,
#                 'env_specs.gear_1': gear_1,
#                 'env_specs.gear_2': gear_2,
#             }
#             name = 'gear_0_{}_gear_1_{}_gear_2_{}'.format(gear_0, gear_1, gear_2)
#             try:
#                 plot_experiment_returns(
#                     '/u/kamyar/oorl_rlkit/output/new-norm-meta-hopper-gear-search',
#                     'hopper ' + name,
#                     '/u/kamyar/oorl_rlkit/plots/new_norm_meta_hopper_gears_search/mea_{}.png'.format(name),
#                     y_axis_lims=[0, 3000],
#                     plot_mean=True,
#                     constraints=constraints
#                 )
#             except Exception as e:
#                 # raise(e)
#                 print('failed ')


# for concat in [True, False]:
#     for normalized in [True, False]:
#         constraints = {
#             'algo_params.concat_env_params_to_obs': concat,
#             'env_specs.normalized': normalized,
#         }
#         name = 'concat_env_params_{}_normalized_{}'.format(concat, normalized)
#         try:
#             plot_experiment_returns(
#                 '/u/kamyar/oorl_rlkit/output/hopper-on-the-fly',
#                 'hopper ' + name,
#                 '/u/kamyar/oorl_rlkit/plots/hopper_on_the_fly/{}.png'.format(name),
#                 y_axis_lims=[0, 3000],
#                 plot_mean=False,
#                 constraints=constraints
#             )
#         except Exception as e:
#             # raise(e)
#             print('failed ')




# MAZE
# for reward_scale in [0.5, 1, 5, 10]:
#     constraints = {
#         'algo_params.reward_scale': reward_scale
#     }
#     name = 'rew_{}'.format(reward_scale)
#     try:
#         plot_experiment_returns(
#             '/u/kamyar/oorl_rlkit/output/3x3-1-obj-sac-meta-maze',
#             '{}'.format(name),
#             '/u/kamyar/oorl_rlkit/plots/3x3-1-obj-sac-meta-maze/{}.png'.format(name),
#             y_axis_lims=[0, 3],
#             plot_mean=False,
#             constraints=constraints
#         )
#     except Exception as e:
#         raise(e)
#         print('failed ')