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

plot_experiment_returns(
    '/scratch/gobi2/kamyar/oorl_rlkit/output/test-gpu-sum-dist-last-30-more-info-normal-try-few-shot-np-bc',
    'gpu sum dist percent good reach',
    '/h/kamyar/oorl_rlkit/plots/gpu_sum_dist_percent_good_reach.png',
    x_axis_lims=[0, 600],
    y_axis_lims=[-0.05, 1.05],
    plot_mean=False,
    column_name=[
        'Percent_Solved_meta_train',
        'Percent_Good_Reach_meta_train',
        'Percent_Solved_meta_test',
        'Percent_Good_Reach_meta_test',
    ],
)


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
# for gear_0 in [[50], [100], [150], [200], [250], [300]]:
#     for gear_1 in [[50], [100], [150], [200], [250], [300]]:
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
#             # for num_train in: [50, 100, 200]:
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
#             for gear_0 in [[50], [100], [200]]:
#                 for gear_1 in [[50], [100], [200]]:
#                     for gear_2 in [[50], [100], [200]]:
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
# for gear_0 in [[50.], [100.], [200.]]:
#     for gear_1 in [[50.], [100.], [200.]]:
#         for gear_2 in [[50.], [100.], [200.]]:
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
