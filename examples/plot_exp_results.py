import argparse
import os
import yaml
from itertools import product

from rlkit.core.eval_util import plot_experiment_returns




def plot_results(exp_name, variables_to_permute, plot_mean=False, y_axis_lims=None):
    output_dir = '/u/kamyar/oorl_rlkit/output'
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
    

# plot_results(
#     'one-hot-5x5-1-obj-sac-meta-maze',
#     [
#         'algo_params.num_updates_per_env_step',
#         'algo_params.reward_scale',
#         'algo_params.soft_target_tau',
#     ],
#     plot_mean=False,
#     y_axis_lims=[-3,3]
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


plot_results(
    'on-the-fly-new-sac-ant',
    [
        'algo_params.epoch_to_start_training',
        'algo_params.soft_target_tau',
        'env_specs.normalized'
    ],
    plot_mean=False,
    y_axis_lims=[0,4000]
)





















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
