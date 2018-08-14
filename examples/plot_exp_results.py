import argparse
from rlkit.core.eval_util import plot_experiment_returns

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
#                 '/u/kamyar/oorl_rlkit/plots/unnom_valid_gears_search_gear_0_{}_gear_1_{}.png'.format(gear_0, gear_1),
#                 y_axis_lims=[-300, 0],
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


# FOR PLOTTING NPV1 regression results
val_context_size = 50
for train_batch_size in [16]:
    for context_size_range in [[50, 101]]:
    # for context_size_range in [[1, 21], [1, 51], [1, 101], [50, 101]]:
        for aggregator in ['sum_aggregator', 'mean_aggregator', 'tanh_sum_aggregator']:
        # for aggregator in ['sum_aggregator', 'mean_aggregator', 'tanh_sum_aggregator']:
        # for num_train in: [50, 100, 200]:
            # for num_encoder_hidden_layers in [2, 4, 6]:
            for num_train in [50, 250]:
                constraints = {
                    'train_batch_size': train_batch_size,
                    'context_size_range': context_size_range,
                    'aggregator': aggregator,
                    'data_prep_specs.num_train': num_train
                    # 'num_hidden_layers': num_hidden_layers
                    # 'num_train': num_train
                }

                try:
                    plot_experiment_returns(
                        '/u/kamyar/oorl_rlkit/output/making-npv1-trans-regr-work',
                        'val_context_size_{}_train_batch_size {} agg {} context_size_range {} num_train {}'.format(val_context_size, train_batch_size, aggregator, context_size_range, num_train),
                        '/u/kamyar/oorl_rlkit/plots/context_mse_val_context_size_{}_train_batch_size_{}_agg_{}_context_size_range_{}_num_train{}.png'.format(val_context_size, train_batch_size, aggregator, context_size_range, num_train),
                        y_axis_lims=[0, 2],
                        constraints=constraints,
                        column_name='con_size_%d_Context_MSE'%val_context_size
                    )
                except Exception as e:
                    print('failed')
                    print(e)



# FOR PLOTTING NO ENV INFO TRANSITION REGRESSION
# try:
#     plot_experiment_returns(
#         '/u/kamyar/oorl_rlkit/output/wtf-man-norm-env',
#         'wtf is going on',
#         '/u/kamyar/oorl_rlkit/plots/wtf_man_norm_env_total_loss.png',
#         y_axis_lims=[0, 2],
#         column_name='Loss'
#     )
# except Exception as e:
#     print('failed')
#     print(e)
