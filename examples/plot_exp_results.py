from rlkit.core.eval_util import plot_experiment_returns

plot_experiment_returns(
    '/u/kamyar/oorl_rlkit/output/un-norm-truth-cond-meta-reacher',
    'truth cond meta reacher hyper param search',
    '/u/kamyar/oorl_rlkit/plots/un_norm_truth_cond_meta_reacher.png',
    y_axis_lims=[-300, 0]
)
