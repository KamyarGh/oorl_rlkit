import argparse
from rlkit.core.eval_util import plot_experiment_returns

plot_experiment_returns(
    '/u/kamyar/oorl_rlkit/output/meta-reacher-robustness-check',
    'meta_reacher_robustness_check',
    '/u/kamyar/oorl_rlkit/plots/meta-reacher-robustness-check.png',
    y_axis_lims=[-300, 0]
)
