import abc
from collections import OrderedDict

import numpy as np

from rlkit.core.base_algorithm import BaseAlgorithm
from rlkit.torch import pytorch_util as ptu
from rlkit.core import logger, eval_util


class TorchBaseAlgorithm(BaseAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self, *args, render_eval_paths=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sampler.animated = render_eval_paths
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths

    @property
    @abc.abstractmethod
    def networks(self):
        """
        Used in many settings such as moving to devices
        """
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device):
        for net in self.networks:
            net.to(device)

    def evaluate(self, epoch):
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except:
            print('No Stats to Eval')

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        statistics.update(eval_util.get_generic_path_information(
            self._exploration_paths, stat_prefix="Exploration",
        ))

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
        if hasattr(self.env, "log_statistics"):
            statistics.update(self.env.log_statistics(test_paths))
        if hasattr(self.env, "log_new_ant_multi_statistics"):
            env_log_stats = self.env.log_new_ant_multi_statistics(test_paths, epoch, logger.get_snapshot_dir())
            statistics.update(env_log_stats)
        
        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        
        best_statistic = statistics[self.best_key]
        if best_statistic > self.best_statistic_so_far:
            self.best_statistic_so_far = best_statistic
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {
                    'epoch': epoch,
                    'statistics': statistics
                }
                data_to_save.update(self.get_epoch_snapshot(epoch))
                logger.save_extra_data(data_to_save, 'best.pkl')
                print('\n\nSAVED BEST\n\n')
