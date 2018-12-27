'''
Literally only difference with TorchIRLAlgorithm is the line:
class TorchMetaIRLAlgorithm(...):
'''

import abc
from collections import OrderedDict
from typing import Iterable

import numpy as np
from torch.autograd import Variable

import rlkit.core.eval_util
from rlkit.core.meta_irl_algorithm import MetaIRLAlgorithm
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.core import logger, eval_util


class TorchMetaIRLAlgorithm(MetaIRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self, *args, render_eval_paths=False, plotter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.plotter = plotter

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def cuda(self):
        for net in self.networks:
            net.cuda()
    
    def cpu(self):
        for net in self.networks:
            net.cpu()

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        # statistics.update(eval_util.get_generic_path_information(
        #     self._exploration_paths, stat_prefix="Exploration",
        # ))

        for mode in ['meta_train', 'meta_test']:
            logger.log("Collecting samples for evaluation")
            test_paths = self.obtain_eval_samples(epoch, mode=mode)

            statistics.update(eval_util.get_generic_path_information(
                test_paths, stat_prefix="Test " + mode,
            ))
            # print(statistics.keys())
            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(test_paths)
            if hasattr(self.env, "log_statistics"):
                log_stats = self.env.log_statistics(test_paths)
                new_log_stats = OrderedDict((k+' '+mode, v) for k, v in log_stats.items())
                statistics.update(new_log_stats)

            average_returns = rlkit.core.eval_util.get_average_returns(test_paths)
            statistics['AverageReturn '+mode] = average_returns

            if self.render_eval_paths:
                self.env.render_paths(test_paths)

        for key, value in statistics.items():
            logger.record_tabular(key, value)
        
        if self.plotter:
            self.plotter.draw()


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }
