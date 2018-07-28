import abc

class BaseNeuralProcess(object, metaclass=abc.ABCMeta):
    '''
    General interface for a neural process
    '''
    @abc.abstractmethod
    def train_step(self, batch, post_init, n_samples=1):
        pass
    

    @abc.abstractmethod
    def update_posteriors(self, batch, post_init, n_samples=1):
        pass

    
    @abc.abstractmethod
    def compute_ELBO(self, posteriors, context_set, test_set):
        pass

    @abc.abstractmethod
    def compute_cond_log_likelihood(self, posteriors, *args):
        pass
