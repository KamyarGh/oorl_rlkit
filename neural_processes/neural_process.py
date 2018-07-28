import torch
import torch.nn as nn
import torch.nn.function as F
from torch.autograd import Variable
from torch import autograd

from numpy import pi
from numpy import log as np_log

from base_neural_process import BaseNeuralProcess
from distributions import sample_diag_gaussians, local_repeat

log_2pi = np_log(2*pi)

def compute_spherical_log_prob(preds, true_outputs, n_samples)
    '''
        Compute log prob assuming spherical Gaussian with some mean
        Up to additive constant
    '''
    return -0.5 * torch.sum((preds - true_outputs)**2) / float(n_samples)


def compute_diag_log_prob(preds_mean, preds_log_cov, true_outputs, n_samples):
    '''
        Compute log prob assuming diagonal Gaussian with some mean and log cov
    '''
    preds_cov = torch.exp(preds_log_cov)

    log_prob = -0.5 * torch.sum(
        (preds_mean - repeated_true_outputs)**2 / preds_cov
    )

    log_det = torch.logsumexp(torch.sum(preds_log_cov, 1))
    log_det += log_2pi
    log_det *= -0.5
    log_prob += log_det

    log_prob /= float(n_samples)
    return log_prob


class NeuralProcessV1(BaseNeuralProcess):
    '''
        Neural process
        This is the version of Neural Processes from equation (7)
        of the Neural Processes paper

        Right now not dealing with taking multiple samples from z to
        do Monte-Carlo estimate. Hopefully won't become necessary.

        Prior is assumed to be a spherical Gaussian
    '''
    def __init__(
        self,
        encoder,
        encoder_optim,
        aggregator,
        r_to_z_map,
        base_map,
        base_map_optim,
        use_nat_grad=True, # whether to use natural gradient for posterior parameter updates
    ):
        self.base_map = base_map
        self.base_map_optim = base_map_optim
        self.encoder = encoder
        self.encoder_optim = encoder_optim
        self.aggregator = aggregator
        self.r_to_z_map = r_to_z_map
        self.use_nat_grad = use_nat_grad


    def infer_posterior_params(self, batch):
        '''
            batch should be dictionary of arrays of size N_tasks x N_samples x dim
            batch = {
                'input_batch_list': ...,
                'output_batch_list': ...,
                'mask': ..., # used to condition on varying number of points
            }
        '''
        input_list = batch['input_list']
        N_tasks, N_samples = input_list[0].size(0), input_list[0].size(1)
        reshaped_input_list = [inp.view(-1, inp.size(2)) for inp in input_list]

        r = self.encoder(reshaped_input_list)
        r = r.view(N_tasks, N_samples, -1)
        r_agg = self.aggregator(r)
        
        # post should be a dictionary containing keys 'means' and 'log_covs'
        post = self.r_to_z_map(r_agg, batch['mask'])
        return post


    def train_step(self, batch):
        self.encoder_optim.zero_grad()
        self.r_to_z_map.zero_grad()
        self.base_map_optim.zero_grad()

        neg_elbo = -1.0*self.compute_ELBO(batch)

        if self.use_nat_grad:
            raise NotImplementedError
            # some code from encoder-free NP
            mean_grad, log_cov_grad = z_means.grad, log_cov_grad.grad
            if self.use_nat_grad:
                mean_grad.mul_(torch.exp(z_log_covs))
                log_cov_grad.mul_(2.0)
            z_means.sub_(mean_grad * mean_lr)
            z_log_covs.sub_(log_cov_grad * log_cov_lr)
        else:
            neg_elbo.backward()
            self.base_map_optim.step()
            self.r_to_z_map.step()
            self.encoder_optim.step()


    def compute_ELBO(self, posteriors, batch):
        '''
            n_samples is the number of samples used for
            Monte Carlo estimate of the ELBO
        '''
        posteriors = self.infer_posterior_params(batch)
        cond_log_likelihood = self.compute_cond_log_likelihood(posteriors, batch)
        KL = self.compute_ELBO_KL(posteriors)
        return cond_log_likelihood - KL
    

    def compute_cond_log_likelihood(self, posteriors, batch):
        '''
            Computer E[log p(y|x,z)] up to constant additional factors

            arrays are N_tasks x N_samples x dim
        '''
        # not dealing with more than 1 case right now
        n_samples = 1

        input_batch_list, output_batch_list = batch['input_batch_list'], batch['output_batch_list']

        z_means = posteriors['means']
        z_log_covs = posteriors['log_covs']
        z_covs = torch.exp(z_log_covs)

        z_samples = sample_diag_gaussians(z_means, z_covs, n_samples)
        z_samples = local_repeat(z_samples, input_batch_list[0].size(1))
        input_batch_list = [inp.view(-1,inp.size(2)) for inp in input_batch_list]
        output_batch_list = [out.view(-1,out.size(2)) for out in output_batch_list]

        preds = self.base_map(z_samples, input_batch_list)

        if self.base_map.siamese_outputs:
            log_prob = 0.0
            if self.base_map.deterministic:
                for pred, output in zip(preds, repeated_true_outputs):
                    log_prob += compute_spherical_log_prob(pred, output, n_samples)
            else:
                for pred, output in zip(preds, repeated_true_outputs):
                    log_prob += compute_diag_log_prob(
                        pred[0], pred[1], output, n_samples
                    )
        else:
            if self.base_map.deterministic:
                log_prob = compute_spherical_log_prob(
                    preds[0], repeated_true_outputs[0], n_samples
                )
            else:
                preds_mean, preds_log_cov = preds[0][0], preds[0][1]
                log_prob = compute_diag_log_prob(
                    preds_mean, preds_log_cov, repeated_true_outputs[0], n_samples
                )

        return log_prob


    def compute_ELBO_KL(self, posteriors):
        '''
            We always deal with spherical Gaussian prior
        '''
        z_means = posteriors['means']
        z_log_covs = posteriors['log_covs']
        z_covs = torch.exp(z_log_covs)
        KL = 0.5 * torch.sum(
            1.0 + z_log_covs - z_means**2 - z_covs
        )
        return KL
