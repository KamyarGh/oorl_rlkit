import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd

from numpy import pi
from numpy import log as np_log

from neural_processes.base_neural_process import BaseNeuralProcess
from neural_processes.distributions import sample_diag_gaussians, local_repeat

log_2pi = np_log(2*pi)


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def compute_spherical_log_prob(preds, true_outputs, mask, n_samples):
    '''
        Compute log prob assuming spherical Gaussian with some mean
        Up to additive constant
    '''
    log_prob = -0.5 * torch.sum(mask*(preds - true_outputs)**2)
    if n_samples > 1: log_prob /= float(n_samples)
    return log_prob


def compute_diag_log_prob(preds_mean, preds_log_cov, true_outputs, mask, n_samples):
    '''
        Compute log prob assuming diagonal Gaussian with some mean and log cov
    '''
    assert False, 'The tests so far have shown this to be unstable'
    preds_cov = torch.exp(preds_log_cov)

    log_prob = -0.5 * torch.sum(
        mask*(preds_mean - true_outputs)**2 / preds_cov
    )

    log_det_temp = mask*(torch.sum(preds_log_cov, 1) + log_2pi)
    log_prob += -0.5*torch.sum(log_det_temp)

    if n_samples > 1: log_prob /= float(n_samples)
    return log_prob


class NeuralProcessV2(BaseNeuralProcess):
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
        r_to_z_map_optim,
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
        self.r_to_z_map_optim = r_to_z_map_optim
        self.use_nat_grad = use_nat_grad
        self.set_mode('train')
    

    def set_mode(self, mode):
        assert mode in ['train', 'eval']
        mode = True if mode=='train' else False
        self.base_map.train(mode)
        self.encoder.train(mode)
        self.r_to_z_map.train(mode)

        self.training = mode


    def infer_posterior_params(self, batch):
        '''
            batch should be dictionary of arrays of size N_tasks x N_samples x dim
            batch = {
                'input_batch_list': ...,
                'output_batch_list': ...,
                'mask': ..., # used to condition on varying number of points
            }
        '''
        input_list = batch['input_batch_list']
        output_list = batch['output_batch_list']
        N_tasks, N_samples = input_list[0].size(0), input_list[0].size(1)
        reshaped_input_list = [inp.contiguous().view(-1, inp.size(2)) for inp in input_list]
        reshaped_output_list = [out.contiguous().view(-1, out.size(2)) for out in output_list]        

        r = self.encoder(reshaped_input_list + reshaped_output_list)[0]
        r = r.view(N_tasks, N_samples, -1)
        r_agg = self.aggregator(r, batch['mask'])
        
        # since r_to_z_map is a generic map, it will output
        # [[mean, log_cov]] because that is the interface
        # of the generic map
        post = self.r_to_z_map([r_agg])
        return post[0]
    

    def sample_outputs(self, posteriors, input_batch_list, n_samples):
        z_means = posteriors[0]
        z_log_covs = posteriors[1]
        z_covs = torch.exp(z_log_covs)
        z_samples = sample_diag_gaussians(z_means, z_covs, n_samples)
        z_samples = local_repeat(z_samples, input_batch_list[0].size(1))

        num_tasks, num_per_task = input_batch_list[0].size(0), input_batch_list[0].size(1)
        input_batch_list = [inp.contiguous().view(-1,inp.size(2)) for inp in input_batch_list]
        input_batch_list = [local_repeat(inp, n_samples) for inp in input_batch_list]

        if (not self.base_map.siamese_output) and self.base_map.deterministic:
            outputs = self.base_map(z_samples, input_batch_list)[0]
            outputs = outputs.view(num_tasks, n_samples, num_per_task, outputs.size(-1))
        else:
            raise NotImplementedError

        return outputs


    def train_step(self, context_batch, test_batch):
        self.encoder_optim.zero_grad()
        self.r_to_z_map.zero_grad()
        self.base_map_optim.zero_grad()

        context_union_test_batch = {
            'input_batch_list': [
                torch.cat([c, t], 1)
                for c, t in zip(context_batch['input_batch_list'], test_batch['input_batch_list'])
            ],
            'output_batch_list': [
                torch.cat([c, t], 1)
                for c, t in zip(context_batch['output_batch_list'], test_batch['output_batch_list'])
            ],
            'mask': torch.cat([context_batch['mask'], test_batch['mask']], 1)
        }

        context_posteriors = self.infer_posterior_params(context_batch)
        context_union_test_posteriors = self.infer_posterior_params(context_union_test_batch)
        neg_elbo = -1.0 * self.compute_ELBO(context_posteriors, context_union_test_posteriors, test_batch)

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
            self.r_to_z_map_optim.step()
            self.encoder_optim.step()


    def compute_ELBO(self, context_posteriors, context_union_test_posteriors, test_batch, mode='train'):
        '''
            n_samples is the number of samples used for
            Monte Carlo estimate of the ELBO
        '''
        context_posteriors = [k.detach() for k in context_posteriors]
        cond_log_likelihood = self.compute_cond_log_likelihood(context_union_test_posteriors, test_batch, mode)
        KL = self.compute_ELBO_KL(context_union_test_posteriors, context_posteriors)
        
        elbo = cond_log_likelihood - KL
        
        # idk whether to put this or not
        elbo = elbo / float(test_batch['input_batch_list'][0].size(0))
        return elbo
    

    def compute_cond_log_likelihood(self, posteriors, batch, mode='train'):
        '''
            Computer E[log p(y|x,z)] up to constant additional factors

            arrays are N_tasks x N_samples x dim
        '''
        # not dealing with more than 1 case right now
        n_samples = 1

        input_batch_list, output_batch_list = batch['input_batch_list'], batch['output_batch_list']
        mask = batch['mask'].view(-1, 1)

        z_means = posteriors[0]
        z_log_covs = posteriors[1]
        z_covs = torch.exp(z_log_covs)
    
        if mode == 'eval':
            z_samples = z_means
        else:
            z_samples = sample_diag_gaussians(z_means, z_covs, n_samples)
        z_samples = local_repeat(z_samples, input_batch_list[0].size(1))
        input_batch_list = [inp.view(-1,inp.size(2)) for inp in input_batch_list]
        output_batch_list = [out.view(-1,out.size(2)) for out in output_batch_list]

        preds = self.base_map(z_samples, input_batch_list)

        if self.base_map.siamese_output:
            log_prob = 0.0
            if self.base_map.deterministic:
                for pred, output in zip(preds, output_batch_list):
                    log_prob += compute_spherical_log_prob(pred, output, mask, n_samples)
            else:
                for pred, output in zip(preds, output_batch_list):
                    log_prob += compute_diag_log_prob(
                        pred[0], pred[1], output, mask, n_samples
                    )
        else:
            if self.base_map.deterministic:
                log_prob = compute_spherical_log_prob(
                    preds[0], output_batch_list[0], mask, n_samples
                )
            else:
                preds_mean, preds_log_cov = preds[0][0], preds[0][1]
                log_prob = compute_diag_log_prob(
                    preds_mean, preds_log_cov, output_batch_list[0], mask, n_samples
                )

        return log_prob


    def compute_ELBO_KL(self, posteriors_1, posteriors_2, detach_posteriors_2=True):
        # '''
        #     We always deal with spherical Gaussian prior
        # '''
        if detach_posteriors_2:
            posteriors_2 = [p.detach() for p in posteriors_2]

        m1, lc1, m2, lc2 = posteriors_1[0], posteriors_1[1], posteriors_2[0], posteriors_2[1]
        KL = 0.5 * (
            torch.sum(lc2, 1) - torch.sum(lc1, 1) - m1.size(1) + 
            torch.sum(torch.exp(lc1 - lc2), 1) + torch.sum((m2 - m1)**2 / torch.exp(lc2), 1)
        )
        KL = torch.sum(KL)
        print('-----')
        print(KL.data[0])
        print('\n')
        print(m1[0,:5].data.numpy())
        print(m2[0,:5].data.numpy())
        print('\n')
        print(torch.exp(lc1[0,:5]).data.numpy())
        print(torch.exp(lc2[0,:5]).data.numpy())

        return KL
