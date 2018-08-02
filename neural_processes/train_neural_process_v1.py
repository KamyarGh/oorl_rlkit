import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam

from numpy import array
from numpy.random import choice, randint

from generic_map import GenericMap
from base_map import BaseMap
from neural_process import NeuralProcessV1
from aggregators import sum_aggregator, mean_aggregator, tanh_sum_aggregator

from tasks.sinusoidal import SinusoidalTask
# -----------------------------------------------------------------------------
N_tasks = 100
r_dim = 10
z_dim = 5
base_map_lr = 1e-3
encoder_lr = 1e-3
r_to_z_map_lr = 1e-3
post_mean_lr = 1e-1
post_log_cov_lr = 1e-1
max_iters = 10001
num_tasks_per_batch = 16

# data_sampling_mode = 'constant'
# num_per_task = 10

data_sampling_mode = 'random'
num_per_task_low = 3
num_per_task_high = 11

aggregator = mean_aggregator

freq_val = 100

# -----------------------------------------------------------------------------
all_tasks = [SinusoidalTask() for _ in range(N_tasks)]
def generate_data_batch(tasks_batch, num_samples_per_task, max_num):
    # Very inefficient will need to fix this
    X = torch.zeros(len(tasks_batch), max_num, 1)
    Y = torch.zeros(len(tasks_batch), max_num, 1)
    for i, (task, num_samples) in enumerate(zip(tasks_batch, num_samples_per_task)):
        num = int(num_samples)
        x, y = task.sample(num)
        X[i,:num] = x
        Y[i,:num] = y

    return Variable(X), Variable(Y)

def generate_mask(num_tasks_per_batch, max_num):
    mask = torch.ones(num_tasks_per_batch.shape[0], max_num, 1)
    for i, num in enumerate(num_tasks_per_batch):
        if num == max_num: continue
        mask[i,num:] = 0.0
    return Variable(mask)

# -----------------------------------------------------------------------------
encoder = GenericMap(
    [1], [r_dim], siamese_input=False,
    num_hidden_layers=2, hidden_dim=40,
    siamese_output=False, act='relu',
    deterministic=True
)
encoder_optim = Adam(encoder.parameters(), lr=encoder_lr)

base_map = BaseMap(
    z_dim, [1], [1], siamese_input=False,
    num_hidden_layers=2, hidden_dim=40,
    siamese_output=False, act='relu',
    deterministic=True
)
base_map_optim = Adam(base_map.parameters(), lr=base_map_lr)

r_to_z_map = GenericMap(
    [r_dim], [z_dim], siamese_input=False,
    num_hidden_layers=2, hidden_dim=40,
    siamese_output=False, act='relu',
    deterministic=False
)
r_to_z_map_optim = Adam(r_to_z_map.parameters(), lr=r_to_z_map_lr)

neural_process = NeuralProcessV1(
    encoder,
    encoder_optim,
    aggregator,
    r_to_z_map,
    r_to_z_map_optim,
    base_map,
    base_map_optim,
    use_nat_grad=False
)

# -----------------------------------------------------------------------------
for iter_num in range(max_iters):
    task_batch_idxs = choice(len(all_tasks), size=num_tasks_per_batch, replace=False)
    if data_sampling_mode == 'random':
        num_samples_per_task = randint(
            num_per_task_low,
            high=num_per_task_high,
            size=(num_tasks_per_batch)
        )
        max_num = num_per_task_high
    else:
        raise NotImplementedError
    
    X, Y = generate_data_batch([all_tasks[i] for i in task_batch_idxs], num_samples_per_task, max_num)
    mask = generate_mask(num_samples_per_task, max_num)
    batch = {
        'input_batch_list': [X],
        'output_batch_list': [Y],
        'mask': mask
    }
    neural_process.train_step(batch)

    if iter_num % freq_val == 0:
        neural_process.set_mode('eval')

        val_tasks = [SinusoidalTask() for _ in range(num_tasks_per_batch)]

        num_samples_per_task = randint(
            num_per_task_low,
            high=num_per_task_high,
            size=(num_tasks_per_batch)
        )
        X_varied, Y_varied = generate_data_batch(val_tasks, num_samples_per_task, max_num)
        mask_varied = generate_mask(num_samples_per_task, max_num)
        batch_varied = {
            'input_batch_list': [X_varied],
            'output_batch_list': [Y_varied],
            'mask': mask_varied
        }
        posts_varied = neural_process.infer_posterior_params(batch_varied)

        num_samples_per_task = array([num_per_task_high for _ in range(num_tasks_per_batch)])
        X_max, Y_max = generate_data_batch(val_tasks, num_samples_per_task, max_num)
        mask_max = generate_mask(num_samples_per_task, max_num)
        batch_max = {
            'input_batch_list': [X_max],
            'output_batch_list': [Y_max],
            'mask': mask_max
        }
        posts_max = neural_process.infer_posterior_params(batch_max)

        num_samples_per_task = array([max_num for _ in range(num_tasks_per_batch)])
        X_test, Y_test = generate_data_batch(val_tasks, num_samples_per_task, max_num)
        mask_test = generate_mask(num_samples_per_task, max_num)
        batch_test = {
            'input_batch_list': [X_test],
            'output_batch_list': [Y_test],
            'mask': mask_test
        }
        elbo_varied = neural_process.compute_ELBO(posts_varied, batch_test)
        test_log_likelihood_varied = neural_process.compute_cond_log_likelihood(posts_varied, batch_test, mode='eval')
        elbo_max = neural_process.compute_ELBO(posts_max, batch_test)
        test_log_likelihood_max = neural_process.compute_cond_log_likelihood(posts_max, batch_test, mode='eval')

        print('\n--------------------')
        print('Iter %d ELBO Var: %.4f' % (iter_num, elbo_varied))
        print('Iter %d Test Log Like Var: %.4f' % (iter_num, test_log_likelihood_varied))
        print('-----')
        print('Iter %d ELBO Max: %.4f' % (iter_num, elbo_max))
        print('Iter %d Test Log Like Max: %.4f' % (iter_num, test_log_likelihood_max))

        # inference batch
        # test batch
        # Check performance conditioning on range(1,22,4) number of points
        # Make the points subsets of one-another to be able to properly evaluate

        neural_process.set_mode('train')
