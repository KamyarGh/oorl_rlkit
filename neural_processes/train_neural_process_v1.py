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
use_bn = True
N_tasks = 100
r_dim = 10
z_dim = 20
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

aggregator = sum_aggregator

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
        if num==max_num:
            X[i,:] = x
            Y[i,:] = y
        else:
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
    deterministic=True,
    use_bn=use_bn
)
encoder_optim = Adam(encoder.parameters(), lr=encoder_lr)

base_map = BaseMap(
    z_dim, [1], [1], siamese_input=False,
    num_hidden_layers=2, hidden_dim=40,
    siamese_output=False, act='relu',
    deterministic=True,
    use_bn=use_bn
)
base_map_optim = Adam(base_map.parameters(), lr=base_map_lr)

r_to_z_map = GenericMap(
    [r_dim], [z_dim], siamese_input=False,
    num_hidden_layers=2, hidden_dim=40,
    siamese_output=False, act='relu',
    deterministic=False,
    use_bn=use_bn
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
        print('-'*80)
        print('Iter %d' % iter_num)
        neural_process.set_mode('eval')

        # get test samples
        NUM_TEST_SAMPLES = 30
        val_tasks = [SinusoidalTask() for _ in range(num_tasks_per_batch)]
        num_samples_per_task = array([NUM_TEST_SAMPLES for _ in range(num_tasks_per_batch)])
        X_test, Y_test = generate_data_batch(val_tasks, num_samples_per_task, NUM_TEST_SAMPLES)
        mask_test = generate_mask(num_samples_per_task, NUM_TEST_SAMPLES)
        batch_test = {
            'input_batch_list': [X_test],
            'output_batch_list': [Y_test],
            'mask': mask_test
        }

        for num_context in range(1,22,4):
            print('-'*5)
            num_samples_per_task = array([num_context for _ in range(num_tasks_per_batch)])
            X_context, Y_context = generate_data_batch(val_tasks, num_samples_per_task, num_context)
            mask_context = generate_mask(num_samples_per_task, num_context)
            batch_context = {
                'input_batch_list': [X_context],
                'output_batch_list': [Y_context],
                'mask': mask_context
            }
            posts = neural_process.infer_posterior_params(batch_context)

            elbo = neural_process.compute_ELBO(posts, batch_test)
            test_log_likelihood = neural_process.compute_cond_log_likelihood(posts, batch_test, mode='eval')

            print('%d Context:' % num_context)
            print('ELBO: %.4f' % elbo)
            print('Test Log Like: %.4f' % test_log_likelihood)

        neural_process.set_mode('train')
