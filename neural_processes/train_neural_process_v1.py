import torch
import torch.nn as nn
import torch.nn.function as F
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam

from numpy.random import choice, randint

from generic_map import GenericMap
from base_map import BaseMap
from neural_process import NeuralProcessV1
from aggregators import sum_aggregator, mean_aggregator, tanh_sum_aggregator

from tasks.sinusoidal import SinusoidalTask
# -----------------------------------------------------------------------------
N_tasks = 100
r_dim = 10
z_dim = 10
base_map_lr = 1e-3
encoder_lr = 1e-3
post_mean_lr = 1e-1
post_log_cov_lr = 1e-1
max_iters = 1e4
num_tasks_per_batch = 16

# data_sampling_mode = 'constant'
# num_per_task = 10

data_sampling_mode = 'random'
num_per_task_low = 3
num_per_task_high = 10

aggregator = tanh_sum_aggregator

freq_val = 100

# -----------------------------------------------------------------------------
all_tasks = [SinusoidalTask() for _ in range(N_tasks)]
def generate_data_batch(task_idxs, num_samples_per_task):
    X, Y = [], []
    for i in task_idxs:
        x, y = all_tasks[i].sample(num_samples_per_task[i])
        X.append(x)
        Y.append(y)
    X = Variable(torch.stack(X))
    Y = Variable(torch.stack(Y))
    return X, Y

def generate_mask(num_tasks_per_batch, max_num):
    mask = torch.ones(num_tasks_per_batch.shape[0], max_num, 1)
    for i, num in enumerate(num_tasks_per_batch):
        mask[i,num:] = 0.0
    return mask

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
    deterministic=True
)

neural_process = NeuralProcessV1(
    encoder,
    encoder_optim,
    aggregator,
    r_to_z_map,
    base_map,
    base_map_optim,
    use_nat_grad=False
)

# -----------------------------------------------------------------------------
for iter_num in range(max_iters):
    task_idxs = choice(idxs, size=num_tasks_per_batch, replace=False)
    if data_sampling_mode == 'random':
        num_samples_per_task = randint(
            num_per_task_low,
            high=num_per_task_high,
            size=(num_tasks_per_batch)
        )
        max_num = num_per_task_high
    else:
        raise NotImplementedError
    
    X, Y = get_data_batch(task_idxs, num_samples_per_task)
    mask = generate_mask(num_tasks_per_batch, max_num)
    batch = {
        'input_batch_list': [X],
        'output_batch_list': [Y],
        'mask': mask
    }
    neural_process.train_step(batch)

    if iter_num % freq_val == 0:
        val_task = SinusoidalTask()
        # inference batch
        # test batch
        # Check performance conditioning on range(1,22,4) number of points
        # Make the points subsets of one-another to be able to properly evaluate
