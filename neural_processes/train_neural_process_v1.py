import torch
import torch.nn as nn
import torch.nn.function as F
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam

from base_map import BaseMap
from enc_free_neural_process import EncFreeNeuralProcess

from tasks.sinusoidal import SinusoidalTask
# -----------------------------------------------------------------------------
N_tasks = 100
z_dim = 10
base_map_lr = 1e-3
post_mean_lr = 1e-1
post_log_cov_lr = 1e-1

max_iters = 1e4
tasks_per_batch = 16
num_per_task = 10
num_z_samples = 10

# -----------------------------------------------------------------------------
tasks = [SinusoidalTask() for _ in range(N_tasks)]
z_means = Variable(torch.zeros(N_tasks,z_dim), requires_grad=True)
z_log_covs = Variable(torch.zeros(N_tasks,z_dim), requires_grad=True)

# -----------------------------------------------------------------------------
base_map = BaseMap(
    z_dim, [1], [1], siamese_input=False,
    num_hidden_layers=2, hidden_dim=40,
    siamese_output=False, act='relu',
    deterministic=True
)
base_map_optim = Adam(base_map.parameters(), lr=base_map_lr)
neural_process = EncFreeNeuralProcess(
    base_map, base_map_optim, use_nat_grad=True
)

# -----------------------------------------------------------------------------
for iter_num in range(max_iters):
    