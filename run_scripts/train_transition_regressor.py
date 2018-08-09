# Imports ---------------------------------------------------------------------
# Python
import argparse
import joblib
import yaml

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam

# Model Building
from neural_processes.generic_map import GenericMap

# Data
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer

# Vistools
from rlkit.core.vistools import save_plot, plot_returns_on_same_plot, plot_multiple_plots

def convert_numpy_dict_to_pytorch(np_dict):
    d = {
        k: torch.FloatTensor(v) for k,v in np_dict.items()
    }
    return d

def experiment(exp_specs):
    # Load the data -----------------------------------------------------------
    extra_data_path = exp_specs['extra_data_path']
    train_replay_buffer = joblib.load(extra_data_path)['replay_buffer']

    sample_batch = train_replay_buffer.random_batch(exp_specs['train_batch_size'])
    obs_dim = sample_batch['observations'].shape[-1]
    act_dim = sample_batch['actions'].shape[-1]

    val_replay_buffer = SimpleReplayBuffer(exp_specs['val_set_size'], obs_dim, act_dim)
    val_replay_buffer.set_buffer_from_dict(
        train_replay_buffer.sample_and_remove(exp_specs['val_set_size'])
    )

    # Model Definitions -------------------------------------------------------
    model = GenericMap(
        [obs_dim + act_dim],
        [obs_dim + 1],
        siamese_input=False,
        siamese_output=False,
        num_hidden_layers=exp_specs['num_hidden_layers'],
        hidden_dim=exp_specs['hidden_dim'],
        act='relu',
        use_bn=True,
        deterministic=True
    )

    model_optim = Adam(model.parameters(), lr=float(exp_specs['lr']))

    # Train -------------------------------------------------------------------
    model.train()
    for iter_num in range(exp_specs['max_iters']):
        model_optim.zero_grad()
        batch = train_replay_buffer.random_batch(exp_specs['train_batch_size'])
        batch = convert_numpy_dict_to_pytorch(batch)
        inputs = Variable(torch.cat([batch['observations'], batch['actions']], -1))
        outputs = Variable(torch.cat([batch['next_observations'], batch['rewards']], -1))

        preds = model([inputs])[0]
        # residual for observations
        # preds = preds + Variable(torch.cat([batch['observations'], torch.zeros(exp_specs['train_batch_size'], 1)], 1))
        
        loss = torch.mean(torch.sum((outputs - preds)**2, -1))
        loss.backward()
        model_optim.step()

        if iter_num % exp_specs['freq_val'] == 0:
            model.eval()

            val_batch = val_replay_buffer.random_batch(exp_specs['val_batch_size'])
            val_batch = convert_numpy_dict_to_pytorch(val_batch)
            inputs = Variable(torch.cat([val_batch['observations'], val_batch['actions']], -1))
            outputs = Variable(torch.cat([val_batch['next_observations'], val_batch['rewards']], -1))
            
            preds = model([inputs])[0]
            # residual for observations
            # pred = preds + Variable(torch.cat([val_batch['observations'], torch.zeros(exp_specs['train_batch_size'], 1)], 1))

            loss = torch.mean(torch.sum((outputs - preds)**2, -1))
            next_obs_loss = torch.mean(torch.sum((outputs[:,:-1] - preds[:,:-1])**2, -1))
            rew_loss = torch.mean(torch.sum((outputs[:,-1:] - preds[:,-1:])**2, -1))

            print('\nIter %d Loss: %.4f' % (iter_num, loss))
            print('Obs Loss: %.4f' % next_obs_loss)
            print('Rew Loss: %.4f' % rew_loss)

            model.train()


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    experiment(exp_specs)
