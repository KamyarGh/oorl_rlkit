import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

act_dict = {
    'relu': (F.relu, nn.ReLU),
    'tanh': (F.tanh, nn.Tanh),
    'None': (lambda x: x, nn.Identity)
}


def make_mlp(input_dim, hidden_dim, num_hidden_layers, act_nn, output_dim=None, output_act_nn=None):
    '''
        Makes an "tower" MLP
    '''
    assert num_hidden_layers > 0
    mod_list = nn.ModuleList(nn.Linear(input_dim, hidden_dim), act_nn)
    for _ in range(num_hidden_layers-1):
        mod_list.extend([nn.Linear(hidden_dim, hidden_dim), act_nn])
    if output_dim is not None:
        mod_list.extend([nn.Linear(hidden_dim, output_dim)])
        if output_act_nn is not None:
            mod_list.extend([output_act_nn])

    return nn.Sequential(mod_list)


class GenericMap(nn.Module):
    '''
        Assuming all inputs and outputs are flat

        if deterministic, output layer has no activation function
        if stochastic, outputs mean and **LOG** diag covariance for a Gaussian
    '''
    def __init__(
        self,
        input_dims,
        output_dims,
        siamese_input=True,
        num_siamese_input_layers=1,
        siamese_input_layer_dim=128,
        num_hidden_layers=1,
        hidden_dim=128,
        siamese_output=True,
        num_siamese_output_layers=1,
        siamese_output_layer_dim=128,
        act='relu',
        deterministic=False
        ):
        super(GenericMap).__init__(self)
        
        self.siamese_input = siamese_input
        self.siamese_output = siamese_output
        self.deterministic = deterministic
        act_fn, act_nn = act_dict[act]
        
        # process the inputs
        if siamese_input:
            assert num_siamese_input_layers > 0
            self.siamese_input_seqs = []
            for dim in input_dims:
                self.siamese_input_seqs.append(
                    make_mlp(
                        dim, siamese_input_layer_dim,
                        num_siamese_input_layers, act_nn
                    )
                )
        
        # pass through common hidden layers
        if siamese_input:
            concat_dim = len(input_dims) * siamese_input_layer_dim
        else:
            concat_dim = sum(input_dims)
        
        assert num_hidden_layers > 0
        self.hidden_seq = make_mlp(
            concat_dim, hidden_dim, num_hidden_layers, act_nn
        )

        # compute outputs
        if siamese_output:
            self.siamese_output_seqs = []
            for dim in output_dims:
                if not deterministic:
                    mean_seq = make_mlp(
                        hidden_dim, siamese_output_layer_dim,
                        num_siamese_output_layers, act_nn,
                        output_dim=dim, output_act_nn=None
                    )
                    log_cov_seq = make_mlp(
                        hidden_dim, siamese_output_layer_dim,
                        num_siamese_output_layers, act_nn,
                        output_dim=dim, output_act_nn=None
                    )
                    self.siamese_output_seqs.append((mean_seq, log_cov_seq))
                else:
                    self.siamese_output_seqs.append(
                        make_mlp(
                            hidden_dim, siamese_output_layer_dim,
                            num_siamese_output_layers, act_nn,
                            output_dim=dim, output_act_nn=None
                        )
                    )
        else:
            if deterministic:
                self.output_seq = nn.Linear(hidden_dim, output_dims[0])
            else:
                self.output_mean_seq = nn.Linear(hidden_dim, output_dims[0])
                self.output_log_cov_seq = nn.Linear(hidden_dim, output_dims[0])


    def forward(inputs):
        '''
            Output is:
                deterministic: a list
                not: a list of lists
        '''
        if self.siamese_input:
            siamese_input_results = list(
                map(
                    lambda x, seq: seq(x),
                    zip(inputs, self.siamese_input_seqs)
                )
            )
            hidden_input = torch.cat(siamese_input_results, dim=1)
        else:
            hidden_input = torch.cat(inputs, dim=1)=
        
        hidden_output = self.hidden_seq(hidden_input)

        if self.siamese_output:
            if self.deterministic:
                outputs = [
                    seq(hidden_output) for seq in self.siamese_output_seqs
                ]
            else:
                outputs = [
                    [mean_seq(hidden_output), log_cov_seq(hidden_output)] \
                    for mean_seq, log_cov_seq in self.siamese_output_seqs
                ]
        else:
            if self.deterministic:
                outputs = [self.output_seq(hidden_output)]
            else:
                outputs = [
                    self.output_mean_seq(hidden_output),
                    self.output_log_cov_seq(hidden_output)
                ]
        
        return outputs