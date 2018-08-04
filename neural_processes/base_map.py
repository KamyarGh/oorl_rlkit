from neural_processes.generic_map import GenericMap

class BaseMap(GenericMap):
    '''
        Parametrizing the mapping used in Neural Processes

        f = BaseMap(..., deterministic=True)
        y = f(x,z)
        -- or --
        f = BaseMap(..., deterministic=False)
        y_mean, y_std = f(x,z)

        The PETS paper (data efficient model-based RL...)
        gets best results when the models are not
        deterministic

        Assuming all inputs and outputs are flat

        if deterministic, output layer has no activation function
        if stochastic, outputs mean and **LOG** diag covariance for a Gaussian
    '''
    def __init__(
        self,
        z_dim,
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
        deterministic=False,
        use_bn=False
    ):
        
        all_input_dims = [z_dim] + input_dims
        super(BaseMap, self).__init__(
            all_input_dims,
            output_dims,
            siamese_input=siamese_input,
            num_siamese_input_layers=num_siamese_input_layers,
            siamese_input_layer_dim=siamese_input_layer_dim,
            num_hidden_layers=num_hidden_layers,
            hidden_dim=hidden_dim,
            siamese_output=siamese_output,
            num_siamese_output_layers=num_siamese_output_layers,
            siamese_output_layer_dim=siamese_output_layer_dim,
            act=act,
            deterministic=deterministic,
            use_bn=use_bn
        )

    def forward(self, z, inputs):
        '''
            Output is:
                deterministic: a list
                not: a list of lists
        '''
        all_inputs = [z] + inputs
        
        return super(BaseMap, self).forward(all_inputs)
