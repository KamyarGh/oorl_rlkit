import numpy as np
from numpy.random import choice

import torch
from torch import nn as nn
from torch.autograd import Variable

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.distributions import TanhNormal, ReparamTanhMultivariateNormal
from rlkit.torch.networks import Mlp


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy


    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)
    
    def train(self, mode):
        pass
    

    def set_num_steps_total(self, num):
        pass



class DiscretePolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = DiscretePolicy(...)
    action, log_prob = policy(obs, return_log_prob=True)
    ```
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=nn.LogSoftmax(1),
            **kwargs
        )

    def get_action(self, obs_np, deterministic=False):
        action = self.get_actions(obs_np[None], deterministic=deterministic)
        return action, {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np)[0]

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
    ):
        log_probs, pre_act = super().forward(obs, return_preactivations=True)

        if deterministic:
            log_prob, idx = torch.max(log_probs, 1)
            return (idx, None)
        else:
            # Using Gumbel-Max trick to sample from the multinomials
            u = Variable(torch.rand(pre_act.size()))
            gumbel = -torch.log(-torch.log(u))
            _, idx = torch.max(gumbel + pre_act, 1)

            idx = torch.unsqueeze(idx, 1)
            log_prob = torch.gather(log_probs, 1, idx)

            # # print(log_probs.size(-1))
            # # print(log_probs.data.numpy())
            # # print(np.exp(log_probs.data.numpy()))
            # idx = choice(
            #     log_probs.size(-1),
            #     size=1,
            #     p=np.exp(log_probs.data.numpy())
            # )
            # log_prob = log_probs[0,idx]

            # print(idx)
            # print(log_prob)

            return (idx, log_prob)
    
    def get_log_pis(self, obs):
        return super().forward(obs)


class MlpPolicy(Mlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
    
    def get_action(self, obs_np, deterministic=False):
        '''
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        '''
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        actions = actions[None]
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np)[0]



class ReparamTanhMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = ReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_tanh_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
            else:
                action = tanh_normal.sample()

        # I'm doing it like this for now for backwards compatibility, sorry!
        if return_tanh_normal:
            return (
                action, mean, log_std, log_prob, expected_log_prob, std,
                mean_action_log_prob, pre_tanh_value, tanh_normal,
            )
        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )

    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std
        
        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class AntRandGoalCustomReparamTanhMultivariateGaussianPolicy(ReparamTanhMultivariateGaussianPolicy):
    """
    Custom for Ant Rand Goal
    The only difference is that it linearly embeds the goal into a higher dimension
    """
    def __init__(
            self,
            goal_dim,
            goal_embed_dim,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            obs_dim + goal_embed_dim,
            action_dim,
            init_w=init_w,
            **kwargs
        )

        self.goal_embed_fc = nn.Linear(goal_dim, goal_embed_dim)
        self.goal_dim = goal_dim
    
    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
        return_tanh_normal=False
    ):
        # obs will actuall be concat of [obs, goal]
        goal = obs[:,-self.goal_dim:]
        goal_embed = self.goal_embed_fc(goal)
        obs = torch.cat([obs[:,:-self.goal_dim], goal_embed], dim=-1)
        return super().forward(
            obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            return_tanh_normal=return_tanh_normal
        )


     
class PostCondMLPPolicyWrapper(ExplorationPolicy):
    def __init__(self, policy, np_post_sample, deterministic=False, obs_mean=None, obs_std=None):
        super().__init__()
        self.policy = policy
        self.np_z = np_post_sample # assuming it is a flat np array
        self.deterministic = deterministic
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        if obs_mean is not None:
            self.normalize_obs = True
        else:
            self.normalize_obs = False


    def get_action(self, obs_np):
        if self.normalize_obs:
            obs_np = (obs_np - self.obs_mean) / self.obs_std
        obs = np.concatenate((obs_np, self.np_z), axis=0)
        return self.policy.get_action(obs, deterministic=self.deterministic)


class ObsPreprocessedReparamTanhMultivariateGaussianPolicy(ReparamTanhMultivariateGaussianPolicy):
    '''
        This is a weird thing and I didn't know what to call.
        Basically I wanted this so that if you need to preprocess
        your inputs somehow (attention, gating, etc.) with an external module
        before passing to the policy you could do so.
        Assumption is that you do not want to update the parameters of the preprocessing
        module so its output is always detached.
    '''
    def __init__(self, preprocess_model, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # this is a hack so that it is not added as a submodule
        self.preprocess_model_list = [preprocess_model]
    

    @property
    def preprocess_model(self):
        # this is a hack so that it is not added as a submodule
        return self.preprocess_model_list[0]


    def preprocess_fn(self, obs_batch):
        mode = self.preprocess_model.training
        self.preprocess_model.eval()
        processed_obs_batch = self.preprocess_model(obs_batch, False).detach()
        self.preprocess_model.train(mode)
        return processed_obs_batch
    

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
        return_tanh_normal=False
    ):
        obs = self.preprocess_fn(obs).detach()
        return super().forward(
            obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            return_tanh_normal=return_tanh_normal
        )


    def get_log_prob(self, obs, acts):
        obs = self.preprocess_fn(obs).detach()
        return super().get_log_prob(obs, acts)


class WithZObsPreprocessedReparamTanhMultivariateGaussianPolicy(ReparamTanhMultivariateGaussianPolicy):
    '''
        This is a weird thing and I didn't know what to call.
        Basically I wanted this so that if you need to preprocess
        your inputs somehow (attention, gating, etc.) with an external module
        before passing to the policy you could do so.
        Assumption is that you do not want to update the parameters of the preprocessing
        module so its output is always detached.
    '''
    def __init__(self, preprocess_model, z_dim, *args, train_preprocess_model=False, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # this is a hack so that it is not added as a submodule
        if train_preprocess_model:
            self._preprocess_model = preprocess_model
        else:
            # this is a hack so that it is not added as a submodule
            self.preprocess_model_list = [preprocess_model]
        self.z_dim = z_dim
        self.train_preprocess_model = train_preprocess_model
    

    @property
    def preprocess_model(self):
        if self.train_preprocess_model:
            return self._preprocess_model
        else:
            # this is a hack so that it is not added as a submodule
            return self.preprocess_model_list[0]


    def preprocess_fn(self, obs_batch):
        if self.train_preprocess_model:
            processed_obs_batch = self.preprocess_model(
                obs_batch[:,:-self.z_dim],
                False,
                obs_batch[:,-self.z_dim:]
            )
        else:
            mode = self.preprocess_model.training
            self.preprocess_model.eval()
            processed_obs_batch = self.preprocess_model(
                obs_batch[:,:-self.z_dim],
                False,
                obs_batch[:,-self.z_dim:]
            ).detach()
            self.preprocess_model.train(mode)
        return processed_obs_batch
    

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
        return_tanh_normal=False
    ):
        if self.train_preprocess_model:
            obs = self.preprocess_fn(obs)
        else:
            obs = self.preprocess_fn(obs).detach()
        return super().forward(
            obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            return_tanh_normal=return_tanh_normal
        )


    def get_log_prob(self, obs, acts):
        if self.train_preprocess_model:
            obs = self.preprocess_fn(obs)
        else:
            obs = self.preprocess_fn(obs).detach()
        return super().get_log_prob(obs, acts)


# class PostCondReparamTanhMultivariateGaussianPolicy(ReparamTanhMultivariateGaussianPolicy):
#     '''
#         This is a very simple version of a policy that conditions on a sample from the posterior
#         I just concatenate the latent to the obs, so for now assuming everyting is flat
#     '''
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.z = None
    

#     def set_post_sample(self, z):
#         self.z = z
    

#     def forward(
#         self,
#         obs,
#         deterministic=False,
#         return_log_prob=False,
#         return_tanh_normal=False
#     ):
#         obs = torch.cat([obs, self.z], dim=-1)
#         return super().forward(
#             obs,
#             deterministic=deterministic,
#             return_log_prob=return_log_prob,
#             return_tanh_normal=return_tanh_normal
#         )


#     def get_log_prob(self, obs, acts):
#         obs = torch.cat([obs, self.z], dim=-1)
#         return super().get_log_prob(obs, acts)
