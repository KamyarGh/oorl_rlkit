from collections import OrderedDict

import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_irl_algorithm import TorchIRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic

# FOR EASY FETCH ENV ----------------------------------------------------------
# acts_max = Variable(ptu.from_numpy(np.array([0.11622048, 0.11837779, 1., 0.05])), requires_grad=False)
# acts_min = Variable(ptu.from_numpy(np.array([-0.11406593, -0.11492375, -0.48009082, -0.005])), requires_grad=False)

# obs_max = np.array([ 1.35211534e+00,  7.59012039e-01,  8.74170327e-01,  1.35216868e+00,
# 7.59075514e-01,  8.65117304e-01,  9.99349991e-03,  9.97504859e-03,
# -5.73782252e-04,  5.14756901e-02,  5.14743797e-02,  3.06240725e-03,
# 1.60782802e-02,  9.09377515e-03,  1.45024249e-03,  1.55772198e-03,
# 1.27349030e-02,  2.10399698e-02,  3.87118880e-03,  1.10660038e-02,
# 2.63549517e-03,  3.08370689e-03,  2.64278933e-02,  2.67708565e-02,
# 2.67707824e-02])
# obs_min = np.array([ 1.32694457e+00,  7.39177494e-01,  4.25007763e-01,  1.33124808e+00,
# 7.39111105e-01,  4.24235324e-01, -9.98595942e-03, -9.98935859e-03,
# -1.10015137e-01,  2.55108763e-06, -8.67902630e-08, -2.71974527e-03,
# -9.63782682e-03, -4.56146656e-04, -1.68586348e-03, -1.55750811e-03,
# -7.64317184e-04, -2.08764492e-02, -3.56580593e-03, -1.05306888e-02,
# -3.47314426e-03, -3.00819907e-03, -1.27082374e-02, -3.65293252e-03,
# -3.65292508e-03])
# goal_max = np.array([1.35216868, 0.75907551, 0.87419374])
# goal_min = np.array([1.33124808, 0.73911111, 0.42423532])
# observation_max = Variable(ptu.from_numpy(np.concatenate((obs_max, goal_max), axis=-1)), requires_grad=False)
# observation_min = Variable(ptu.from_numpy(np.concatenate((obs_min, goal_min), axis=-1)), requires_grad=False)

# SCALE = 0.99
# -----------------------------------------------------------------------------

# FOR SUPER EASY FETCH ENV ----------------------------------------------------

# -----------------------------------------------------------------------------


class GAIL(TorchIRLAlgorithm):
    '''
        This is actually AIRL / DAC, sorry!
        
        I did not implement the reward-wrapping mentioned in
        https://arxiv.org/pdf/1809.02925.pdf though
    '''
    # FOR SUPER EASY FETCH
    # acts_max = Variable(ptu.from_numpy(np.array([0.24968111, 0.24899998, 0.24999904, 0.01499934])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24993695, -0.24931063, -0.24999953, -0.01499993])), requires_grad=False)
    # observation_max = Variable(ptu.from_numpy(np.array([0.0152033 , 0.01572069, 0.00401832, 0.02023052, 0.03041435,
    #     0.20169743, 0.05092416, 0.05090878, 0.01017929, 0.01013457])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([-1.77039428e-02, -1.64070528e-02, -1.10015137e-01, -2.06485778e-02,
    #     -2.99603855e-02, -3.43990285e-03,  0.00000000e+00, -8.67902630e-08,
    #     -9.50872658e-03, -9.28206220e-03])), requires_grad=False)
    # SCALE = 0.99
    # ------------------------------
    # FOR FETCH (with Max-Ent Demos)
    observation_max = Variable(ptu.from_numpy(np.array([0.14997844, 0.14999457, 0.0066419 , 0.2896332 , 0.29748688,
       0.4510363 , 0.05095725, 0.05090321, 0.01027833, 0.01043796])), requires_grad=False)
    observation_min = Variable(ptu.from_numpy(np.array([-0.14985769, -0.14991582, -0.11001514, -0.29275747, -0.28962639,
       -0.01673591, -0.00056493, -0.00056452, -0.00953662, -0.00964976])), requires_grad=False)
    acts_max = Variable(ptu.from_numpy(np.array([0.24999679, 0.24999989, 0.24999854, 0.01499987])), requires_grad=False)
    acts_min = Variable(ptu.from_numpy(np.array([-0.24999918, -0.24999491, -0.24998883, -0.01499993])), requires_grad=False)
    SCALE = 0.99
    
    def __init__(
            self,
            env,
            policy,
            discriminator,

            policy_optimizer,
            expert_replay_buffer,

            disc_optim_batch_size=32,
            policy_optim_batch_size=1000,

            disc_lr=1e-3,
            disc_optimizer_class=optim.Adam,

            use_grad_pen=True,
            grad_pen_weight=10,

            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
            **kwargs
    ):
        assert disc_lr != 1e-3, 'Just checking that this is being taken from the spec file'
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        super().__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            expert_replay_buffer=expert_replay_buffer,
            policy_optimizer=policy_optimizer,
            **kwargs
        )

        self.discriminator = discriminator
        self.rewardf_eval_statistics = None
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
        )

        self.disc_optim_batch_size = disc_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(self.disc_optim_batch_size, 1),
                torch.zeros(self.disc_optim_batch_size, 1)
            ],
            dim=0
        )
        self.bce_targets = Variable(self.bce_targets)
        if ptu.gpu_enabled():
            self.bce.cuda()
            self.bce_targets = self.bce_targets.cuda()
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight


    def _normalize_obs(self, observation):
        observation = (observation - GAIL.observation_min) / (GAIL.observation_max - GAIL.observation_min)
        observation *= 2 * GAIL.SCALE
        observation -= GAIL.SCALE
        return observation


    def _normalize_acts(self, action):
        action = (action - GAIL.acts_min) / (GAIL.acts_max - GAIL.acts_min)
        action *= 2 * GAIL.SCALE
        action -= GAIL.SCALE
        return action


    def get_expert_batch(self, batch_size):
        batch = self.expert_replay_buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        batch['observations'] = self._normalize_obs(batch['observations'])
        batch['actions'] = self._normalize_acts(batch['actions'])
        batch['next_observations'] = self._normalize_obs(batch['next_observations'])

        if self.wrap_absorbing:
            if isinstance(batch['observations'], np.ndarray):
                obs = batch['observations']
                assert len(obs.shape) == 2
                batch['observations'] = np.concatenate((obs, np.zeros((obs.shape[0],1))), -1)
                if 'next_observations' in batch:
                    next_obs = batch['next_observations']
                    batch['next_observations'] = np.concatenate((next_obs, np.zeros((next_obs.shape[0],1))), -1)
            else:
                raise NotImplementedError()
        return batch
    

    def get_policy_batch(self, batch_size):
        batch = self.replay_buffer.random_batch(batch_size)
        return np_to_pytorch_batch(batch)


    def _do_reward_training(self):
        '''
            Train the discriminator
        '''
        self.disc_optimizer.zero_grad()

        expert_batch = self.get_expert_batch(self.disc_optim_batch_size)
        expert_obs = expert_batch['observations']
        expert_actions = expert_batch['actions']

        policy_batch = self.get_policy_batch(self.disc_optim_batch_size)
        policy_obs = policy_batch['observations']
        policy_actions = policy_batch['actions']

        obs = torch.cat([expert_obs, policy_obs], dim=0)
        actions = torch.cat([expert_actions, policy_actions], dim=0)

        disc_logits = self.discriminator(obs, actions)
        disc_preds = (disc_logits > 0).type(torch.FloatTensor)
        disc_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = Variable(torch.rand(self.disc_optim_batch_size, 1))
            if ptu.gpu_enabled(): eps = eps.cuda()
            
            interp_obs = eps*expert_obs + (1-eps)*policy_obs
            interp_obs.detach()
            interp_obs.requires_grad = True
            interp_actions = eps*expert_actions + (1-eps)*policy_actions
            interp_actions.detach()
            interp_actions.requires_grad = True
            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs, interp_actions).sum(),
                inputs=[interp_obs, interp_actions],
                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                create_graph=True, retain_graph=True, only_inputs=True
            )
            total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()

            disc_loss = disc_loss + gradient_penalty * self.grad_pen_weight

        disc_loss.backward()
        self.disc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.rewardf_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.rewardf_eval_statistics = OrderedDict()
            self.rewardf_eval_statistics['Disc Loss'] = np.mean(ptu.get_numpy(disc_loss))
            self.rewardf_eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))


    def _do_policy_training(self):
        policy_batch = self.get_policy_batch(self.policy_optim_batch_size)
        # If you compute log(D) - log(1-D) then you just get the logits
        policy_batch['rewards'] = self.discriminator(policy_batch['observations'], policy_batch['actions'])
        self.policy_optimizer.train_step(policy_batch)
    

    # if the discriminator has batch norm we have to use the following
    # def _do_policy_training(self):
    #     # this is a hack right now to avoid problems when using batchnorm
    #     # since if we use batchnorm the statistics of the disc will be messed up
    #     # if we only evaluate using policy samples
    #     expert_batch = self.get_expert_batch(self.policy_optim_batch_size)
    #     expert_obs = expert_batch['observations']
    #     expert_actions = expert_batch['actions']

    #     policy_batch = self.get_policy_batch(self.policy_optim_batch_size)
    #     policy_obs = policy_batch['observations']
    #     policy_actions = policy_batch['actions']

    #     obs = torch.cat([expert_obs, policy_obs], dim=0)
    #     actions = torch.cat([expert_actions, policy_actions], dim=0)

    #     # If you compute log(D) - log(1-D) then you just get the logits
    #     policy_batch['rewards'] = self.discriminator(obs, actions)[self.policy_optim_batch_size:]
    #     self.policy_optimizer.train_step(policy_batch)


    @property
    def networks(self):
        return [self.discriminator] + self.policy_optimizer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
        snapshot.update(self.policy_optimizer.get_snapshot())
        return snapshot


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
