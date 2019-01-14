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
from rlkit.core.train_util import linear_schedule


def concat_trajs(trajs):
    new_dict = {}
    for k in trajs[0].keys():
        if isinstance(trajs[0][k], dict):
            new_dict[k] = concat_trajs([t[k] for t in trajs])
        else:
            new_dict[k] = np.concatenate([t[k] for t in trajs], axis=0)
    return new_dict


class GAILWithTrajBatches(TorchIRLAlgorithm):
    '''
        This is actually AIRL / DAC, sorry!
        
        I did not implement the reward-wrapping mentioned in
        https://arxiv.org/pdf/1809.02925.pdf though
    '''
    def __init__(
            self,
            env,
            policy,
            discriminator,

            policy_optimizer,
            expert_replay_buffer,

            disc_num_expert_trajs_per_batch=16,
            disc_policy_batch_size=1024,
            policy_optim_batch_size=1000,

            disc_lr=1e-3,
            disc_momentum=0.0,
            disc_optimizer_class=optim.Adam,

            use_grad_pen=True,
            grad_pen_weight=10,

            disc_ce_grad_clip=0.5,
            disc_gp_grad_clip=10.0,

            use_target_disc=False,
            target_disc=None,
            soft_target_disc_tau=0.005,

            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,

            use_disc_input_noise=False,
            disc_input_noise_scale_start=0.1,
            disc_input_noise_scale_end=0.0,
            epochs_till_end_scale=50.0,
            **kwargs
    ):
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
            betas=(disc_momentum, 0.999)
        )
        print('\n\nDISC MOMENTUM: %f\n\n' % disc_momentum)

        self.disc_num_expert_trajs_per_batch = disc_num_expert_trajs_per_batch
        self.disc_policy_batch_size = disc_policy_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(self.max_path_length * self.disc_num_expert_trajs_per_batch, 1),
                torch.zeros(self.max_path_length * self.disc_num_expert_trajs_per_batch, 1)
            ],
            dim=0
        )
        self.bce_targets = Variable(self.bce_targets)
        if ptu.gpu_enabled():
            self.bce.cuda()
            self.bce_targets = self.bce_targets.cuda()
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.disc_ce_grad_clip = disc_ce_grad_clip
        self.disc_gp_grad_clip = disc_gp_grad_clip
        self.disc_grad_buffer = {}
        self.disc_grad_buffer_is_empty = True

        self.use_target_disc = use_target_disc
        self.soft_target_disc_tau = soft_target_disc_tau

        if use_target_disc:
            if target_disc is None:
                print('\n\nMAKING TARGET DISC\n\n')
                self.target_disc = self.discriminator.copy()
            else:
                print('\n\nUSING GIVEN TARGET DISC\n\n')
                self.target_disc = target_disc
        
        self.disc_ce_grad_norm = 0.0
        self.disc_ce_grad_norm_counter = 0.0
        self.max_disc_ce_grad = 0.0

        self.disc_gp_grad_norm = 0.0
        self.disc_gp_grad_norm_counter = 0.0
        self.max_disc_gp_grad = 0.0

        self.use_disc_input_noise = use_disc_input_noise
        self.disc_input_noise_scale_start = disc_input_noise_scale_start
        self.disc_input_noise_scale_end = disc_input_noise_scale_end
        self.epochs_till_end_scale = epochs_till_end_scale


    def get_expert_batch(self, num_trajs):
        batch = self.expert_replay_buffer.sample_trajs(num_trajs, keys=['observations', 'actions'])
        # for b in batch:
            # print(b['observations'].shape)
            # b['observations'] = b['observations'][:self.max_path_length]
            # b['actions'] = b['actions'][:self.max_path_length]

        batch = concat_trajs(batch)
        # if self.batch_size < batch['observations'].shape[0]:
        #     idx = np.random.choice(batch['observations'].shape[0], size=self.batch_size, replace=False)
        #     for k in batch:
        #         batch[k] = batch[k][idx]
        # elif self.batch_size > self.traj_len * self.num_trajs_per_update:
        #     print('batch_size is larger than the sampled batch')

        batch = np_to_pytorch_batch(batch)

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


    def _do_reward_training(self, epoch):
        '''
            Train the discriminator
        '''
        self.disc_optimizer.zero_grad()

        expert_batch = self.get_expert_batch(self.disc_num_expert_trajs_per_batch)
        expert_obs = expert_batch['observations']
        expert_actions = expert_batch['actions']

        policy_batch = self.get_policy_batch(self.disc_policy_batch_size)
        policy_obs = policy_batch['observations']
        policy_actions = policy_batch['actions']

        if self.use_disc_input_noise:
            noise_scale = linear_schedule(
                epoch,
                self.disc_input_noise_scale_start,
                self.disc_input_noise_scale_end,
                self.epochs_till_end_scale
            )
            if noise_scale > 0.0:
                expert_obs = expert_obs + noise_scale * Variable(torch.randn(expert_obs.size()))
                expert_actions = expert_actions + noise_scale * Variable(torch.randn(expert_actions.size()))

                policy_obs = policy_obs + noise_scale * Variable(torch.randn(policy_obs.size()))
                policy_actions = policy_actions + noise_scale * Variable(torch.randn(policy_actions.size()))
        
        obs = torch.cat([expert_obs, policy_obs], dim=0)
        actions = torch.cat([expert_actions, policy_actions], dim=0)

        disc_logits = self.discriminator(obs, actions)
        disc_preds = (disc_logits > 0).type(torch.FloatTensor)
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        disc_ce_loss.backward()

        ce_grad_norm = 0.0
        for name, param in self.discriminator.named_parameters():
            if param.grad is not None:
                if self.disc_grad_buffer_is_empty:
                    self.disc_grad_buffer[name] = param.grad.data.clone()
                else:
                    self.disc_grad_buffer[name].copy_(param.grad.data)
                
                param_norm = param.grad.data.norm(2)
                ce_grad_norm += param_norm ** 2
        ce_grad_norm = ce_grad_norm ** 0.5
        self.disc_grad_buffer_is_empty = False

        ce_clip_coef = self.disc_ce_grad_clip / (ce_grad_norm + 1e-6)
        if ce_clip_coef < 1.:
            for name, grad in self.disc_grad_buffer.items():
                grad.mul_(ce_clip_coef)
        
        if ce_clip_coef < 1.0: ce_grad_norm *= ce_clip_coef
        self.max_disc_ce_grad = max(ce_grad_norm, self.max_disc_ce_grad)
        self.disc_ce_grad_norm += ce_grad_norm
        self.disc_ce_grad_norm_counter += 1
        
        self.disc_optimizer.zero_grad()

        # # do gradient clipping for the cross-entropy loss gradients
        # torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), self.disc_grad_clip, norm_type=2)

        # check the magnitude of the gradients
        # this_ce_loss_grad = 0.0
        # for p in list(filter(lambda p: p.grad is not None, self.discriminator.parameters())):
        #     this_ce_loss_grad += p.grad.pow(2).sum()
        # this_ce_loss_grad = this_ce_loss_grad ** 0.5

        if self.use_grad_pen:
            eps = Variable(torch.rand(expert_obs.size(0), 1))
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

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight

            disc_grad_pen_loss.backward()

            gp_grad_norm = 0.0
            for p in list(filter(lambda p: p.grad is not None, self.discriminator.parameters())):
                param_norm = p.grad.data.norm(2)
                gp_grad_norm += param_norm ** 2
            gp_grad_norm = gp_grad_norm ** 0.5

            gp_clip_coef = self.disc_gp_grad_clip / (gp_grad_norm + 1e-6)
            if gp_clip_coef < 1.:
                for p in self.discriminator.parameters():
                    p.grad.data.mul_(gp_clip_coef)
            
            if gp_clip_coef < 1.: gp_grad_norm *= gp_clip_coef
            self.max_disc_gp_grad = max(gp_grad_norm, self.max_disc_gp_grad)
            self.disc_gp_grad_norm += gp_grad_norm
            self.disc_gp_grad_norm_counter += 1
        
        # now add back the gradients from the CE loss
        for name, param in self.discriminator.named_parameters():
            param.grad.data.add_(self.disc_grad_buffer[name])

        self.disc_optimizer.step()

        if self.use_target_disc:
            ptu.soft_update_from_to(self.discriminator, self.target_disc, self.soft_target_disc_tau)

        """
        Save some statistics for eval
        """
        if self.rewardf_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.rewardf_eval_statistics = OrderedDict()
            
            if self.use_target_disc:
                target_disc_logits = self.target_disc(obs, actions)
                target_disc_preds = (target_disc_logits > 0).type(torch.FloatTensor)
                target_disc_ce_loss = self.bce(target_disc_logits, self.bce_targets)
                target_accuracy = (target_disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

                if self.use_grad_pen:
                    eps = Variable(torch.rand(expert_obs.size(0), 1))
                    if ptu.gpu_enabled(): eps = eps.cuda()
                    
                    interp_obs = eps*expert_obs + (1-eps)*policy_obs
                    interp_obs.detach()
                    interp_obs.requires_grad = True
                    interp_actions = eps*expert_actions + (1-eps)*policy_actions
                    interp_actions.detach()
                    interp_actions.requires_grad = True
                    target_gradients = autograd.grad(
                        outputs=self.target_disc(interp_obs, interp_actions).sum(),
                        inputs=[interp_obs, interp_actions],
                        # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True
                    )
                    total_target_grad = torch.cat([target_gradients[0], target_gradients[1]], dim=1)

                    # GP from Gulrajani et al.
                    target_gradient_penalty = ((total_target_grad.norm(2, dim=1) - 1) ** 2).mean()

                    # # GP from Mescheder et al.
                    # target_gradient_penalty = (total_target_grad.norm(2, dim=1) ** 2).mean()

                self.rewardf_eval_statistics['Target Disc CE Loss'] = np.mean(ptu.get_numpy(target_disc_ce_loss))
                self.rewardf_eval_statistics['Target Disc Acc'] = np.mean(ptu.get_numpy(target_accuracy))
                self.rewardf_eval_statistics['Target Grad Pen'] = np.mean(ptu.get_numpy(target_gradient_penalty))
                self.rewardf_eval_statistics['Target Grad Pen W'] = np.mean(self.grad_pen_weight)
            
            self.rewardf_eval_statistics['Disc CE Loss'] = np.mean(ptu.get_numpy(disc_ce_loss))
            self.rewardf_eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))
            self.rewardf_eval_statistics['Grad Pen'] = np.mean(ptu.get_numpy(gradient_penalty))
            self.rewardf_eval_statistics['Grad Pen W'] = np.mean(self.grad_pen_weight)
            self.rewardf_eval_statistics['Disc Avg CE Grad Norm this epoch'] = np.mean(self.disc_ce_grad_norm / self.disc_ce_grad_norm_counter)
            self.rewardf_eval_statistics['Disc Max CE Grad Norm this epoch'] = np.mean(self.max_disc_ce_grad)
            self.rewardf_eval_statistics['Disc Avg GP Grad Norm this epoch'] = np.mean(self.disc_gp_grad_norm / self.disc_gp_grad_norm_counter)
            self.rewardf_eval_statistics['Disc Max GP Grad Norm this epoch'] = np.mean(self.max_disc_gp_grad)
            if self.use_disc_input_noise:
                self.rewardf_eval_statistics['Disc Input Noise Scale'] = noise_scale

            self.max_disc_ce_grad = 0.0
            self.disc_ce_grad_norm = 0.0
            self.disc_ce_grad_norm_counter = 0.0
            self.max_disc_gp_grad = 0.0
            self.disc_gp_grad_norm = 0.0
            self.disc_gp_grad_norm_counter = 0.0


    def _do_policy_training(self, epoch):
        policy_batch = self.get_policy_batch(self.policy_optim_batch_size)
        if self.use_target_disc:
            self.target_disc.eval()
            # If you compute log(D) - log(1-D) then you just get the logits
            policy_batch['rewards'] = self.target_disc(policy_batch['observations'], policy_batch['actions']).detach()
            self.target_disc.train()
        else:
            self.discriminator.eval()
            # If you compute log(D) - log(1-D) then you just get the logits
            policy_batch['rewards'] = self.discriminator(policy_batch['observations'], policy_batch['actions']).detach()
            self.discriminator.train()

        self.policy_optimizer.train_step(policy_batch)

        self.rewardf_eval_statistics['Disc Rew Mean'] = np.mean(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Std'] = np.std(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Max'] = np.max(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Min'] = np.min(ptu.get_numpy(policy_batch['rewards']))


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
