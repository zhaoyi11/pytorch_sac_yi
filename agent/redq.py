import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import distributions as pyd

from agent import Agent
import utils

import hydra


from agent.actor import DiagGaussianActor
from agent.critic import DoubleQCritic, EnsembleQCritic, Critic
# REDQ
class REDQAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature,
                 num_min, num_q, utd_ratio):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
    
        self.num_q = num_q
        self.critics = [Critic(**critic_cfg.params).to(self.device) for _ in range(self.num_q)]
        self.critics_target = [Critic(**critic_cfg.params).to(self.device) for _ in range(self.num_q)]
        for c, c_tar in zip(self.critics, self.critics_target):
            c_tar.load_state_dict(c.state_dict())
        
        self.actor = DiagGaussianActor(**actor_cfg.params).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        self.num_min = num_min
        self.utd_ratio = utd_ratio

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critics_optimizer = [torch.optim.Adam(c.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas) for c in self.critics]

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        for c_tar in self.critics_target:
            c_tar.train()


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        for c in self.critics:
            c.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger,
                      step):


        sampled_idxs = np.random.choice(self.num_q, self.num_min, replace=False)

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            
            q_preds = [self.critics_target[idx](next_obs, next_action) for idx in sampled_idxs]
            target_V = torch.min(*q_preds) - self.alpha.detach() * log_prob
            
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Qs = [self.critics[i](obs, action) for i in range(self.num_q)]
        critic_loss = sum([F.mse_loss(current_Q, target_Q) for current_Q in current_Qs])
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        for c_opt in self.critics_optimizer:
            c_opt.zero_grad()
        critic_loss.backward()

        for c_opt in self.critics_optimizer:
            c_opt.step()

        # logging
        for c in self.critics:
            c.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Qs = [c(obs, action) for c in self.critics]
        actor_Q = torch.stack(actor_Qs, 0).mean(0)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        for _ in range(self.utd_ratio):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            logger.log('train/batch_reward', reward.mean(), step)

            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                            logger, step)
  
            if step % self.critic_target_update_frequency == 0:
                for c, c_tar in zip(self.critics, self.critics_target):
                    utils.soft_update_params(c, c_tar, self.critic_tau)

        # only update actor every utd_ratio                               
        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)