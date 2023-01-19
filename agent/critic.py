import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils



class EnsembleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, num_q=2):
        super().__init__()

        self.Qs = nn.ModuleList([utils.q_net(obs_dim + action_dim, hidden_dim, 1, hidden_depth) for _ in range(num_q)])

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, idxs=None):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        if idxs is None:
            qs = [Q(obs_action) for Q in self.Qs]
        else: # pick qs according to idxs
            assert isinstance(idxs, (list, np.ndarray))
            qs = [self.Qs[i](obs_action) for i in idxs]
        self.outputs['q'] = qs[0]
        return qs

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)



class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.Q1 = utils.mlp(hidden_dim, hidden_dim, 1, hidden_depth-1)
        self.Q2 = utils.mlp(hidden_dim, hidden_dim, 1, hidden_depth-1)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        obs_action = self.trunk(obs_action)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
