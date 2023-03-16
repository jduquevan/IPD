import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from functools import reduce

from ipd import IPD
from models import DRLActor

class BaseAgent():
    def __init__(self,
                 device,
                 gamma,
                 n_actions,
                 hidden_size,
                 num_layers, 
                 obs_shape,
                 opt_type):
        self.steps_done = 0
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.obs_shape = obs_shape
        self.opt_type = opt_type
        
        self.obs_size = reduce(lambda a, b: a * b, self.obs_shape)

        self.obs_history = []
        self.act_history = []

    def update_history(self, act, state):
        self.act_history.insert(0, torch.tensor(act).to(self.device))
        self.obs_history.insert(0,  torch.tensor(state).to(self.device))
        self.act_history = self.act_history[0:self.history_len]
        self.obs_history = self.obs_history[0:self.history_len]


class DifferentiableRLAgent(BaseAgent):
    def __init__(self,
                 config,
                 optim_config,
                 steps_reset,
                 num_rollouts,
                 rollout_len,
                 communication_len,
                 device,
                 n_actions,
                 obs_shape):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape)

        self.steps_reset = steps_reset
        self.num_rollouts = num_rollouts
        self.rollout_len = rollout_len
        self.communication_len = communication_len
        self.n_actions = n_actions
        self.transition: list = list()

        self.actor = DRLActor(in_size=self.obs_size + 6,
                              out_size=self.n_actions,
                              device=self.device,
                              hidden_size=self.hidden_size)
        self.actor.to(self.device)
        self.model = IPD(self.device)

        if self.opt_type.lower() == "sgd":
            self.optimizer = optim.SGD(self.actor.parameters(), 
                                       lr=optim_config["lr"],
                                       momentum=optim_config["momentum"],
                                       weight_decay=optim_config["weight_decay"],
                                       maximize=True)
        elif self.opt_type.lower() == "adam":
            self.optimizer = optim.Adam(self.actor.parameters(), 
                                        lr=optim_config["lr"],
                                        weight_decay=optim_config["weight_decay"],
                                        maximize=True)

    def select_action(self, state, agent=None):
        self.steps_done += 1
        h_0, dist_a = self.actor(state)
        if not agent is None:
            dist_a, dist_b = self.unroll_policies(state, h_0, dist_a, agent)
        index = torch.tensor([np.random.choice(self.n_actions, p=dist_a.cpu().detach().numpy())],
                              requires_grad=False,
                              device=self.device)
        action = torch.zeros(self.n_actions).to(self.device)
        action = action.scatter(0, index, 1)
        return action

    def unroll_policies(self, state, h_0, dist_a, agent):
        j_0, dist_b = agent.actor(state, dist_a)
        for i in range(self.communication_len):
            h_0, dist_a = self.actor(state, dist_b, h_0)
            j_0, dist_b = agent.actor(state, dist_a, j_0)
        return dist_a, dist_b

    def optimize_model(self, agent):

        estimated_rewards = []

        # Monte-carlo rollouts TODO: Implement parallel rollouts
        for i in range(self.num_rollouts):
            t_rewards = []
            log_probs = []
            obs, last_actions = self.transition
            state = torch.cat([obs.flatten(), last_actions])

            for j in range(self.rollout_len):
                h_0, dist_a = self.actor(state)
                dist_a, dist_b = self.unroll_policies(state, h_0, dist_a, agent)
                index_a = torch.tensor([np.random.choice(self.n_actions, p=dist_a.cpu().detach().numpy())],
                              requires_grad=False,
                              device=self.device)
                index_b = torch.tensor([np.random.choice(self.n_actions, p=dist_b.cpu().detach().numpy())],
                              requires_grad=False,
                              device=self.device)
                action_a = torch.zeros(self.n_actions).to(self.device)
                action_b = torch.zeros(self.n_actions).to(self.device)
                action_a = action_a.scatter(0, index_a, 1)
                action_b = action_b.scatter(0, index_b, 1)

                a_t_prob = torch.take(dist_a, index_a)
                b_t_prob = torch.take(dist_b, index_b)
                log_probs.append(torch.log(a_t_prob))
                log_probs.append(torch.log(b_t_prob))

                obs, r1, r2, _, _, _  = self.model.step([action_a, action_b])
                t_rewards.append(r1)
                last_actions = torch.cat([action_a, action_b])
            reward_t = torch.sum(torch.cat(t_rewards, dim=0))
            sum_log_probs = torch.sum(torch.cat(log_probs, dim=0))

            # Reinforce estimator
            estimated_rewards.append((reward_t.detach() * sum_log_probs).unsqueeze(dim=0))

        game_value = torch.sum(torch.cat(estimated_rewards, dim=0))/self.num_rollouts

        self.optimizer.zero_grad()
        game_value.backward()
        self.optimizer.step()

        return game_value.detach()

