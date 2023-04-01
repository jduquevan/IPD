import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from functools import reduce

from .algos import get_reset_history
from .ipd import IPD
from .models import VIPActor, HistoryAggregator, RolloutBuffer
from .optimizers import ExtraAdam

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


class VIPAgent(BaseAgent):
    def __init__(self,
                 config,
                 optim_config,
                 steps_reset,
                 num_rollouts,
                 rollout_len,
                 communication_len,
                 representation_size,
                 history_len,
                 collab_weight,
                 exploit_weight,
                 entropy_weight,
                 device,
                 n_actions,
                 history_size,
                 obs_shape,
                 model):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape)
        self.cum_steps = 0
        self.steps_reset = steps_reset
        self.num_rollouts = num_rollouts
        self.rollout_len = rollout_len
        self.communication_len = communication_len
        self.representation_size =representation_size
        self.history_len = history_len
        self.exploit_weight = exploit_weight
        self.collab_weight = collab_weight
        self.entropy_weight = entropy_weight
        self.n_actions = n_actions
        self.history_size = history_size
        self.transition: list = list()
        self.history = []

        self.history_aggregator = HistoryAggregator(in_size=self.obs_size + 2*n_actions,
                                                    out_size=self.representation_size,
                                                    device=self.device,
                                                    hidden_size=self.hidden_size)
        self.actor = VIPActor(in_size=self.obs_size + representation_size + n_actions,
                              out_size=self.n_actions,
                              device=self.device,
                              hidden_size=self.hidden_size)
        self.history_aggregator.to(self.device)
        self.actor.to(self.device)
        self.model = model

        if self.opt_type.lower() == "sgd":
            self.optimizer = optim.SGD(list(self.actor.parameters()) + 
                                       list(self.history_aggregator.parameters()), 
                                       lr=optim_config["lr"],
                                       momentum=optim_config["momentum"],
                                       weight_decay=optim_config["weight_decay"],
                                       maximize=True)
        elif self.opt_type.lower() == "adam":
            self.optimizer = optim.Adam(list(self.actor.parameters()) + 
                                        list(self.history_aggregator.parameters()), 
                                        lr=optim_config["lr"],
                                        weight_decay=optim_config["weight_decay"],
                                        maximize=True)
        elif self.opt_type.lower() == "eg":
            self.optimizer = ExtraAdam(list(self.actor.parameters()) + 
                                       list(self.history_aggregator.parameters()),
                                       lr=optim_config["lr"],
                                       betas=(optim_config["beta_1"], optim_config["beta_2"]),
                                       weight_decay=optim_config["weight_decay"])
    
    def reset_history(self):
        self.history = []

    def set_history(self, history):
        self.history = history

    def update_history(self, a, b, obs):
        act_obs = torch.cat([a, b, obs.flatten()])
        self.history.append(act_obs)
        self.history = self.history[-self.history_len:]

    def aggregate_history(self):
        if self.history:
            history_tensor = torch.cat(self.history).to(self.device).reshape((1, len(self.history), -1))
            history = self.history_aggregator(history_tensor)
        else:
            history = -1 * torch.ones(self.representation_size).to(self.device)
        return history.flatten()

    def select_action(self, state, agent=None, dist_b=None, state_b=None):
        self.steps_done += 1
        h_0, dist_a = self.actor(state, dist_b)
        if not agent is None and not state_b is None:
            dist_a, dist_b = self.unroll_policies(state, h_0, dist_a, agent, state_b)
        elif not agent is None:
            dist_a, dist_b = self.unroll_policies(state, h_0, dist_a, agent)
        index = torch.tensor([np.random.choice(self.n_actions, p=dist_a.cpu().detach().numpy())],
                              requires_grad=False,
                              device=self.device)
        action = torch.zeros(self.n_actions).to(self.device)
        action = action.scatter(0, index, 1)
        return action

    def unroll_policies(self, state, h_0, dist_a, agent, state_b=None):
        j_0, dist_b = agent.actor(state, dist_a)
        for i in range(self.communication_len):
            h_0, dist_a = self.actor(state, dist_b, h_0)
            if not state_b is None:
                j_0, dist_b = agent.actor(state_b, dist_a, j_0)
            else:
                j_0, dist_b = agent.actor(state, dist_a, j_0)
        return dist_a, dist_b

    def compute_value(self, agent, env_type="ipd", communication=True):

        self.cum_steps = self.cum_steps + 1
        estimated_rewards = []

        # Monte-carlo rollouts TODO: Implement parallel rollouts
        for i in range(self.num_rollouts):

            steps = self.cum_steps
            t_rewards = []
            log_probs = []
            if env_type == "ipd":
                obs, history = self.transition
            elif env_type == "cg":
                obs, history_a, hist_a = self.transition
                _, history_b, hist_b = agent.transition

            for j in range(self.rollout_len):
                if env_type == "ipd":
                    state = torch.cat([obs.flatten(), history])
                elif env_type == "cg":
                    state_a = torch.cat([obs.flatten(), history_a])
                    state_b = torch.cat([obs.flatten(), history_b])

                if steps % self.steps_reset == 0 and env_type == "ipd":
                    history = get_reset_history(env_type, self.device)
                elif steps % self.steps_reset == 0 and env_type == "cg":
                    self.reset_history()
                    agent.reset_history()
                    history_a = self.aggregate_history()
                    history_b = agent.aggregate_history()

                if env_type == "ipd":
                    h_0, dist_a = self.actor(state)
                    if communication:
                        dist_a, dist_b = self.unroll_policies(state, h_0, dist_a, agent)
                    else:
                        j_0, dist_b = agent.actor(state, dist_a)
                        h_0, dist_a = self.actor(state, dist_b)
                elif env_type == "cg":
                    h_0, dist_a = self.actor(state_a)
                    if communication:
                        dist_a, dist_b = self.unroll_policies(state_a, h_0, dist_a, agent, state_b)
                    else:
                        j_0, dist_b = agent.actor(state_b, dist_a)
                        h_0, dist_a = self.actor(state_a, dist_b)
                uniform = np.ones(self.n_actions)/self.n_actions
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
                if not communication:
                    b_t_prob = b_t_prob.detach()

                entropy = self.entropy_weight * torch.dot(dist_a, torch.log(dist_a))
                log_probs.append(entropy.reshape(1))
                log_probs.append(self.exploit_weight * torch.log(a_t_prob))
                log_probs.append(self.collab_weight *torch.log(b_t_prob))


                last_obs = obs
                obs, r1, r2, _, _, _  = self.model.step([action_a, action_b])
                if env_type == "ipd":
                    history = torch.cat([action_a, action_b])
                elif env_type == "cg":
                    self.update_history(action_a, action_b, last_obs)
                    agent.update_history(action_b, action_a, last_obs)
                    history_a = self.aggregate_history()
                    history_b = agent.aggregate_history()

                t_rewards.append(r1)
                steps = steps + 1

            reward_t = torch.sum(torch.cat(t_rewards, dim=0))
            sum_log_probs = torch.sum(torch.cat(log_probs, dim=0))

            # Reinforce estimator
            estimated_rewards.append((reward_t.detach() * sum_log_probs).unsqueeze(dim=0))

        if env_type == "cg":
            self.set_history(hist_a)
            agent.set_history(hist_b)

        game_value = torch.sum(torch.cat(estimated_rewards, dim=0))/self.num_rollouts

        return game_value

class VIPAgentV2(BaseAgent):
    def __init__(self,
                 config,
                 optim_config,
                 batch_size,
                 steps_reset,
                 num_rollouts,
                 rollout_len,
                 communication_len,
                 representation_size,
                 history_len,
                 collab_weight,
                 exploit_weight,
                 entropy_weight,
                 epsilon,
                 device,
                 n_actions,
                 history_size,
                 obs_shape,
                 model):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape)
        self.batch_size = batch_size
        self.cum_steps = 0
        self.steps_reset = steps_reset
        self.num_rollouts = num_rollouts
        self.rollout_len = rollout_len
        self.communication_len = communication_len
        self.representation_size =representation_size
        self.history_len = history_len
        self.exploit_weight = exploit_weight
        self.collab_weight = collab_weight
        self.entropy_weight = entropy_weight
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.history_size = history_size
        self.transition: list = list()
        self.history = []
        self.memory = RolloutBuffer()

        self.history_aggregator = HistoryAggregator(in_size=self.obs_size + 2*n_actions,
                                                    out_size=self.representation_size,
                                                    device=self.device,
                                                    hidden_size=self.hidden_size)
        self.actor = VIPActor(in_size=self.obs_size + representation_size + n_actions,
                              out_size=self.n_actions,
                              device=self.device,
                              hidden_size=self.hidden_size)
        self.history_aggregator.to(self.device)
        self.actor.to(self.device)
        self.model = model

        if self.opt_type.lower() == "sgd":
            self.optimizer = optim.SGD(list(self.actor.parameters()) + 
                                       list(self.history_aggregator.parameters()), 
                                       lr=optim_config["lr"],
                                       momentum=optim_config["momentum"],
                                       weight_decay=optim_config["weight_decay"],
                                       maximize=True)
        elif self.opt_type.lower() == "adam":
            self.optimizer = optim.Adam(list(self.actor.parameters()) + 
                                        list(self.history_aggregator.parameters()), 
                                        lr=optim_config["lr"],
                                        weight_decay=optim_config["weight_decay"],
                                        maximize=True)
        elif self.opt_type.lower() == "eg":
            self.optimizer = ExtraAdam(list(self.actor.parameters()) + 
                                       list(self.history_aggregator.parameters()),
                                       lr=optim_config["lr"],
                                       betas=(optim_config["beta_1"], optim_config["beta_2"]),
                                       weight_decay=optim_config["weight_decay"])
    
    def reset_history(self):
        self.history = []

    def set_history(self, history):
        self.history = history

    def update_history(self, a, b, obs):
        act_obs = torch.cat([a, b, obs.flatten()])
        self.history.append(act_obs)
        self.history = self.history[-self.history_len:]

    def aggregate_history(self, history):
        return self.history_aggregator(history)

    def select_action(self, state, agent, dist_b, state_b, device):
        self.steps_done += 1
        h_0, dist_a = self.actor(state, dist_b).to(device)
        dist_a, dist_b = self.unroll_policies(state, h_0, dist_a, agent, state_b)
        index = torch.tensor([np.random.choice(self.n_actions, p=dist_a.cpu().detach().numpy())],
                              requires_grad=False,
                              device=device)
        action = torch.zeros(self.n_actions).to(self.device)
        action = action.scatter(0, index, 1)
        return action

    def unroll_policies(self, states, hists_a, hists_b, agent):
        states = states.reshape(self.batch_size * self.rollout_len, self.obs_size)
        hists_a = hists_a.reshape(self.batch_size * self.rollout_len, self.history_len, -1)
        hists_b = hists_b.reshape(self.batch_size * self.rollout_len, self.history_len, -1)
        state_reps_a = torch.cat([states, self.aggregate_history(hists_a)], dim=1)
        state_reps_b = torch.cat([states, self.aggregate_history(hists_b)], dim=1)

        h_0, dists_a = self.actor.batch_forward(state_reps_a, None)
        h_0 = h_0.reshape(1, self.batch_size * self.rollout_len, self.hidden_size)
        dists_a = dists_a.reshape(self.batch_size * self.rollout_len, self.n_actions)
        j_0, dists_b = self.actor.batch_forward(state_reps_a, dists_a)
        j_0 = j_0.reshape(1, self.batch_size * self.rollout_len, self.hidden_size)
        dists_b = dists_b.reshape(self.batch_size * self.rollout_len, self.n_actions)

        for i in range(self.communication_len):
            h_0, dists_a = self.actor.batch_forward(state_reps_a, dists_b, h_0)
            h_0 = h_0.reshape(1, self.batch_size * self.rollout_len, self.hidden_size)
            dists_a = dists_a.reshape(self.batch_size * self.rollout_len, self.n_actions)
            j_0, dists_b = agent.actor.batch_forward(state_reps_b, dists_a, j_0)
            j_0 = j_0.reshape(1, self.batch_size * self.rollout_len, self.hidden_size)
            dists_b = dists_b.reshape(self.batch_size * self.rollout_len, self.n_actions)

        return dists_a, dists_b

    def compute_surr_loss(self, agent):
        states = torch.stack(self.memory.states).to(self.device)
        rewards = torch.stack(self.memory.rewards).to(self.device)
        old_logprobs_a = torch.stack(self.memory.logprobs_a).to(self.device)
        old_logprobs_b = torch.stack(self.memory.logprobs_b).to(self.device)
        hists_a = torch.stack(self.memory.hists_a).to(self.device)
        hists_b = torch.stack(self.memory.hists_b).to(self.device)
        indices_a = torch.stack(self.memory.indices_a).to(self.device)
        indices_b = torch.stack(self.memory.indices_b).to(self.device)
        indices_a = indices_a.reshape(self.batch_size * self.rollout_len, 1)
        indices_b = indices_b.reshape(self.batch_size * self.rollout_len, 1)

        dists_a, dists_b = self.unroll_policies(states, hists_a, hists_b, agent)
        logprobs_a = torch.log(torch.gather(dists_a, 1, indices_a))
        logprobs_b = torch.log(torch.gather(dists_b, 1, indices_b))
        logprobs_a = logprobs_a.reshape(self.batch_size, self.num_rollouts, 1)
        logprobs_b = logprobs_b.reshape(self.batch_size, self.num_rollouts, 1)

        # Estimate Q value explicitly
        gammas = torch.tensor(self.gamma).repeat(self.batch_size, self.rollout_len - 1).to(self.device)
        gammas = torch.exp(torch.cumsum(torch.log(gammas), dim=1))
        gammas = torch.cat([torch.ones(self.batch_size, 1).to(self.device), gammas], dim=1)
        
        rewards = rewards.reshape(self.batch_size, self.rollout_len)
        traj_rewards = torch.flip(torch.cumsum(torch.flip(rewards, dims=(1,)), dim=1), dims=(1,))
        traj_rewards = gammas * traj_rewards

        ratios = torch.exp(self.exploit_weight * logprobs_a - old_logprobs_a.detach() + 
                           self.collab_weight * logprobs_b - old_logprobs_b.detach())
        ratios = ratios.reshape(self.batch_size, self.num_rollouts)

        surr_1 = ratios * traj_rewards
        # PPO style trust region
        surr_2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * traj_rewards

        # Entropy computation
        entropy_a = torch.bmm(torch.log(dists_a).reshape(-1, 1, self.n_actions),
                              dists_a.reshape(-1, self.n_actions, 1))
        entropy_b = torch.bmm(torch.log(dists_b).reshape(-1, 1, self.n_actions),
                              dists_b.reshape(-1, self.n_actions, 1))
        entropy = (entropy_a.reshape(self.batch_size *self.rollout_len) +
                   entropy_b.reshape(self.batch_size *self.rollout_len))

        loss = (torch.min(surr_1, surr_2).reshape(self.batch_size *self.rollout_len) + 
                self.entropy_weight * entropy)

        return loss.mean(), rewards.mean()

    def set_weights(self, agent):
        self.history_aggregator.load_state_dict(agent.history_aggregator.state_dict())
        self.actor.load_state_dict(agent.actor.state_dict())

class VIPActorV2(BaseAgent):
    def __init__(self,
                 config,
                 communication_len,
                 representation_size,
                 history_len,
                 device,
                 n_actions,
                 history_size,
                 obs_shape,
                 hist_state_dict,
                 actor_state_dict):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape)
        self.communication_len = communication_len
        self.representation_size =representation_size
        self.history_len = history_len
        self.n_actions = n_actions
        self.history_size = history_size
        self.history = []

        self.history_aggregator = HistoryAggregator(in_size=self.obs_size + 2*n_actions,
                                                    out_size=self.representation_size,
                                                    device=self.device,
                                                    hidden_size=self.hidden_size)
        self.actor = VIPActor(in_size=self.obs_size + representation_size + n_actions,
                              out_size=self.n_actions,
                              device=self.device,
                              hidden_size=self.hidden_size)
        self.history_aggregator.to(self.device)
        self.actor.to(self.device)

        self.history_aggregator.load_state_dict(hist_state_dict)
        self.actor.load_state_dict(actor_state_dict)

    def reset_history(self):
        self.history = []
        for i in range(self.history_len):
            self.history.append(-1 * torch.ones(2 * self.n_actions + self.obs_size).to(self.device))

    def set_history(self, history):
        self.history = history

    def update_history(self, a, b, obs):
        act_obs = torch.cat([a, b, obs.flatten()])
        self.history.append(act_obs)
        self.history = self.history[-self.history_len:]

    def aggregate_history(self):
        if self.history:
            history_tensor = torch.cat(self.history).to(self.device).reshape((1, len(self.history), -1))
            history = self.history_aggregator(history_tensor)
        else:
            history = -1 * torch.ones(self.representation_size).to(self.device)
        return history.flatten()

    def select_action(self, state, agent, dist_b, state_b, device):
        self.steps_done += 1
        h_0, dist_a = self.actor(state, dist_b)
        dist_a, dist_b = self.unroll_policies(state, h_0, dist_a, agent, state_b)
        index = torch.tensor([np.random.choice(self.n_actions, p=dist_a.cpu().detach().numpy())],
                              requires_grad=False,
                              device=device)
        action = torch.zeros(self.n_actions).to(self.device)
        action = action.scatter(0, index, 1)
        log_prob = torch.log(torch.take(dist_a, index))
        return action, log_prob, dist_a, index

    def unroll_policies(self, state, h_0, dist_a, agent, state_b=None):
        j_0, dist_b = agent.actor(state, dist_a)
        for i in range(self.communication_len):
            h_0, dist_a = self.actor(state, dist_b, h_0)
            if not state_b is None:
                j_0, dist_b = agent.actor(state_b, dist_a, j_0)
            else:
                j_0, dist_b = agent.actor(state, dist_a, j_0)
        return dist_a, dist_b

    def set_weights(self, agent):
        self.history_aggregator.load_state_dict(agent.history_aggregator.state_dict())
        self.actor.load_state_dict(agent.actor.state_dict())