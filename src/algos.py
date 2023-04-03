import hydra
import random
import torch
import wandb
import numpy as np

from itertools import count
from multiprocessing import Pool
from typing import Any, Dict, Optional

from .optimizers import ExtraAdam
from .utils import WandbLogger

def optimize_models(opt_type, opt_1, opt_2, loss_1, loss_2, t):
    if opt_type == "sgd" or opt_type == "adam":
        opt_1.zero_grad()
        opt_2.zero_grad()
        loss_1.backward()
        loss_2.backward()
        opt_1.step()
        opt_2.step()
    elif opt_type == "eg":
        loss_1 = -1 * loss_1
        loss_2 = -1 * loss_2 
        opt_1.zero_grad()
        opt_2.zero_grad()
        loss_1.backward(retain_graph=True)
        loss_2.backward(retain_graph=True)
        if t % 2 == 0:
            opt_1.extrapolation()
            opt_2.extrapolation()
        else:
            opt_1.step()
            opt_2.step()

def get_reset_history(env, device, agent=None):
    if env == "ipd":
        return torch.tensor([-1, -1, -1, -1], device=device)
    elif env == "cg":
        agent.reset_history()
        return agent.aggregate_history()

def evaluate_agents(agent_1, agent_2, evaluation_steps, eval_env, env_type, device):
    # TODO: Implement always cooperate and defect agents for IPD and CG
    d_score_1 = evaluate_agent(agent_1, 
                               evaluation_steps, 
                               eval_env,
                               env_type, 
                               device, 
                               torch.FloatTensor([0, 1]).to(device))
    c_score_1 = evaluate_agent(agent_1, 
                               evaluation_steps, 
                               eval_env,
                               env_type, 
                               device, 
                               torch.FloatTensor([1, 0]).to(device))
    d_score_2 = evaluate_agent(agent_2, 
                               evaluation_steps, 
                               eval_env,
                               env_type, 
                               device, 
                               torch.FloatTensor([0, 1]).to(device))
    c_score_2 = evaluate_agent(agent_2, 
                               evaluation_steps, 
                               eval_env,
                               env_type, 
                               device, 
                               torch.FloatTensor([1, 0]).to(device))
    return d_score_1, c_score_1, d_score_2, c_score_2
    

def evaluate_agent(agent, evaluation_steps, env, env_type, device, pi):
    scores = []
    obs, _ = env.reset()
    history = get_reset_history(env_type, device)

    for i in range(evaluation_steps):
        agent.transition = [obs, history]

        state = torch.cat([obs.flatten(), history])
        action_1 = agent.select_action(state, dist_b=pi)

        obs, r1, r2, _, _, _  = env.step([action_1, pi])
        if env_type == "ipd":
            history = torch.cat([action_1,  pi])
        elif env_type == "cg":
            history = torch.cat([env.j1, env.j2])

        scores.append(r1.detach().cpu().numpy().item())
    score = sum(scores[1:])/(evaluation_steps-1)
        
    return score

def run_vip(env,
            eval_env,
            env_type,
            obs, 
            agent_1, 
            agent_2,  
            reward_window, 
            device,
            num_episodes,
            evaluate_every=10,
            evaluation_steps=10):

    logger = WandbLogger(reward_window, env_type)
    steps_reset = agent_1.steps_reset

    for i_episode in range(num_episodes):
        obs, _ = env.reset()

        if env_type == "ipd":
            history = get_reset_history(env_type, device)
        elif env_type == "cg":
            history_1 = get_reset_history(env_type, device, agent_1)
            history_2 = get_reset_history(env_type, device, agent_2)
        
        for t in count():
            if t % steps_reset == 0 and env_type == "ipd":
                history = get_reset_history(env_type, device)
            elif t % steps_reset == 0 and env_type == "cg":
                history_1 = get_reset_history(env_type, device, agent_1)
                history_2 = get_reset_history(env_type, device, agent_2)

            if env_type == "ipd":
                agent_1.transition = [obs, history]
                agent_2.transition = [obs, history]
                state = torch.cat([obs.flatten(), history])
                action_1 = agent_1.select_action(state, agent_2)
                action_2 = agent_2.select_action(state, agent_1)
            elif env_type == "cg":
                agent_1.transition = [obs, history_1, agent_1.history]
                agent_2.transition = [obs, history_2, agent_2.history]
                state_1 = torch.cat([obs.flatten(), history_1])
                state_2 = torch.cat([obs.flatten(), history_2])
                action_1 = agent_1.select_action(state_1, agent_2, state_b=state_2)
                action_2 = agent_2.select_action(state_2, agent_1, state_b=state_1)
            
            last_obs = obs
            obs, r1, r2, _, _, _  = env.step([action_1, action_2])

            if env_type == "ipd":
                history = torch.cat([action_1, action_2])
            elif env_type == "cg":
                agent_1.update_history(action_1, action_2, last_obs)
                agent_2.update_history(action_2, action_1, last_obs)
                history_1 = agent_1.aggregate_history()
                history_2 = agent_2.aggregate_history()

            if t % 4 == 0 or t % 4 == 1:
                value_1 = agent_1.compute_value(agent_2, env_type=env_type)
                value_2 = agent_2.compute_value(agent_1, env_type=env_type, agent_t=2)
            else:
                value_1 = agent_1.compute_value(agent_2, env_type=env_type, communication=False)
                value_2 = agent_2.compute_value(agent_1, env_type=env_type, communication=False, agent_t=2)

            optimize_models(agent_1.opt_type, 
                            agent_1.optimizer, 
                            agent_2.optimizer,
                            value_1,
                            value_2,
                            t)

            d_1, c_1, d_2, c_2 = None, None, None, None
            if t % evaluate_every == 0 and env_type == "ipd":
                d_1, c_1, d_2, c_2 = evaluate_agents(agent_1, 
                                                     agent_2, 
                                                     evaluation_steps,
                                                     eval_env,
                                                     env_type, 
                                                     device)

            logger.log_wandb_info(action_1, 
                                  action_2, 
                                  r1, 
                                  r2, 
                                  value_1, 
                                  value_2,
                                  d_score_1=d_1,
                                  c_score_1=c_1,
                                  d_score_2=d_2,
                                  c_score_2=c_2,
                                  obs=obs)

def run_vip_v2(env,
               eval_env,
               env_type,
               obs, 
               agent_1, 
               agent_2,  
               reward_window, 
               device,
               num_episodes,
               actors_1,
               actors_2,
               envs,
               evaluate_every=10,
               evaluation_steps=10,
               num_epochs=8):

    logger = WandbLogger(reward_window, env_type)
    batch_size = agent_1.batch_size
    steps_reset = agent_1.steps_reset

    # For debugging
    # compute_trajectory(actors_1[0], actors_2[0], envs[0], steps_reset)

    for i_episode in range(num_episodes):
        actors_and_envs = [(actors_1[i], actors_2[i], envs[i], steps_reset)
                       for i in range(batch_size)]
        # Runs trajectories in parallel (cpu)
        with Pool() as pool:
            result = pool.starmap(compute_trajectory, actors_and_envs)
        
        set_memories(result, agent_1, agent_2, batch_size)

        loss_1, reward_1 = agent_1.compute_surr_loss(agent_2)
        loss_2, reward_2 = agent_2.compute_surr_loss(agent_1)

        logger.log_wandb_info_v2(reward_1, reward_2, loss_1, loss_2)

        optimize_models(agent_1.opt_type, 
                        agent_1.optimizer, 
                        agent_2.optimizer,
                        loss_1,
                        loss_2,
                        i_episode)

        if i_episode % 2 == 1:
            for k in range(batch_size):
                actors_1[k].set_weights(agent_1)
                actors_2[k].set_weights(agent_2)
        
        #TODO: Update environments
        
def compute_trajectory(agent_1, agent_2, env, steps_reset):
    np.random.seed()
    states_1, rewards_1, logprobs_1, dists_1, hists_1, indices_1 = [], [], [], [], [], []
    states_2, rewards_2, logprobs_2, dists_2, hists_2, indices_2 = [], [], [], [], [], []
    
    obs, _ = env.reset()
    history_1 = get_reset_history("cg", "cpu", agent_1)
    history_2 = get_reset_history("cg", "cpu", agent_2)

    for j in range(steps_reset):
        agent_1.transition = [obs, history_1, agent_1.history]
        agent_2.transition = [obs, history_2, agent_2.history]
        state_1 = torch.cat([obs.flatten(), history_1])
        state_2 = torch.cat([obs.flatten(), history_2])

        states_1.append(obs.flatten())
        states_2.append(obs.flatten())

        hists_1.append(torch.stack(agent_1.history))
        hists_2.append(torch.stack(agent_2.history))

        action_1, logprob_1, dist_1, index_1 = agent_1.select_action(state=state_1, 
                                                                     agent=agent_2, 
                                                                     dist_b=None, 
                                                                     state_b=state_2, 
                                                                     device="cpu")
        action_2, logprob_2, dist_2, index_2 = agent_2.select_action(state=state_2, 
                                                                     agent=agent_1, 
                                                                     dist_b=None, 
                                                                     state_b=state_1, 
                                                                     device="cpu")

        logprobs_1.append(logprob_1.detach())
        logprobs_2.append(logprob_2.detach())
        dists_1.append(dist_1.detach())
        dists_2.append(dist_2.detach())
        indices_1.append(index_1.detach())
        indices_2.append(index_2.detach())

        last_obs = obs
        obs, r1, r2, _, _, _  = env.step([action_1, action_2])

        rewards_1.append(r1.detach())
        rewards_2.append(r2.detach())

        agent_1.update_history(action_1, action_2, last_obs)
        agent_2.update_history(action_2, action_1, last_obs)
        history_1 = agent_1.aggregate_history()
        history_2 = agent_2.aggregate_history()

    return torch.stack(states_1), \
           torch.stack(rewards_1), \
           torch.stack(logprobs_1), \
           torch.stack(dists_1), \
           torch.stack(hists_1), \
           torch.stack(states_2), \
           torch.stack(rewards_2), \
           torch.stack(logprobs_2), \
           torch.stack(dists_2), \
           torch.stack(hists_2), \
           torch.stack(indices_1), \
           torch.stack(indices_2)

def set_memories(result, agent_1, agent_2, batch_size):
    agent_1.memory.states = [result[i][0] for i in range(batch_size)]
    agent_1.memory.rewards = [result[i][1] for i in range(batch_size)]
    agent_1.memory.logprobs_a = [result[i][2] for i in range(batch_size)]
    agent_1.memory.logprobs_b = [result[i][7] for i in range(batch_size)]
    agent_1.memory.dists = [result[i][3] for i in range(batch_size)]
    agent_1.memory.hists_a = [result[i][4] for i in range(batch_size)]
    agent_1.memory.hists_b = [result[i][9] for i in range(batch_size)]
    agent_1.memory.indices_a = [result[i][10] for i in range(batch_size)]
    agent_1.memory.indices_b = [result[i][11] for i in range(batch_size)]

    agent_2.memory.states = [result[i][5] for i in range(batch_size)]
    agent_2.memory.rewards = [result[i][6] for i in range(batch_size)]
    agent_2.memory.logprobs_a = [result[i][7] for i in range(batch_size)]
    agent_2.memory.logprobs_b = [result[i][2] for i in range(batch_size)]
    agent_2.memory.dists = [result[i][8] for i in range(batch_size)]
    agent_2.memory.hists_a = [result[i][9] for i in range(batch_size)]
    agent_2.memory.hists_b = [result[i][4] for i in range(batch_size)]
    agent_2.memory.indices_a = [result[i][11] for i in range(batch_size)]
    agent_2.memory.indices_b = [result[i][10] for i in range(batch_size)]

def run_vip_v3(env,
               eval_env,
               env_type,
               obs, 
               agent_1, 
               agent_2,  
               reward_window, 
               device,
               num_episodes,
               evaluate_every=10,
               evaluation_steps=10):

    logger = WandbLogger(reward_window, env_type)
    steps_reset = agent_1.steps_reset

    for i_episode in range(num_episodes):
        obs, _ = env.reset()
        
        for t in count():
            if t % steps_reset == 0:
                history_1 = None
                history_2 = None
                action_1, dist_1 = agent_1.get_empty_action()
                action_2, dist_2 = agent_2.get_empty_action()

            agent_1.transition = [obs, history_1, history_2, action_1, dist_1, action_2, dist_2]
            agent_2.transition = [obs, history_2, history_1, action_2, dist_2, action_1, dist_1]
            dists_1 = torch.cat([action_2, dist_2])
            dists_2 = torch.cat([action_1, dist_1])
            action_1, history_1, dist_1 = agent_1.select_action(obs.flatten(), 
                                                                dist_b=dists_1, 
                                                                h_0=history_1)
            action_2, history_2, dist_2 = agent_2.select_action(obs.flatten(), 
                                                                dist_b=dists_2,
                                                                h_0=history_2)
            
            obs, r1, r2, _, _, _  = env.step([action_1, action_2])

            if t % 4 == 0 or t % 4 == 1:
                value_1 = agent_1.compute_value(agent_2)
                value_2 = agent_2.compute_value(agent_1, communication=False, agent_type=2)
            else:
                value_1 = agent_1.compute_value(agent_2, communication=False)
                value_2 = agent_2.compute_value(agent_1, agent_type=2)

            optimize_models(agent_1.opt_type, 
                            agent_1.optimizer, 
                            agent_2.optimizer,
                            value_1,
                            value_2,
                            t)

            d_1, c_1, d_2, c_2 = None, None, None, None

            logger.log_wandb_info(action_1, 
                                  action_2, 
                                  r1, 
                                  r2, 
                                  value_1, 
                                  value_2,
                                  d_score_1=d_1,
                                  c_score_1=c_1,
                                  d_score_2=d_2,
                                  c_score_2=c_2,
                                  obs=obs)