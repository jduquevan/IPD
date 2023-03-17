import hydra
import random
import torch
import wandb
import numpy as np

from itertools import count
from typing import Any, Dict, Optional

from .optimizers import ExtraAdam
from .utils import WandbLogger

def optimize_models(opt_type, opt_1, opt_2, loss_1, loss_2):
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
        loss_1.backward()
        loss_2.backward()
        opt_1.extrapolation()
        opt_2.extrapolation()
        opt_1.step()
        opt_2.step()

def evaluate_agent(agent):
    pass

def run_differentiable_rl(env, 
                          obs, 
                          agent_1, 
                          agent_2,  
                          reward_window, 
                          device,
                          num_episodes,
                          evaluate_every=10,
                          evaluation_steps=10):

    logger = WandbLogger(reward_window)
    steps_reset = agent_1.steps_reset

    for i_episode in range(num_episodes):
        obs, _ = env.reset()
        last_actions = torch.tensor([-1, -1, -1, -1], device=device)
        for t in count():
            if t % steps_reset == 0:
                last_actions = torch.tensor([-1, -1, -1, -1], device=device)
            agent_1.transition = [obs, last_actions]
            agent_2.transition = [obs, last_actions]

            state = torch.cat([obs.flatten(), last_actions])
            action_1 = agent_1.select_action(state, agent_2)
            action_2 = agent_2.select_action(state, agent_1)

            obs, r1, r2, _, _, _  = env.step([action_1, action_2])
            last_actions = torch.cat([action_1, action_2])

            value_1 = agent_1.compute_value(agent_2)
            value_2 = agent_2.compute_value(agent_1)

            optimize_models(agent_1.opt_type, 
                            agent_1.optimizer, 
                            agent_2.optimizer,
                            value_1,
                            value_2)

            logger.log_wandb_info(action_1, action_2, r1, r2, value_1, value_2)

