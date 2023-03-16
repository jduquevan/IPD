import hydra
import random
import torch
import wandb
import numpy as np

from itertools import count
from typing import Any, Dict, Optional

from utils import WandbLogger

def run_differentiable_rl(env, 
                          obs, 
                          agent_1, 
                          agent_2,  
                          reward_window, 
                          device,
                          num_episodes):

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

            value_1 = agent_1.optimize_model(agent_2)
            value_2 = agent_2.optimize_model(agent_1)

            logger.log_wandb_info(action_1, action_2, r1, r2, value_1, value_2)

