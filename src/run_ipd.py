import hydra
import random
import sys
import torch
import wandb
import numpy as np

from omegaconf import DictConfig, OmegaConf

from agents import DifferentiableRLAgent
from algos import run_differentiable_rl
from ipd import IPD

N_ACTIONS = 2

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(config_path="../scripts", config_name="ipd_config", version_base=None)
def main(args: DictConfig):
    config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)
    
    seed_all(config["seed"])

    wandb.init(config=config, dir="/network/scratch/j/juan.duque/wandb/", project="Co-games", reinit=True, anonymous="allow")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = IPD(device)
    obs, _ = env.reset()

    reward_window = config["reward_window"]

    if config["agent_type"] == "differentiable_rl":
        agent_1 = DifferentiableRLAgent(config["base_agent"],
                                        config["optim"],
                                        **config["drl_agent"],
                                        device=device,
                                        n_actions=N_ACTIONS,
                                        obs_shape=obs.shape)

        agent_2 = DifferentiableRLAgent(config["base_agent"],
                                        config["optim"],
                                        **config["drl_agent"],
                                        device=device,
                                        n_actions=N_ACTIONS,
                                        obs_shape=obs.shape)

        run_differentiable_rl(env=env, 
                              obs=obs, 
                              agent_1=agent_1, 
                              agent_2=agent_2,  
                              reward_window=reward_window, 
                              device=device,
                              num_episodes=config["num_episodes"])

if __name__ == "__main__":
    main()