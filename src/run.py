import hydra
import random
import sys
import torch
import wandb
import numpy as np

from omegaconf import DictConfig, OmegaConf

from .agents import VIPAgent, VIPAgentV2, VIPActorV2
from .algos import run_vip, run_vip_v2
from .coin_game import CoinGame
from .ipd import IPD

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(config_path="../scripts", config_name="config", version_base=None)
def main(args: DictConfig):
    config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)
    
    seed_all(config["seed"])

    wandb.init(config=config, dir="/network/scratch/j/juan.duque/wandb/", project="Co-games", reinit=True, anonymous="allow")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_window = config["reward_window"]
    env_type = config["env"]

    if env_type == "ipd":
        env = IPD(device)
        eval_env = IPD(device)
        model = IPD(device)
        n_actions = 2
        history_size = 4
    elif env_type == "cg":
        env = CoinGame(2, 1, device)
        eval_env = CoinGame(2, 1, device)
        model = CoinGame(2, 1, device)
        n_actions = 4
        history_size = 2

    obs, _ = env.reset()

    if config["agent_type"] == "vip":
        agent_1 = VIPAgent(config["base_agent"],
                           config["optim"],
                           **config["vip_agent"],
                           device=device,
                           n_actions=n_actions,
                           history_size=history_size,
                           obs_shape=obs.shape,
                           model=model)

        agent_2 = VIPAgent(config["base_agent"],
                           config["optim"],
                           **config["vip_agent"],
                           device=device,
                           n_actions=n_actions,
                           history_size=history_size,
                           obs_shape=obs.shape,
                           model=model)

        run_vip(env=env,
                eval_env=eval_env,
                env_type=env_type,
                obs=obs, 
                agent_1=agent_1, 
                agent_2=agent_2,  
                reward_window=reward_window, 
                device=device,
                num_episodes=config["num_episodes"])

    elif config["agent_type"] == "vip_v2":
        actors_1, actors_2, envs = [], [], []
        batch_size = config["vip_agent_v2"]["batch_size"]

        agent_1 = VIPAgentV2(config["base_agent"],
                             config["optim"],
                             **config["vip_agent_v2"],
                             device=device,
                             n_actions=n_actions,
                             history_size=history_size,
                             obs_shape=obs.shape,
                             model=model)

        agent_2 = VIPAgentV2(config["base_agent"],
                             config["optim"],
                             **config["vip_agent_v2"],
                             device=device,
                             n_actions=n_actions,
                             history_size=history_size,
                             obs_shape=obs.shape,
                             model=model)
                             
        for i in range(batch_size):
            actors_1.append(
                VIPActorV2(config["base_agent"],
                          **config["vip_actor_v2"],
                          device="cpu",
                          n_actions=n_actions,
                          history_size=history_size,
                          obs_shape=obs.shape,
                          hist_state_dict=agent_1.history_aggregator.state_dict(),
                          actor_state_dict=agent_1.actor.state_dict())
            )
            actors_2.append(
                VIPActorV2(config["base_agent"],
                          **config["vip_actor_v2"],
                          device="cpu",
                          n_actions=n_actions,
                          history_size=history_size,
                          obs_shape=obs.shape,
                          hist_state_dict=agent_2.history_aggregator.state_dict(),
                          actor_state_dict=agent_2.actor.state_dict())
            )
            envs.append(CoinGame(2, 1, "cpu"))

        agent_2.set_weights(agent_1)

        run_vip_v2(env=env,
                   eval_env=eval_env,
                   env_type=env_type,
                   obs=obs, 
                   agent_1=agent_1, 
                   agent_2=agent_2,  
                   reward_window=reward_window, 
                   device=device,
                   num_episodes=config["num_episodes"],
                   actors_1=actors_1,
                   actors_2=actors_2,
                   envs=envs)

if __name__ == "__main__":
    main()