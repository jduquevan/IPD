import wandb

class WandbLogger():
    def __init__(self, reward_window):
        self.reward_window = reward_window
        self.cum_steps = 0
        self.avg_reward_1 = []
        self.avg_reward_2 = []
        self.adversity_1 = []
        self.adversity_2 = []

    def log_wandb_info(self, action_1, action_2,r1, r2, value_1, value_2):
        self.cum_steps = self.cum_steps + 1
        self.avg_reward_1.insert(0, r1)
        self.avg_reward_2.insert(0, r2)

        self.avg_reward_1 = self.avg_reward_1[0:self.reward_window]
        self.avg_reward_2 = self.avg_reward_2[0:self.reward_window]

        avg_1 = sum(self.avg_reward_1)/len(self.avg_reward_1)
        avg_2 = sum(self.avg_reward_2)/len(self.avg_reward_2)

        if action_1[0].item():
            self.adversity_1.insert(0, 0)
        else:
            self.adversity_1.insert(0, 1)

        if action_2[0].item():
            self.adversity_2.insert(0, 0)
        else:
            self.adversity_2.insert(0, 1)

        self.adversity_1 = self.adversity_1[0:self.reward_window]
        self.adversity_2 = self.adversity_2[0:self.reward_window]

        adv_1 = sum(self.adversity_1)/self.reward_window
        adv_2 = sum(self.adversity_2)/self.reward_window

        wandb_info = {}
        wandb_info['cum_steps'] = self.cum_steps
        wandb_info['agent_1_avg_reward'] = avg_1
        wandb_info['agent_2_avg_reward'] = avg_2
        wandb_info['total_avg_reward'] = avg_1 + avg_2
        wandb_info['value_1'] = value_1
        wandb_info['value_2'] = value_2
        wandb_info['adversity_1'] = adv_1
        wandb_info['adversity_2'] = adv_2

        wandb.log(wandb_info)
        