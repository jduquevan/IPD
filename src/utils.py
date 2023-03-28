import wandb

class WandbLogger():
    def __init__(self, reward_window, env_type):
        self.reward_window = reward_window
        self.env_type = env_type
        self.cum_steps = 0
        self.d_score_1 = 0
        self.d_score_2 = 0
        self.c_score_1 = 0
        self.c_score_2 = 0
        self.avg_reward_1 = []
        self.avg_reward_2 = []
        self.adversity_1 = []
        self.adversity_2 = []

    def log_wandb_info(self, 
                      action_1, 
                      action_2,
                      r1, 
                      r2, 
                      value_1, 
                      value_2, 
                      d_score_1=None, 
                      c_score_1=None,
                      d_score_2=None,
                      c_score_2=None):
        self.cum_steps = self.cum_steps + 1
        self.avg_reward_1.insert(0, r1)
        self.avg_reward_2.insert(0, r2)

        self.avg_reward_1 = self.avg_reward_1[0:self.reward_window]
        self.avg_reward_2 = self.avg_reward_2[0:self.reward_window]

        avg_1 = sum(self.avg_reward_1)/len(self.avg_reward_1)
        avg_2 = sum(self.avg_reward_2)/len(self.avg_reward_2)

        if d_score_1:
            self.d_score_1 = d_score_1
        if d_score_2:
            self.d_score_2 = d_score_2
        if c_score_1:
            self.c_score_1 = c_score_1
        if c_score_2:
            self.c_score_2 = c_score_2

        if self.env_type == "ipd" and action_1[0].item():
            self.adversity_1.insert(0, 0)
        elif self.env_type == "ipd":
            self.adversity_1.insert(0, 1)

        if self.env_type == "ipd" and action_2[0].item():
            self.adversity_2.insert(0, 0)
        elif self.env_type == "ipd":
            self.adversity_2.insert(0, 1)

        self.adversity_1 = self.adversity_1[0:self.reward_window]
        self.adversity_2 = self.adversity_2[0:self.reward_window]

        adv_1 = sum(self.adversity_1)/self.reward_window
        adv_2 = sum(self.adversity_2)/self.reward_window

        wandb_info = {}
        wandb_info['cum_steps'] = self.cum_steps
        wandb_info['agent_1_avg_reward'] = avg_1
        wandb_info['agent_2_avg_reward'] = avg_2
        wandb_info['total_avg_reward'] = (avg_1 + avg_2)/2
        wandb_info['loss_1'] = value_1
        wandb_info['loss_2'] = value_2
        wandb_info['d_score_1'] = d_score_1
        wandb_info['d_score_2'] = d_score_2
        wandb_info['c_score_1'] = c_score_1
        wandb_info['c_score_2'] = c_score_2

        if self.env_type == "ipd":
            wandb_info['adversity_1'] = adv_1
            wandb_info['adversity_2'] = adv_2

        wandb.log(wandb_info)

    def log_wandb_info_v2(self, 
                        avg_1, 
                        avg_2, 
                        value_1, 
                        value_2):
        self.cum_steps = self.cum_steps + 1

        wandb_info = {}
        wandb_info['cum_steps'] = self.cum_steps
        wandb_info['agent_1_avg_reward'] = avg_1
        wandb_info['agent_2_avg_reward'] = avg_2
        wandb_info['total_avg_reward'] = (avg_1 + avg_2)/2
        wandb_info['loss_1'] = value_1
        wandb_info['loss_2'] = value_2

        wandb.log(wandb_info)