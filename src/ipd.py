import gym
import torch

import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding

class IPD(gym.Env):

    def __init__(self, device):
        self.device = device
        self.payout = torch.Tensor([[-1,-3],[0,-2]]).to(self.device)

    def step(self, actions):
        terminated = False
        a1, a2 = actions
        payout_T = torch.transpose(self.payout, 0, 1) 

        r1 = torch.mm(torch.mm(a1, self.payout), a2)
        r2 = torch.mm(torch.mm(a2, payout_T), a1)

        return self.payout, r1, r2, False, False, {}

    def reset(self):
        return self.payout, {}