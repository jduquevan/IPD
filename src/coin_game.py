import gym
import torch

import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding
from gym.error import DependencyNotInstalled

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

EMPTY = 0
COIN_1 = 1
COIN_2 = 2
AGENT_1 = 3
AGENT_2 = 4
BOTH = 5
BOTH_AND_COIN = 6

BLOCKSIZE = 100

class CoinGame(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size_h, size_v, device, default=True, positions=None, render_mode=None):
        if size_h < 2 or size_v < 1:
            raise Exception("Invalid world size")
        
        self.rows = size_v
        self.cols = size_h
        self.device = device
        self.default = default
        self.grid = np.zeros((self.rows, self.cols))
        self.curr_coin = COIN_1 if np.random.binomial(1, 0.5) else COIN_2

        self.init_pos = self.init_grid(positions)
        self.coin, self.a_1, self.a_2 = self.init_pos

        self.screen_width = size_h * BLOCKSIZE
        self.screen_height = size_v * BLOCKSIZE

        self.j1 = torch.Tensor([-1]).to(self.device)
        self.j2 = torch.Tensor([-1]).to(self.device)

        self.screen = None
        self.clock = None

        self.render_mode = render_mode

    def init_grid(self, positions=None):
        if positions is not None:
            pass
        elif self.rows==1 and self.cols==2:
            positions = np.array([0, 1, 1])
        elif self.default and self.rows==3 and self.cols==3:
            positions = np.array([4, 0, 2])
        else:
            positions = np.random.choice(self.rows*self.cols, 3, replace=False)

        coin, a_1, a_2 = positions
        if a_1==a_2:
            self.grid[coin//self.cols , coin%self.cols] = self.curr_coin
            self.grid[a_1//self.cols , a_1%self.cols] = BOTH
        else:
            self.grid[coin//self.cols , coin%self.cols] = self.curr_coin
            self.grid[a_1//self.cols , a_1%self.cols] = AGENT_1
            self.grid[a_2//self.cols , a_2%self.cols] = AGENT_2

        return positions

    def step(self, actions):
        terminated = False
        a1, a2 = actions
        a1 = torch.argmax(a1).detach().cpu().numpy().item()
        a2 = torch.argmax(a2).detach().cpu().numpy().item()
        a_1_x, a_1_y = self.a_1//self.cols, self.a_1%self.cols
        a_2_x, a_2_y = self.a_2//self.cols, self.a_2%self.cols

        self.grid[a_1_x, a_1_y] = EMPTY
        self.grid[a_2_x, a_2_y] = EMPTY

        new_a_1_y, new_a_1_x = self.calculate_pos(a_1_x, a_1_y, a1)
        new_a_2_y, new_a_2_x = self.calculate_pos(a_2_x, a_2_y, a2)

        self.a_1 = new_a_1_y*self.cols + new_a_1_x
        self.a_2 = new_a_2_y*self.cols + new_a_2_x

        r1, r2 = 0, 0

        if self.a_1==self.coin or self.a_2==self.coin:
            terminated = True
            # A1 has coin 1
            if self.a_1==self.coin and self.curr_coin==COIN_1:
                r1 = 1
            # A2 has coin 2
            if self.a_2==self.coin and self.curr_coin==COIN_2:
                r2 = 1
            # A1 has coin 2
            if self.a_1==self.coin and self.curr_coin==COIN_2:
                r1 = 1
                r2 = r2 - 2
            # A2 has coin 1
            if self.a_2==self.coin and self.curr_coin==COIN_1:
                r2 =  1
                r1 = r1 - 2

        if self.a_1==self.coin and self.a_2==self.coin:
            self.grid[new_a_1_y, new_a_1_x] = BOTH_AND_COIN
        if self.a_1==self.a_2:
            self.grid[new_a_1_y, new_a_1_x] = BOTH
        else:
            self.grid[new_a_1_y, new_a_1_x] = AGENT_1
            self.grid[new_a_2_y, new_a_2_x] = AGENT_2

        if terminated:
            if self.curr_coin==COIN_1:
                self.curr_coin=COIN_2
            else:
                self.curr_coin=COIN_1
            self.grid = self.reset(reset_history=False)[0]

        grid = torch.Tensor(self.grid).to(self.device)
        r1 = torch.Tensor([r1]).to(self.device)
        r2 = torch.Tensor([r2]).to(self.device)

        return grid, r1, r2, False, False, {}

    def calculate_pos(self, a_y, a_x, a):
        if a==UP:
            a_y = max(0, a_y-1)
        elif a==DOWN:
            a_y = min(self.rows-1, a_y+1)
        elif a==LEFT:
            a_x = max(0, a_x-1)
        elif a==RIGHT:
            a_x = min(self.cols-1, a_x+1)
        return (a_y, a_x)

    def reset(self, reset_history=True):
        self.coin, self.a_1, self.a_2 = self.init_pos
        self.init_grid(self.init_pos)
        if reset_history:
            self.reset_history()

        if self.render_mode == "human":
            self.render()
        
        grid = torch.Tensor(self.grid).to(self.device)

        return grid, {}

    def reset_history(self):
        self.j1 = torch.Tensor([-1]).to(self.device)
        self.j2 = torch.Tensor([-1]).to(self.device)

    def render(self, mode='human', close=False):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        for x in range(0, self.screen_width, BLOCKSIZE):
            for y in range(0, self.screen_height, BLOCKSIZE):
                rect = pygame.Rect(x, y, BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(self.surf, (0, 0, 0), rect, 1)

        for i, x in enumerate(range(0, self.screen_width, BLOCKSIZE)):
            for j, y in enumerate(range(0, self.screen_height, BLOCKSIZE)):
                if self.grid[j, i]==BOTH or self.grid[i, j]==BOTH_AND_COIN:
                    rect_a1 = pygame.Rect(x+2, y+2, BLOCKSIZE-4, BLOCKSIZE//2 - 2)
                    rect_a2 = pygame.Rect(x+2, y + BLOCKSIZE//2, BLOCKSIZE-4, BLOCKSIZE//2 - 2)
                    pygame.draw.rect(self.surf, (255, 0, 0), rect_a1, 0)
                    pygame.draw.rect(self.surf, (0, 0, 255), rect_a2, 0)
                if self.grid[j, i]==COIN_1:
                    pos = (x+BLOCKSIZE//2, y+BLOCKSIZE//2)
                    pygame.draw.circle(self.surf, (255,0,0), pos, BLOCKSIZE//2 - 4, 0)
                    pygame.draw.circle(self.surf, (139,0,0), pos, BLOCKSIZE//2 - 4, 4)
                if self.grid[j, i]==COIN_2:
                    pos = (x+BLOCKSIZE//2, y+BLOCKSIZE//2)
                    pygame.draw.circle(self.surf, (0,0,255), pos, BLOCKSIZE//2 - 4, 0)
                    pygame.draw.circle(self.surf, (0,0,139), pos, BLOCKSIZE//2 - 4, 4)
                if self.grid[j, i]==AGENT_1:
                    rect_a1 = pygame.Rect(x+2, y+2, BLOCKSIZE-4, BLOCKSIZE//2 - 2)
                    pygame.draw.rect(self.surf, (255, 0, 0), rect_a1, 0)
                if self.grid[j, i]==AGENT_2:
                    rect_a2 = pygame.Rect(x+2, y + BLOCKSIZE//2, BLOCKSIZE-4, BLOCKSIZE//2 - 2)
                    pygame.draw.rect(self.surf, (0, 0, 255), rect_a2, 0)
                

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(30)
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
