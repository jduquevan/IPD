import random
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque
from torch.distributions import Normal

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

class DRLActor(nn.Module):
    def __init__(self, in_size, out_size, device, hidden_size=40, num_layers=1):
        super(DRLActor, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(in_size, hidden_size, num_layers, batch_first=True)
        self.hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x, pi_b=None, h_0=None):
        if pi_b == None:
            pi_b = -1 * torch.ones(self.out_size).to(self.device)
        
        x = torch.cat([x, pi_b])
        if h_0 is not None:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]), h_0)
        else:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]))
        x = F.relu(self.hidden(x))
        return output, F.softmax(self.linear(x).flatten(), dim=0)

    def batch_forward(self, x, pi_b=None, h_0=None):
        if pi_b == None:
            pi_b = -1 * torch.ones(self.out_size).to(self.device)
        
        x = torch.cat([x, pi_b])
        if h_0 is not None:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]), h_0)
        else:
            output, x = self.gru(x.reshape(1, 1, x.shape[0]))
        x = F.relu(self.hidden(x))
        return output, F.softmax(self.linear(x), dim=2)

class HistoryAggregator(nn.Module):
    def __init__(self, in_size, out_size, device, hidden_size=40, num_layers=1):
        super(HistoryAggregator, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(in_size, out_size, num_layers, batch_first = True)
        self.out_layer = nn.Linear(out_size, out_size)

    def forward(self, x):
        x, hidden = self.lstm(x)
        return F.relu(self.out_layer(F.relu(x[:,-1,:])))