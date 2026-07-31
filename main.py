import numpy as np
import pandas as pd
import gymnasium as gym
import torch
from torch import nn
from collections import namedtuple, deque
import math, random
import matplotlib.pyplot as plt

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

state = np.zeros((96,96,4))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.set_default_device(device=device)
print(f"Using {device} device")

class Q_Value_Function(nn.Module): 
    def __init__(self, number_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=12, stride=4), # different from the paper convolutional size because 8x8 stride 4 would give a 23x23 output, which would be awkward
            nn.ReLU(),
            nn.LazyConv2d(32, kernel_size=4, stride=2), # same size as paper because this [produces a nice 10x10x32 output
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(number_actions)
        )

        def forward(self, x):
            return self.model(x)



def preprocess(obs, prev_state):
    #use ITU-R 601-2 luma formula to convert to grayscale
    obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])

    #keep runnning list of past 4 frames to check for movement
    obs = obs[..., np.newaxis]

    state = np.append(prev_state, obs, axis=2)
    if np.size(state, axis=2) > 4:
        state = state[..., 1:]

    return state


        

env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False, max_episode_steps=1000)

policy_model = Q_Value_Function(number_actions=env.action_space.n)
target_model = Q_Value_Function(number_actions=env.action_space.n)
target_model.load_state_dict(policy_model.state_dict())

optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=LR, alpha=0.95,)


for i in range(5):
    obs, _ = env.reset()
    state = preprocess(obs=obs, prev_state=state)
    print(state.shape)
    print(state[0, 0, 0])

