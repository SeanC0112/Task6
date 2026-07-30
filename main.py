import numpy as np
import pandas as pd
import gymnasium as gym
import torch
from torch import nn

replay_memory = pd.DataFrame(columns=["state", "action", "reward", "next_state", "done"])
state = np.ndarray(shape = (4, 96, 96))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.set_default_device(device=device)
print(f"Using {device} device")

# class Q_Value_Function(nn.Module): #use more low-level solution instead of this
#     def __init__(self):
#         super.__init__()
#         self.flatten = nn.Flatten()

def preprocess(obs, prev_state):
    #use ITU-R 601-2 luma formula to convert to grayscale
    obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])

    #keep runnning list of past 4 frames to check for movement
    obs = obs[np.newaxis, ...]

    state = np.append(prev_state, obs, axis=0)
    if np.size(state, axis=0) > 4:
        state = state[1:, ...]

    return state


        

env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True, max_episode_steps=1000,)

for i in range(5):
    obs, _ = env.reset()
    state = preprocess(obs=obs, prev_state=state)
    print(state.shape)
    print(state[0, 0, 0])

