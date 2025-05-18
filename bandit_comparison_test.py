import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
import tyro

import custom_envs
from NoisyLinear import NoisyLinear
from conbandit_dqn2_0 import Network as DQNNetwork
from conbandit_dqn2_0 import DQNAgent as DQNAgent
from conbandit_dqn2_0 import Network as DQNNetwork
from conbandit_noisydqn2_1 import DQNAgent as NoisyDQNAgent

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = np.random.randint(0, 10000)
    """seed of the experiment"""
    wandb_project_name: str = "noisynet-dqn"
    """the wandb's project name"""
    logging: bool = True
    """whether to log to wandb"""

    env_id: str = "ContextualBandit-v1"
    """the id of the environment"""
    num_episodes: int = 3000
    """the number of episodes to run"""
    memory_size: int = 1000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_update: int = 50
    """the timesteps it takes to update the target network"""
    batch_size: int = 50
    """the batch size of sample from the reply memory"""

    arms: int = 10
    states: int = 2
    optimal_arms: int | list[int] = 1
    dynamic_rate: int | None = None
    pace: int = 5
    optimal_mean: float = 10
    optimal_std: float = 1
    min_suboptimal_mean: float = 0
    max_suboptimal_mean: float = 5
    suboptimal_std: float = 1


if __name__ == "__main__":
