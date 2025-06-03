from typing import Any, TypeVar, SupportsFloat
import random
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class NNBanditEnv(gym.Env):
    """
    Multi-armed bandit environment with a continuous state
    """
    metadata = {'render_modes': []}

    def __init__(self,
                 arms: int = 10,
                 dynamic_rate: int | None = None,
                 seed: int | None = None,
                 noisy: bool = False):

        self.arms = arms
        self.dynamic_rate = dynamic_rate
        self.initial_seed = seed
        self.seed = seed
        self.noisy = noisy
        torch.manual_seed(self.seed) # Should be done here instead of in the dqn file
        self.rng = np.random.default_rng(self.seed)

        self._total_regret = 0.

        self.action_space = gym.spaces.Discrete(self.arms)
        self.observation_space = gym.spaces.Box(low=-4, high=4, shape=(1,), dtype=np.float32)
        self.state = None

        self.pulls = 0
        self.ssr = 0

        self.net = Network(1, 10, self.arms)

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment
        :param seed: WARN unused, defaults to None
        :param options: WARN unused, defaults to None
        :return: observation, info
        """
        self.state = self.rng.normal(0, 1, 1)
        self.seed = seed
        self.pulls = 0
        self.ssr = 0

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: arm to pull
        :return: observation, reward, done, term, info
        """
        q_values = self.net(torch.FloatTensor(self.state)).detach().numpy()
        reward = q_values[action].item()
        if self.noisy:
            reward = self.rng.normal(reward, 0.1, 1)[0]
        
        regret = np.max(q_values) - reward
        self._total_regret += regret

        self.state = self.rng.normal(0, 1, 1)
        
        # self.pulls += 1
        # if self.dynamic_rate is not None and self.pulls % self.dynamic_rate == 0:
        #     print(f"Changing arms")
        #     if self.seed is not None:
        #         self.seed += 1
        #     self.__draw_arms()

        return self.state, reward, False, False, {'regret': regret, 'total_regret': self._total_regret}
    

class Network(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

if __name__ == '__main__':
    # env = NNBanditEnv(arms=10, seed=666)
    torch.manual_seed(42)
    net = Network(1, 10, 10)
    print(net(torch.FloatTensor([0.])))
    # print(net(torch.tensor(np.random.uniform(0, 1, 1), dtype=torch.float32)))
    # print(net(torch.FloatTensor(np.random.uniform(0, 1, 1))))
    # print(np.argmax(net(torch.FloatTensor(np.random.uniform(0, 1, 1))).detach()))
    # print(np.argmax(net(torch.FloatTensor(np.random.uniform(0, 1, 1))).detach().numpy()))

    # env = NNBanditEnv(1, 10, 10)
    # obs, _ = env.reset()
    # print(env.step(1))