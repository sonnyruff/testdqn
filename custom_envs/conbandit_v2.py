from typing import Any, TypeVar, SupportsFloat
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

import random
import matplotlib.pyplot as plt

import numpy as np
import gymnasium as gym



class ConbanditEnv2(gym.Env):
    """
    Multi-armed bandit environment with a continuous state
    """
    metadata = {'render_modes': []}

    def reward(self, state: float, action: int) -> float:
        # LINEAR
        # return self.slopes[action] * state + self.intercepts[action]

        # SIGMOID
        return (1 / (1 + np.exp(-self.intercepts[action] - self.slopes[action] * state))).item()

        # NORMAL PDF
        # mean = self.intercepts[action]
        # std_dev = self.slopes[action]
        # return np.sum((1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((state - mean) / std_dev) ** 2))

    def optimal_reward(self, state: np.ndarray) -> int:
        arr = [self.reward(state, action) for action in range(self.arms)]
        return np.argmax(arr), np.max(arr)

    def __draw_arms(self):
        """
        Draw new arms
        """
        # self.offsets = np.random.uniform(self.min_suboptimal_mean, self.max_suboptimal_mean,
        #                                  size=(1, self.states, self.arms))
        # self.stds = []
        # for state in range(self.states):
        #     optimal_arms = self.rng.choice(range(self.arms), self.optimal_arms, replace=False)
        #     for arm in optimal_arms:
        #         self.offsets[0, state, arm] = self.optimal_mean
        #     self.stds.append([self.optimal_std if arm in optimal_arms else self.suboptimal_std
        #                       for arm in range(self.arms)])

        self.intercepts = self.rng.uniform(-2, 2, self.arms)
        self.slopes = self.rng.uniform(-1, 1, self.arms)

    def __draw_state(self): 
        self.state = self.rng.normal(0, 1, 1)
        # self.state = self.rng.uniform(-3.0, 3.0, 1)

    def __init__(self,
                 arms: int = 10,
                 dynamic_rate: int | None = None,
                 seed: int | None = None,
                 noisy: bool = False):
        """
        Multi-armed bandit environment with k arms and n states
        :param arms: number of arms
        :param dynamic_rate: number of steps between drawing new arm means, None means no dynamic rate
        :param seed: random seed
        """
        self.arms = arms
        self.dynamic_rate = dynamic_rate
        self.initial_seed = seed
        self.seed = seed
        self.noisy = noisy

        self.rng = np.random.default_rng(self.seed)

        self._total_regret = 0.
        self.action_space = gym.spaces.Discrete(self.arms)
        # todo shouldn't low and high be -4 to 4?
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.pulls = 0
        self.ssr = 0
        self.__draw_state()
        self.__draw_arms()

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

        self.seed = seed
        self.pulls = 0
        self.ssr = 0
        self.__draw_state()
        self.__draw_arms()

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: arm to pull
        :return: observation, reward, done, term, info
        """
        reward = self.reward(self.state, action)
        if self.noisy:
            reward = self.rng.normal(reward, 0.1, 1)[0]

        # Calculate regret before redrawing state and arms!!
        optimal_reward_action, optimal_reward = self.optimal_reward(self.state)
        regret = optimal_reward - reward
        self._total_regret += regret

        self.__draw_state()

        self.pulls += 1
        if self.dynamic_rate is not None and self.pulls % self.dynamic_rate == 0:
            print(f"Changing arms")
            if self.seed is not None:
                self.seed += 1
            self.__draw_arms()

        return np.array(self.state, dtype=np.float32), reward, False, False, {'regret': regret, 'total_regret': self._total_regret}