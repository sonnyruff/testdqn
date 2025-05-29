from typing import Any, TypeVar, SupportsFloat
import random
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ConbanditEnv2(gym.Env):
    """
    Multi-armed bandit environment with a continuous state
    """
    metadata = {'render_modes': []}

    # def sigmoid(self, z, a):
    #     return 1 / (1 + np.exp(-z + a))

    def reward(self, state: np.ndarray, action: int) -> float:
        # LINEAR
        # return np.sum(self.slopes[action] * state + self.intercepts[action])

        # SIGMOID
        return np.sum(1 / (1 + np.exp(-self.intercepts[action] - self.slopes[action] * state)))

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

        self.intercepts = self.rng.uniform(-2, 2, (self.arms, self.dims))
        self.slopes = self.rng.uniform(-1, 1, (self.arms, self.dims))

    def __draw_state(self): 
        self.state = self.rng.normal(0, 1, self.dims)
        # self.state = self.rng.uniform(-3.0, 3.0, 1)

    def __init__(self, dims: int = 1, arms: int = 10, states: int = 2, optimal_arms: int | list[int] = 1,
                 dynamic_rate: int | None = None, pace: int = 1, seed: int | None = None, optimal_mean: float = 10,
                 optimal_std: float = 1, min_suboptimal_mean: float = 0, max_suboptimal_mean: float = 5,
                 suboptimal_std: float = 1, noisy: bool = False):
        """
        Multi-armed bandit environment with k arms and n states
        :param arms: number of arms
        :param states: number of states
        :param optimal_arms: number of optimal arms or list of optimal orms in each state
        :param dynamic_rate: number of steps between drawing new arm means, None means no dynamic rate
        :param seed: random seed
        :param optimal_mean: mean of optimal arms
        :param optimal_std: std of optimal arms
        :param min_suboptimal_mean: min mean of suboptimal arms
        :param max_suboptimal_mean: max mean of suboptimal arms
        :param suboptimal_std: std of suboptimal arms
        """
        self.dims = dims
        self.arms = arms
        # self.states = states
        self.dynamic_rate = dynamic_rate
        self.pace = pace
        self.initial_seed = seed
        self.seed = seed
        self.noisy = noisy

        # TODO reimplement
        # self.optimal_mean = optimal_mean
        # self.min_suboptimal_mean = min_suboptimal_mean
        # self.max_suboptimal_mean = max_suboptimal_mean

        # TODO reimplement
        # self.optimal_std = optimal_std
        # self.suboptimal_std = suboptimal_std

        # if optimal_arms is list and len(optimal_arms) != self.arms:
        #     raise ValueError("Optimal arms list must have equal number of arms")
        # self.optimal_arms = optimal_arms

        self.rng = np.random.default_rng(self.seed)

        self.action_space = gym.spaces.Discrete(self.arms)
        # todo shouldn't low and high be -4 to 4?
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.dims,), dtype=np.float32)

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

        self.ssr += 1
        if self.pace is None or self.ssr % self.pace == 0:
            self.__draw_state()

        self.pulls += 1
        if self.dynamic_rate is not None and self.pulls % self.dynamic_rate == 0:
            print(f"Changing arms")
            if self.seed is not None:
                self.seed += 1
            self.__draw_arms()

        return np.array(self.state, dtype=np.float32), reward, False, False, {'regret': regret}