import gymnasium as gym
from custom_envs.base_bandit import CustomBanditEnv
from custom_envs.base_conbandit import CustomContextualBanditEnv

gym.register(
    id='CustomBandit-v0',
    entry_point=__name__ + ":CustomBanditEnv",
    max_episode_steps=1000
)

gym.register(
    id='CustomContextualBandit-v0',
    entry_point=__name__ + ":CustomContextualBanditEnv",
    max_episode_steps=1000
)