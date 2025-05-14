import gymnasium as gym
from custom_envs.bandit_v0 import CustomBanditEnv
from custom_envs.conbandit_v0 import CustomContextualBanditEnv

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