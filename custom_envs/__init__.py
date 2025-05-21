import gymnasium as gym
from custom_envs.bandit_v0 import BanditEnv0
from custom_envs.conbandit_v0 import ConbanditEnv0
from custom_envs.conbandit_v1 import ConbanditEnv1
from custom_envs.conbandit_v2 import ConbanditEnv2

gym.register(
    id='Bandit-v0',
    entry_point=__name__ + ":BanditEnv0",
    max_episode_steps=1000
)

gym.register(
    id='ContextualBandit-v0',
    entry_point=__name__ + ":ConbanditEnv0",
    max_episode_steps=1000
)

gym.register(
    id='ContextualBandit-v1',
    entry_point=__name__ + ":ConbanditEnv1",
    max_episode_steps=1000
)

gym.register(
    id='ContextualBandit-v2',
    entry_point=__name__ + ":ConbanditEnv2",
    max_episode_steps=1000
)