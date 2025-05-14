import matplotlib.pyplot as plt

import gymnasium as gym

from custom_envs.bandit_v0 import CustomBanditEnv
from custom_envs.conbandit_v0 import CustomContextualBanditEnv

def testCustomBanditEnv():
    env = gym.make("CustomBandit-v0", arms=5, optimal_arms=1)
    _, _ = env.reset()

    rewards_by_action = {}

    for _ in range(10000):
        action = env.action_space.sample()  # Random action
        _, reward, _, _, _ = env.step(action)

        if action not in rewards_by_action:
            rewards_by_action[action] = []

        rewards_by_action[action].append(reward)

    plt.figure(figsize=(14, 4))

    for action, rewards in rewards_by_action.items():
        plt.hist(rewards, bins=50, alpha=0.6, label=f"Action {action}")

    plt.title("Reward Distribution by Action")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def testCustomContextualBanditEnv():
    env = gym.make("CustomContextualBandit-v0", arms=5, optimal_arms=1, states=3, pace=1)
    state, _ = env.reset()

    rewards_by_state_action = {}
 
    for _ in range(10000):
        action = env.action_space.sample()  # Random action
        next_state, reward, _, _, _ = env.step(action)
        state_index = int(state[0])
        state = next_state

        if state_index not in rewards_by_state_action:
            rewards_by_state_action[state_index] = {}
        if action not in rewards_by_state_action[state_index]:
            rewards_by_state_action[state_index][action] = []
        
        rewards_by_state_action[state_index][action].append(reward)

    num_states = len(rewards_by_state_action)
    plt.figure(figsize=(14, 3 * num_states))

    for i, (state_index, actions) in enumerate(rewards_by_state_action.items()):
        # print(f"State {state_index}: {env.optimal_arms_list}")
        plt.subplot(num_states, 1, i + 1)
        for action, rewards in actions.items():
            plt.hist(rewards, bins=50, alpha=0.6, label=f"Action {action}")
        plt.title(f"Reward Distribution in State {state_index}")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

####################################################################################################

if __name__ == "__main__":
    testCustomBanditEnv()
    testCustomContextualBanditEnv()