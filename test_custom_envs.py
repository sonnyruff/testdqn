import matplotlib.pyplot as plt
import numpy as np 

import gymnasium as gym

import custom_envs



def test_bandit_v0():
    env = gym.make("Bandit-v0", arms=5, optimal_arms=1)
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

def test_conbandit_v0():
    env = gym.make("ContextualBandit-v0", arms=5, optimal_arms=1, states=3, pace=1, suboptimal_std=0.2)
    state, _ = env.reset()

    print(env.unwrapped.offsets)
    print(env.unwrapped.stds)
    print(np.argmax(env.unwrapped.offsets, axis=2)[0]) # !!!
    

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

def test_conbandit_v1_0():
    arms = 5
    env = gym.make("ContextualBandit-v1", arms=arms, optimal_arms=1, states=3, pace=1, suboptimal_std=0.2)
    state, _ = env.reset()

    states = []
    rewards_by_state_action = {}
 
    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        next_state, reward, _, _, _ = env.step(action)
        # state_index = int(state[0])
        state = next_state
        states.append(state.item())

        # if state_index not in rewards_by_state_action:
        #     rewards_by_state_action[state_index] = {}
        # if action not in rewards_by_state_action[state_index]:
        #     rewards_by_state_action[state_index][action] = []
        
        # rewards_by_state_action[state_index][action].append(reward)

    # num_states = len(rewards_by_state_action)
    # plt.figure(figsize=(14, 3 * num_states))

    # for i, (state_index, actions) in enumerate(rewards_by_state_action.items()):
    #     # print(f"State {state_index}: {env.optimal_arms_list}")
    #     plt.subplot(num_states, 1, i + 1)
    #     for action, rewards in actions.items():
    #         plt.hist(rewards, bins=50, alpha=0.6, label=f"Action {action}")
    #     plt.title(f"Reward Distribution in State {state_index}")
    #     plt.xlabel("Reward")
    #     plt.ylabel("Frequency")
    #     plt.legend()
    #     plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    plt.hist(states, bins=10)
    plt.title("State Frequency Histogram")
    plt.xlabel("State Index")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    x = np.linspace(-3, 3, 100)
    y = np.zeros((arms, len(x)))

    for i in range(arms):
        for j in range(len(x)):
            y[i, j] = env.unwrapped.reward(x[j], i)

    plt.figure(figsize=(10, 6))
    for i in range(arms):
        plt.scatter(x, y[i], label=f'Arm {i}', s=10, alpha=0.7)

    plt.title("Reward Scatter Plot by Arm")
    plt.xlabel("State")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

def test_conbandit_v1_1():
    arms = 5
    env = gym.make("ContextualBandit-v1", arms=arms, optimal_arms=1, states=3, pace=1, suboptimal_std=0.2)
    state, _ = env.reset()

    data = []
 
    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        next_state, reward, _, _, _ = env.step(action)
        state_index = float(state[0])
        data.append([state_index, action, float(reward)])
        state = next_state

    data = np.array(data)

    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(data[:, 0], data[:, 2], c=data[:, 1], cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label="Action")
    plt.xlabel("State")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

####################################################################################################

if __name__ == "__main__":
    # test_bandit_v0()
    # test_conbandit_v0()
    # test_conbandit_v1_0()
    test_conbandit_v1_1()
