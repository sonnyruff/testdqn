import gymnasium as gym
import buffalo_gym
import numpy as np
import matplotlib.pyplot as plt  # only for this program
import pickle
from tqdm import tqdm

def run(episodes, is_training=True):
    name = 'conbandit'
    n = 2
    env = gym.make("MultiBuffalo-v0")

    # Q-Table and Hyperparameters Initialization -------------------------
    if(is_training):
        # q = np.zeros((env.observation_space.n, env.action_space.n))
        q = np.zeros((n, env.action_space.n))
    else:
        f = open('rsc/' + name + '.pkl','rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros((n, episodes))
    epsilon_history = []

    # Q-Learning Training Loop -------------------------------------------
    progress_bar = tqdm(range(episodes))
    for i in progress_bar:
        _, _ = env.reset()
        state = np.random.choice(n)

        if is_training and rng.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q[state, :])     # Exploit

        _, reward, _, _, _ = env.step(action)
        if is_training:
            q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[state, :]) - q[state, action])
            # q[action] = q[action] + learning_rate_a * (reward + discount_factor_g * np.max(q) - q[action])

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[state, i] = reward
        epsilon_history.append(epsilon)

    env.close()


    # Save Trained Q-Table -----------------------------------------------
    if is_training:
        with open('rsc/' + name + ".pkl", "wb") as f:
            pickle.dump(q, f)


    # Performance Visualization ------------------------------------------
    sum_rewards = np.zeros((n, episodes))
    for state in range(n):
        for t in range(episodes):
            sum_rewards[state, t] = np.sum(rewards_per_episode[state, max(0, t - 100):(t + 1)])

    plt.figure(figsize=(10, 5))
    plt.subplot(121) 
    plt.plot(sum_rewards[0], label='Context 0', color='blue')
    plt.plot(sum_rewards[1], label='Context 1', color='red')
    plt.title("Raw Rewards per Context")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(epsilon_history, label='Epsilon', color='green')
    plt.title("Epsilon History")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True)

    plt.savefig('rsc/' + name + '.png')
    plt.show()


# ========================================================================
# Main
# ========================================================================
if __name__ == '__main__':
    run(2000)
    # run(100, is_training=False)
