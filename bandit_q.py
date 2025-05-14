import gymnasium as gym
import buffalo_gym
import numpy as np
import matplotlib.pyplot as plt  # only for this program
import pickle
from tqdm import tqdm

def run(episodes, is_training=True, render=False):
    name = "bandit"
    env = gym.make("Buffalo-v0")

    # Q-Table and Hyperparameters Initialization -------------------------
    if(is_training):
        # q = np.zeros((env.observation_space.n, env.action_space.n))
        q = np.zeros(env.action_space.n)
    else:
        f = open('rsc/' + name + '.pkl','rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    # Q-Learning Training Loop -------------------------------------------
    progress_bar = tqdm(range(episodes))
    for i in progress_bar:
        _, _ = env.reset()
        if is_training and rng.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q)     # Exploit

        _, reward, _, _, _ = env.step(action)
        if is_training:
            # q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])
            q[action] = q[action] + learning_rate_a * (reward + discount_factor_g * np.max(q) - q[action])

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = reward

    env.close()

    # Performance Visualization ------------------------------------------
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.savefig('rsc/' + name + '.png')
    plt.show()

    # Save Trained Q-Table -----------------------------------------------
    if is_training:
        with open('rsc/' + name + ".pkl", "wb") as f:
            pickle.dump(q, f)

# ========================================================================
# Main
# ========================================================================
if __name__ == '__main__':
    # run(2000)
    run(100, is_training=False)
