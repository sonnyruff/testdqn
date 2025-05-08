# https://www.youtube.com/watch?v=ZhoIgo3qqLU
# https://www.youtube.com/watch?v=EUrWGTCGzlA
# https://www.youtube.com/watch?v=qKePPepISiA

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt  # only for this program
import pickle
from tqdm import tqdm

def run(episodes, is_training=True, render=False):
    name = 'frozen_lake_q'
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)

    # Q-Table and Hyperparameters Initialization -------------------------
    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
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
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q[state, :])     # Exploit

            new_state, reward, terminated, truncated, _ = env.step(action)
            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    # Performance Visualization ------------------------------------------
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.savefig('rsc/' + name + '.png')

    # Save Trained Q-Table -----------------------------------------------
    if is_training:
        with open("rsc/" + name + ".pkl", "wb") as f:
            pickle.dump(q, f)

# ========================================================================
# Main Entry Point
# ========================================================================
if __name__ == '__main__':
    run(15000)
    # run(1, is_training=False, render=True)
