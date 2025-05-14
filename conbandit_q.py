import gymnasium as gym
import buffalo_gym
import numpy as np
import matplotlib.pyplot as plt  # only for this program
import pickle
from tqdm import tqdm

def run(episodes, is_training=True):
    name = 'conbandit_q'
    # n = 2
    env = gym.make("ContextualBandit-v0")

    # Q-Table and Hyperparameters Initialization -------------------------
    if(is_training):
        q = np.zeros((env.observation_space.shape[0], env.action_space.n))
    else:
        f = open('rsc/' + name + '.pkl','rb')
        q = pickle.load(f)
        f.close()
    
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros((env.observation_space.shape[0], episodes))
    epsilon_history = []
    
    arm_weights = []

    # Q-Learning Training Loop -------------------------------------------
    # state, _ = env.reset()
    state = env.reset()[0][0].astype(int)
    progress_bar = tqdm(range(episodes))
    for i in progress_bar:
        # state = np.random.choice(n)

        if is_training and rng.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q[state, :])     # Exploit

        next_state, reward, _, _, _ = env.step(action)
        next_state = next_state[0].astype(int)
        if is_training:
            q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[state, :]) - q[state, action])
            # q[action] = q[action] + learning_rate_a * (reward + discount_factor_g * np.max(q) - q[action])

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        arm_weights.append((state, q[state, :]))

        state = next_state

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

    context_dict = {}
    for state, q_values in arm_weights:
        ctx_key = tuple(state)
        if ctx_key not in context_dict:
            context_dict[ctx_key] = []
        context_dict[ctx_key].append(q_values)

    # print(f"Contexts seen: {len(context_dict)}")

    n_contexts = len(context_dict)
    fig, axs = plt.subplots(1, n_contexts, figsize=(3 * n_contexts, 5), squeeze=False)
    fig.suptitle("Q-value Heatmaps by Context", fontsize=16)

    for idx, (ctx, q_values_list) in enumerate(context_dict.items()):
        ax = axs[0, idx]
        q_matrix = np.array(q_values_list)
        im = ax.imshow(q_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(f'Context: {int(ctx[0])}')
        ax.set_xlabel('Action Index')
        ax.set_ylabel('Training Step')
        fig.colorbar(im, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ========================================================================
# Main
# ========================================================================
if __name__ == '__main__':
    run(2000)
    # run(100, is_training=False)
