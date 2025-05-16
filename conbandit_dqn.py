import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from NoisyLinear import NoisyLinear


name = 'conbandit_dqn'

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class ConbanditNoisyDQN():
    # Hyperparameters
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    def train(self, episodes):
        env = gym.make("ContextualBandit-v0")
        num_states = 2
        num_actions = env.action_space.n
        
        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        # 1. Create policy network
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        # 2. Create target network
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # 2. Copy policy network to target network
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = np.zeros((num_states, episodes))
        epsilon_history = []

        step_count=0
        
        progress_bar = tqdm(range(episodes))
        # 9. repeat from 3
        for i in progress_bar:
            _, _ = env.reset()
            state = np.random.choice(num_states)

            ################ REMOVE THIS ################
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

            # 3. Execute action
            _,reward,_,_,_ = env.step(action)

            # 3a. memorize
            memory.append((state, action, reward))

            step_count+=1
            print(reward)
            # Keep track of the rewards collected per episode.
            rewards_per_episode[state, i] = reward

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # 10. after N episodes, sync policy network with target network by copying the weights and biases from the policy network to the target network
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "rsc/" + name + ".pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('rsc/' + name + '.png')


    # Optimize policy network -----------------------------------------------------------
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, reward in mini_batch:

            # 6. Calculate target q value
            with torch.no_grad():
                target = torch.FloatTensor(
                    reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(state, num_states)).max()
                )

            # 4. Set input nodes of policy network corresponding to the location of the player to 1
            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # 5. Set input nodes of target network corresponding to the location of the player to 1
            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 

            # 7. Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
        
        # 8. Use target values to train policy network
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        """
        Converts an state (int) to a tensor representation.
        For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

        Parameters: state=1, num_states=16
        Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        """
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy -----------------------------
    def test(self, episodes):
        env = gym.make("ContextualBandit-v0")
        num_states = 2
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("rsc/" + name + ".pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            _, _ = env.reset()
            state = np.random.choice(num_states)

            # Select best action   
            with torch.no_grad():
                action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

            # Execute action
            state,reward,_,_,_ = env.step(action)

        env.close()

    # Print DQN: state, best action, q values -------------------------------------------
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            # best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            # print(f'{s:02},{best_action},[{q_values}]', end=' ')
            print(f'{s:02},{s},[{q_values}]', end=' ')
            print()



# ========================================================================
# Main
# ========================================================================
if __name__ == '__main__':
    cbndqn = ConbanditNoisyDQN()
    cbndqn.train(500)
    cbndqn.test(1)
