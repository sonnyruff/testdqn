"""
NoisyNet-DQN implementation for ContextualBandit-v1 - Continuous input state
Single loop version
Single network

Author: Sonny Ruff
Date: 12-05-2025

Based on:
- NoisyNet-DQN implementation from https://nbviewer.org/github/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb
- Parts of https://github.com/knyazer/nanodqn/tree/main
- OpenAI Gym (https://github.com/openai/gym) & Buffalo Gym environment (https://github.com/foreverska/buffalo-gym)

e.g.
 - Environment visualisation is only available for 1 dimension
    py conbandit_combi1_0.py --network-type NOISY
 - More dimensions only show a reward and loss plot
    py conbandit_combi1_0.py --network-type NOISY --dims 20
 - The network with regular linear layers provide an epsilon history plot
    py conbandit_combi1_0.py --network-type REGULAR
 - Non-static environements
    py conbandit_combi1_0.py --network-type REGULAR --dims 1 --dynamic-rate 100 

    py conbandit_combi1_0.py --seed 8796 --no-logging --network-type NOISY --dims 20    
"""
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
import tyro

import custom_envs
from NoisyLinear import NoisyLinear

####################################################################################################

class NetworkType(str, Enum):
    REGULAR = "Regular"
    NOISY = "NoisyNetwork"

@dataclass
class Args:
    network_type: NetworkType = NetworkType.REGULAR
    """the type of network to use; either 'Regular' or'NoisyNetwork'"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = np.random.randint(0, 10000)
    """seed of the experiment"""
    wandb_project_name: str = "noisynet-dqn"
    """the wandb's project name"""
    logging: bool = True
    """whether to log to wandb"""

    env_id: str = "ContextualBandit-v2"
    """the id of the environment"""
    num_episodes: int = 3000
    """the number of episodes to run"""
    memory_size: int = 1000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    batch_size: int = 50
    """the batch size of sample from the reply memory"""

    dims: int = 1
    arms: int = 10
    states: int = 2
    optimal_arms: int | list[int] = 1
    dynamic_rate: int | None = None
    pace: int = 5
    optimal_mean: float = 10
    optimal_std: float = 1
    min_suboptimal_mean: float = 0
    max_suboptimal_mean: float = 5
    suboptimal_std: float = 1

####################################################################################################

class NoisyNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(NoisyNetwork, self).__init__()

        self.feature = nn.Linear(in_dim, in_dim)
        self.noisy_layer1 = NoisyLinear(in_dim, out_dim)
        self.noisy_layer2 = NoisyLinear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = F.relu(self.feature(x))
        hidden = F.relu(self.noisy_layer1(feature))
        out = self.noisy_layer2(hidden)
        
        return out
    
    def resample_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer1.resample_noise()
        self.noisy_layer2.resample_noise()

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.feature = nn.Linear(in_dim, in_dim)
        self.layer1 = nn.Linear(in_dim, out_dim)
        self.layer2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = F.relu(self.feature(x))
        hidden = F.relu(self.layer1(feature))
        out = self.layer2(hidden)
        
        return out

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.int64)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: int, 
        rew: float, 
    ):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs])

    def __len__(self) -> int:
        return self.size

class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        seed: int,
        gamma: float = 0.99,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            gamma (float): discount factor
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.seed = seed
        self.gamma = gamma
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # network: dqn
        if args.network_type == NetworkType.NOISY:
            self.dqn = NoisyNetwork(obs_dim, action_dim).to(self.device)
        else:
            self.dqn = Network(obs_dim, action_dim).to(self.device)
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray, epsilon: float) -> np.ndarray:
        """Select an action from the input state."""
        if np.random.rand() < epsilon and args.network_type == NetworkType.REGULAR:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, state: np.ndarray, action: np.ndarray) -> float:
        """Take an action and return the response of the env."""
        next_state, reward, _, _, _ = self.env.step(action)
        
        if not self.is_test:
            self.memory.store(state, action, reward)
    
        return next_state, reward
        
    def train(self, num_episodes: int):
        """Train the agent."""
        self.is_test = False
        epsilon = 1
        
        update_cnt = 0
        losses = []
        rewards = []
        scores = []
        arm_weights = []
        data = []
        epsilons = []
        
        state, _ = self.env.reset(seed=self.seed)

        # Double loop isn't necessary
        for step_id in tqdm(range(1, num_episodes + 1)):
            score = 0
            
            if args.network_type == NetworkType.NOISY:
                self.dqn.resample_noise() # line 5

            action = self.select_action(state, epsilon) # line 6

            next_state, reward = self.step(state, action) # line 7

            data.append([step_id, float(state[0]), action, float(reward)]) # line 8

            rewards.append(reward)
            state = next_state
            if args.network_type == NetworkType.REGULAR:
                epsilons.append(epsilon)

            if step_id % 10 == 0 and args.dims == 1:
                ## Scatterplot background ======
                x = np.linspace(-3, 3, 100)
                # put each x value forward through the network
                q_values = self.dqn(torch.FloatTensor(x).unsqueeze(1).to(self.device)).detach().cpu().numpy()
                best_actions = np.argmax(q_values, axis=1)
                arm_weights.append((step_id, best_actions))
                ## =============================

            if step_id % 50 == 0:
                # score += sum(rewards[-50:])
                score += np.mean(rewards[-50:])
                scores.append(score)
                if args.logging: wandb.log({"score": score})

            # if training is ready
            if len(self.memory) >= self.batch_size:
                samples = self.memory.sample_batch() # line 12
                
                if args.network_type == NetworkType.NOISY:
                    self.dqn.resample_noise() # line 13
                    noise_l1 = self.dqn.noisy_layer1.get_noise()
                    noise_l2 = self.dqn.noisy_layer2.get_noise()
                    if args.logging: wandb.log({
                        "noisy_layer1/weight_epsilon_std": np.std(noise_l1["weight_epsilon"]),
                        "noisy_layer1/bias_epsilon_std": np.std(noise_l1["bias_epsilon"]),
                        "noisy_layer2/weight_epsilon_std": np.std(noise_l2["weight_epsilon"]),
                        "noisy_layer2/bias_epsilon_std": np.std(noise_l2["bias_epsilon"])
                    })
                else:
                    epsilon = max(epsilon - 2/num_episodes, 0)
                        
                loss = self._compute_dqn_loss(samples)
                losses.append(loss)
                if args.logging: wandb.log({"loss": loss})
                
                update_cnt += 1
                
        print(f"Mean rewards: {np.mean(rewards)}")
        self.env.close()
        self._plot(scores, losses, arm_weights, np.array(data), epsilons)
        
    def test(self, episode_length) -> None:
        """Test the agent."""
        self.is_test = True
        
        # for recording a video
        naive_env = self.env # remove?
        
        state, _ = self.env.reset()
        score = 0
        
        for _ in range(episode_length):
            action = self.select_action(state, 0)
            next_state, reward = self.step(state, action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        self.env = naive_env # remove?

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)

        # line 18, 19, 24; every state is terminal
        curr_q_value = self.dqn(state).gather(1, action)
        loss = F.mse_loss(curr_q_value, reward) # line 25
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _plot(
        self,
        scores: List[float], 
        losses: List[float],
        arm_weights: List[np.ndarray],
        data: np.ndarray,
        epsilons: List[float]
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.title('score: %s' % (np.mean(scores[-10:])))
        plt.plot(scores)
        plt.xlabel('Episode')
        plt.ylabel('Score')

        plt.subplot(122)
        plt.title('loss')
        plt.plot(losses)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        
        if args.network_type == NetworkType.REGULAR:
            plt.figure(figsize=(10, 10))
            plt.plot(epsilons)
        # ------------------------------------------------------

        if args.dims == 1:
            fig, ax = plt.subplots(2, 1, figsize=(15, 10))

            # --- Top subplot: heatmap + scatter overlay ---
            step_ids = [step for step, _ in arm_weights]
            x_vals = np.linspace(-3, 3, 100)
            action_matrix = np.stack([actions for _, actions in arm_weights], axis=0)

            # Heatmap
            im = ax[0].imshow(
                action_matrix,
                aspect='auto',
                extent=[x_vals[0], x_vals[-1], step_ids[0], step_ids[-1]],
                origin='lower',
                cmap='viridis'
            )

            # Overlay scatter1
            scatter1 = ax[0].scatter(
                data[:, 1], data[:, 0],
                c=data[:, 2],
                cmap="viridis",
                alpha=0.6,
                s=15,
                edgecolors='black',
                linewidths=0.2
            )
            fig.colorbar(scatter1, ax=ax[0], label="Action")
            ax[0].set_xlabel("State")
            ax[0].set_ylabel("Training Step")
            ax[0].set_title("Best Action Heatmap and Scatter Overlay")
            ax[0].grid(True)


            # --- Bottom subplot: second scatter ---
            sample_data = sample_env(
                gym.make(
                    args.env_id,
                    arms=args.arms,
                    states=args.states,
                    optimal_arms=args.optimal_arms,
                    dynamic_rate=args.dynamic_rate,
                    pace=args.pace,
                    seed=args.seed,
                    optimal_mean=args.optimal_mean,
                    optimal_std=args.optimal_std,
                    min_suboptimal_mean=args.min_suboptimal_mean,
                    max_suboptimal_mean=args.max_suboptimal_mean,
                    suboptimal_std=args.suboptimal_std), 
                1000)

            group_ids = np.unique(sample_data[:, 1])

            for gid in group_ids:
                group_mask = sample_data[:, 1] == gid
                group_data = sample_data[group_mask]
                sorted_indices = np.argsort(group_data[:, 0])
                ax[1].plot(group_data[sorted_indices, 0], group_data[sorted_indices, 2], alpha=0.4, linewidth=1.5,)

            timesteps = data[:, 0]
            # normalized = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min() + 1e-8)
            # size = 80 * normalized
            size = 80 * timesteps / (timesteps.max() - timesteps.min())

            scatter2 = ax[1].scatter(data[:, 1], data[:, 3], c=data[:, 2], cmap="viridis", alpha=0.6, s=size)
            fig.colorbar(scatter2, ax=ax[1], label="Action")
            ax[1].set_xlabel("State")
            ax[1].set_ylabel("Reward")
            ax[1].grid(True)

            plt.tight_layout()

            if args.logging:
                wandb.log({"Reward Scatter": wandb.Image(fig)})
    
        plt.show()

def sample_env(env, num_samples=1000):
    """somehow just sampling the reward functions didn't work"""
    state, _ = env.reset(seed=args.seed)
    _data = []
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, _, _, _ = env.step(action)
        state_index = float(state[0])
        _data.append([state_index, action, float(reward)])
        state = next_state
    return np.array(_data)


####################################################################################################

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    if args.logging:wandb.init(
        project=args.wandb_project_name,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env = gym.make(
        args.env_id,
        dims=args.dims,
        arms=args.arms,
        states=args.states,
        optimal_arms=args.optimal_arms,
        dynamic_rate=args.dynamic_rate,
        pace=args.pace,
        seed=args.seed,
        optimal_mean=args.optimal_mean,
        optimal_std=args.optimal_std,
        min_suboptimal_mean=args.min_suboptimal_mean,
        max_suboptimal_mean=args.max_suboptimal_mean,
        suboptimal_std=args.suboptimal_std)

    agent = DQNAgent(
        env,
        args.memory_size,
        args.batch_size,
        args.seed,
        args.gamma
    )

    print(f"[ Environment: '{args.env_id}' | Type: {args.network_type} | Seed: {args.seed} | Device: {agent.device} ]")
    print(agent.dqn)

    agent.train(args.num_episodes)
    agent.test(100)

    if args.logging: wandb.finish()
