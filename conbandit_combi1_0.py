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

@dataclass
class Args:
    noisy_net: bool = True
    """the type of network to use'"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = np.random.randint(0, 10000)
    """seed of the experiment"""
    wandb_project_name: str = "noisynet-dqn"
    """the wandb's project name"""
    plotting: bool = False
    """whether to plot the results"""
    logging: bool = True
    """whether to log to wandb"""

    env_id: str = "ContextualBandit-v2"
    """the id of the environment"""
    num_episodes: int = 2000
    """the number of episodes to run"""
    memory_size: int = 1000
    """the replay memory buffer size"""
    batch_size: int = 50
    """the batch size of sample from the reply memory"""

    noisy_layer_distr_type: str = "normal" # or uniform
    """the distribution of the noisy layer"""
    noisy_layer_init_std: float = 0.5
    """the initial standard deviation of the noisy layer"""
    noisy_output: bool = True

    hidden_layer_size: int = 10

    arms: int = 10
    dynamic_rate: int | None = None
    noisy: bool = False

####################################################################################################

class NoisyNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, distr_type: str, init_std: float, noisy_output: bool):
        super().__init__()

        if noisy_output:
            last_layer = NoisyLinear(hidden_dim, out_dim, distr_type, init_std)
        else:
            last_layer = nn.Linear(hidden_dim, out_dim)
            
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, hidden_dim, distr_type, init_std),
            nn.ReLU(),
            last_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def resample_noise(self):
        for layer in self.net:
            if isinstance(layer, NoisyLinear):
                layer.resample_noise()

class Network(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        """Initialization."""
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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
        dqn (Network): model to train and select actions
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        env: gym.Env,
        args: Args = None
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
        """
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.epsilon = 1
        
        self.env = env
        self.args = args
        self.memory = ReplayBuffer(self.obs_dim, self.args.memory_size, self.args.batch_size)
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # network: dqn
        if args.noisy_net:
            self.dqn = NoisyNetwork(self.obs_dim,
                           self.args.hidden_layer_size,
                           self.action_dim,
                           self.args.noisy_layer_distr_type,
                           self.args.noisy_layer_init_std,
                           self.args.noisy_output
                        ).to(self.device)
        else:
            self.dqn = Network(self.obs_dim,
                           self.args.hidden_layer_size,
                           self.action_dim
                        ).to(self.device)
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        if np.random.rand() < self.epsilon and not args.noisy_net:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, state: np.ndarray, action: np.ndarray) -> float:
        """Take an action and return the response of the env."""
        next_state, reward, _, _, info = self.env.step(action)
        
        if not self.is_test:
            self.memory.store(state, action, reward)
    
        return next_state, reward, info
        
    def train(self, num_episodes: int):
        """Train the agent."""
        self.is_test = False
        
        rewards = []
        scores = []
        losses = []
        regrets = []
        arm_weights = []
        data = []
        epsilons = []
        
        state, _ = self.env.reset(seed=self.args.seed)

        # Double loop isn't necessary
        for step_id in tqdm(range(1, num_episodes + 1)):
            score = 0
            
            if args.noisy_net:
                self.dqn.resample_noise() # line 5

            action = self.select_action(state) # line 6
            next_state, reward, info = self.step(state, action) # line 7

            data.append([step_id, float(state[0]), action, float(reward)]) # line 8

            # regrets.append(self.env.unwrapped.reward(state, action) - reward) # !!! THIS DOESN'T WORK
            regrets.append(info['total_regret'])
            rewards.append(reward)

            state = next_state

            if self.args.logging:
                wandb.log({"regret": info['regret']})
            if not args.noisy_net:
                epsilons.append(self.epsilon)

            # Scatterplot background
            if step_id % 10 == 0 and self.obs_dim == 1:
                x = np.linspace(-3, 3, 100)
                # put each x value forward through the network
                q_values = self.dqn(torch.FloatTensor(x).unsqueeze(1).to(self.device)).detach().cpu().numpy()
                best_actions = np.argmax(q_values, axis=1)
                arm_weights.append((step_id, best_actions))

            if step_id % 50 == 0:
                score += np.mean(rewards[-50:])
                scores.append(score)
                if self.args.logging: wandb.log({"score": score})

            if args.noisy_net:
                self.dqn.resample_noise() # line 13
            else:
                self.epsilon = max(self.epsilon - 2/num_episodes, 0)

            # if training is ready
            if len(self.memory) >= self.args.batch_size:
                samples = self.memory.sample_batch() # line 12
                        
                loss = self._compute_dqn_loss(samples)
                losses.append(loss)
                if self.args.logging: wandb.log({"loss": loss})
                
        if self.args.logging:
            wandb.run.summary["mean_regret"] = np.mean(regrets[-100:])
        
        if self.args.plotting:
            self._plot(rewards, scores, losses, regrets, arm_weights, np.array(data), epsilons)

        self.env.close()

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
        rewards: List[float],
        scores: List[float], 
        losses: List[float],
        regrets: List[float],
        arm_weights: List[np.ndarray],
        data: np.ndarray,
        epsilons: List[float]
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(17, 5))

        plt.subplot(131)
        plt.title('Rewards and Scores')
        plt.plot(rewards, label='Reward', alpha=0.5)
        plt.plot(np.arange(0, len(rewards), 50), scores, label='Score (mean of 50)', linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Value')
        plt.legend()

        plt.subplot(132)
        plt.title('Loss')
        plt.plot(losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')

        plt.subplot(133)
        plt.title('Regret')
        plt.plot(regrets)
        plt.xlabel('Training Step')
        plt.ylabel('Regret')
        
        if not args.noisy_net:
            plt.figure(figsize=(6, 6))
            plt.plot(epsilons)
        # ------------------------------------------------------

        if self.obs_dim == 1:
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
            sample_data = sample_env(self.args)

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
        plt.show()

        if self.args.logging:
            wandb.log({"Reward Scatter": wandb.Image(fig)})

def sample_env(args, num_samples=1000):
    """somehow just sampling the reward functions didn't work"""
    _env = gym.make(
                args.env_id,
                arms=args.arms,
                dynamic_rate=args.dynamic_rate,
                seed=args.seed,
                noisy = False)
    state, _ = _env.reset()
    _data = []
    for _ in range(num_samples):
        action = _env.action_space.sample()
        next_state, reward, _, _, _ = _env.step(action)
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
        arms=args.arms,
        dynamic_rate=args.dynamic_rate,
        seed=args.seed,
        noisy=args.noisy
    )

    agent = DQNAgent(env, args)

    print(f"[ Environment: '{args.env_id}' | Type: {args.noisy_net} | Seed: {args.seed} | Device: {agent.device} ]")

    agent.train(args.num_episodes)

    if args.logging:
        wandb.finish()
