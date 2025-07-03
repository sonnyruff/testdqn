"""
NoisyNet-DQN implementation for ContextualBandit-v2, MNISTBandit-v0 and NNBandit-v0
Continuous input state
Single loop version
Single network

Author: Sonny Ruff
Date: 12-05-2025

Based on:
- NoisyNet-DQN implementation from https://nbviewer.org/github/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb
- Parts of https://github.com/knyazer/nanodqn/tree/main
- OpenAI Gym (https://github.com/openai/gym) & Buffalo Gym environment (https://github.com/foreverska/buffalo-gym)

e.g.
py conbandit_combi1_0.py --no-logging --plotting
py conbandit_combi1_0.py --no-logging --plotting --env-id MNISTBandit-v0
py conbandit_combi1_0.py --no-logging --plotting --env-id NNBandit-v0
py conbandit_combi1_0.py --no-logging --plotting --no-noisy-net
py conbandit_combi1_0.py --no-logging --plotting --env-id MNISTBandit-v0 --no-noisy-net
py conbandit_combi1_0.py --no-logging --plotting --env-id NNBandit-v0 --no-noisy-net
"""
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import pandas as pd
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
    """whether to use NoisyNet"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = np.random.randint(0, 10000)
    """seed of the experiment"""
    wandb_project_name: str = "NoisyNeuralNet-v7"
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

    noisy_layer_distr_type: str = "uniform" # or normal
    """the distribution of the noisy layer"""
    noisy_layer_init_std: float = 0.
    """the initial standard deviation of the noisy layer"""

    hidden_layer_size: int = 40
    noisy_output_layer: bool = False
    """whether to have to last layer of the network be a NoisyLinear layer"""

    arms: int = 10
    dynamic_rate: int | None = None
    noisy_reward: bool = False
    """whether to add noise to the reward"""

####################################################################################################

class NoisyNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, distr_type: str, init_std: float, noisy_output_layer: bool):
        super().__init__()

        print(init_std)

        self.noisy_output_layer = noisy_output_layer

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim, distr_type, init_std)
        self.relu2 = nn.ReLU()

        if noisy_output_layer:
            self.fc3 = NoisyLinear(hidden_dim, out_dim, distr_type, init_std)
        else:
            self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, use_noise: bool = True) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x, use_noise)
        x = self.relu2(x)
        x = self.fc3(x, use_noise) if self.noisy_output_layer else self.fc3(x)
        return x

    def resample_noise(self):
        self.fc2.resample_noise()
        if self.noisy_output_layer:
            self.fc3.resample_noise()

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
    """

    def __init__(
        self, 
        env: gym.Env,
        args: Args = None,
        is_sweep: bool = False
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
        """
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.env = env
        self.args = args
        self.memory = ReplayBuffer(self.obs_dim, self.args.memory_size, self.args.batch_size)
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        match args.env_id:
            case "ContextualBandit-v2":
                self.args.noisy_layer_init_std = 0.7
                self.args.hidden_layer_size = 40
                match args.dynamic_rate:
                    case None:
                        self.epsilon = 0.218
                    case 200:
                        self.epsilon = 0.218
                    case 1000:
                        self.epsilon = 0.218
            case "MNISTBandit-v0":
                self.args.noisy_layer_init_std = 0.5 # DON'T FORGET TO REMOVE THIS
                self.args.hidden_layer_size = 100
                self.epsilon = 1
            case "NNBandit-v0":
                self.args.noisy_layer_init_std = 0.4
                self.args.hidden_layer_size = 24
                match args.dynamic_rate:
                    case None:
                        self.epsilon = 0.219
                    case 200:
                        self.epsilon = 0.219
                    case 1000:
                        self.epsilon = 0.219

        print(self.args.noisy_layer_init_std)

        if self.args.noisy_net:
            self.dqn = NoisyNetwork(self.obs_dim,
                           self.args.hidden_layer_size,
                           self.action_dim,
                           self.args.noisy_layer_distr_type,
                           self.args.noisy_layer_init_std,
                           self.args.noisy_output_layer
                        ).to(self.device)
        else:
            self.dqn = Network(self.obs_dim,
                           self.args.hidden_layer_size,
                           self.action_dim
                        ).to(self.device)
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())
        
        # mode: train / test
        self.is_test = False
        self.is_sweep = is_sweep

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        if self.args.noisy_net:
            _predictions = self.dqn(torch.FloatTensor(state).to(self.device), use_noise=True)
            selected_action = _predictions.argmax().detach().numpy()
        else:
            if np.random.rand() < self.epsilon:
                selected_action = self.env.action_space.sample()
            else:
                _predictions = self.dqn(torch.FloatTensor(state).to(self.device))
                selected_action = _predictions.argmax().detach().numpy()
        
        return selected_action

    def sample_exploration_rate(self, state: np.ndarray) -> np.ndarray:
        match_count = 0
        sample_size = 10
        non_noisy_action = self.dqn(torch.FloatTensor(state).to(self.device), use_noise=False).argmax().detach().numpy()
        for _ in range(sample_size):
            noisy_action = self.dqn(torch.FloatTensor(state).to(self.device), use_noise=True).argmax().detach().numpy()
            if non_noisy_action == noisy_action:
                match_count += 1
            self.dqn.resample_noise()
            
        return 1. - float(match_count / sample_size)

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
        exploration_rates = []
        mean_exploration_rates = []
        visited_states = set()
        
        state, _ = self.env.reset(seed=self.args.seed)

        for step_id in tqdm(range(1, num_episodes + 1)):
            score = 0
            visited_states.add(tuple(state.tolist()))
            
            # if args.noisy_net:
            #     self.dqn.resample_noise() # line 5

            action = self.select_action(state) # line 6

            next_state, reward, info = self.step(state, action) # line 7

            data.append([step_id, float(state[0]), action, float(reward)]) # line 8

            # regrets.append(self.env.unwrapped.reward(state, action) - reward) # !!! THIS DOESN'T WORK
            regrets.append(info['total_regret'])
            rewards.append(reward)

            state = next_state

            if self.args.logging:
                wandb.log({"regret": info['total_regret']}, step=step_id)
            if not self.args.noisy_net:
                epsilons.append(self.epsilon)

            # Scatterplot background
            if step_id % 10 == 0 and self.obs_dim == 1:
                x = np.linspace(-3, 3, 100)
                # put each x value forward through the network
                q_values = self.dqn(torch.FloatTensor(x).unsqueeze(1).to(self.device)).detach().cpu().numpy()
                best_actions = np.argmax(q_values, axis=1)
                arm_weights.append((step_id, best_actions))

            if step_id % 1 == 0:
                score += np.mean(rewards[-50:])
                scores.append(score)
                if self.args.logging: wandb.log({"reward": reward}, step=step_id)

            # if training is ready
            if len(self.memory) >= self.args.batch_size:
                samples = self.memory.sample_batch() # line 12
                
                loss = self._compute_dqn_loss(samples)
                losses.append(loss)
                if self.args.logging: wandb.log({"loss": loss}, step=step_id)
            
            if self.args.noisy_net:
                if step_id % 1 == 0:
                    exploration_rate = self.sample_exploration_rate(state)
                    exploration_rates.append(exploration_rate)
                    mean_exploration_rate = np.mean(exploration_rates[-2:])
                    mean_exploration_rates.append(mean_exploration_rate)
                    if self.args.logging:
                        wandb.log({"exploration_rate": exploration_rate}, step=step_id)
                        wandb.log({"state": state}, step=step_id)
            else:
                # TODO rework
                exploration_rate = self.epsilon
                mean_exploration_rates.append(exploration_rate)
                if self.args.logging: wandb.log({"exploration_rate": exploration_rate}, step=step_id)

            if self.args.noisy_net:
                self.dqn.resample_noise() # line 13
            else:
                # non-stationary dyn200
                # if self.args.env_id == "ContextualBandit-v2":
                #     self.epsilon *= 0.998
                # elif self.args.env_id == "MNISTBandit-v0":
                #     if step_id > 150:
                #         self.epsilon -= .4/200
                # elif self.args.env_id == "NNBandit-v0":
                #     self.epsilon *= 0.996
                # non-stationary dyn1000
                # if self.args.env_id == "ContextualBandit-v2":
                #     self.epsilon *= 0.9965
                # elif self.args.env_id == "MNISTBandit-v0":
                #     self.epsilon = max(self.epsilon - 1/num_episodes, 0)
                # elif self.args.env_id == "NNBandit-v0":
                #     self.epsilon *= 0.9965
                # stationary
                # if self.args.env_id == "ContextualBandit-v2":
                #     if step_id >= 100:
                #         self.epsilon *= 0.997
                # elif self.args.env_id == "MNISTBandit-v0":
                #     if step_id > 150:
                #         self.epsilon -= .4/200
                # elif self.args.env_id == "NNBandit-v0":
                #     if step_id >= 130:
                #         self.epsilon *= 0.995
                # self.epsilon = max(self.epsilon, 0.)

                # self.epsilon = max(self.epsilon - 1/num_episodes, 0)

                match self.args.env_id:
                    case "ContextualBandit-v2":
                        match self.args.dynamic_rate:
                            case None:
                                self.epsilon = 0.179 * np.exp(-0.00267 * step_id) + 0.039
                            case 200:
                                self.epsilon = 0.174 * np.exp(-0.00325 * step_id) + 0.051
                            case 1000:
                                self.epsilon = 0.177 * np.exp(-0.00292 * step_id) + 0.044
                    case "MNISTBandit-v0":
                        self.epsilon = 1
                    case "NNBandit-v0":
                        match self.args.dynamic_rate:
                            case None:
                                self.epsilon = 0.207 * np.exp(-0.00965 * step_id) + 0.012
                            case 200:
                                self.epsilon = 0.209 * np.exp(-0.01039 * step_id) + 0.014
                            case 1000:
                                self.epsilon = 0.207 * np.exp(-0.00965 * step_id) + 0.012
                
        if self.args.logging:
            # wandb.run.summary["mean_regret"] = np.mean(regrets)
            wandb.run.summary["regret"] = regrets[len(regrets)-1]
        
        if not self.is_sweep:
            self._plot(rewards, scores, losses, regrets, arm_weights, np.array(data), epsilons, exploration_rates, mean_exploration_rates)
        
        print(f"Unique states visited during training: {len(visited_states)}")
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
        epsilons: List[float],
        exploration_rates: List[float],
        mean_exploration_rates: List[float]
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(17, 5))

        plt.subplot(131)
        plt.title('Rewards and Scores')
        plt.plot(rewards, label='Reward', alpha=0.5)
        # plt.plot(np.arange(0, len(rewards), 50), scores, label='Score (mean of 50)', linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Value')
        plt.legend()

        plt.subplot(132)
        plt.title('Loss')
        plt.plot(losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')

        # Regret
        plt.subplot(133)
        plt.title('Regret')
        plt.plot(regrets)
        plt.xlabel('Training Step')
        plt.ylabel('Regret')
        
        if not self.args.noisy_net:
            plt.figure(figsize=(6, 6))
            plt.plot(epsilons)

        # Exploration Rate
        plt.figure(figsize=(10, 5))
        plt.plot(exploration_rates, label='Exploration', alpha=0.5)
        ema = pd.Series(exploration_rates).ewm(span=10, adjust=False).mean()
        plt.plot(ema, label='EMA (span=10)', color='orange')
        plt.xlabel('Training Step')
        plt.ylabel('Exploration Rate')
        plt.legend()
        plt.title('Exploration Rate with EMA Smoothing')
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

            if self.args.logging:
                wandb.log({"Reward Scatter": wandb.Image(fig)})
        
        if self.args.plotting:
            plt.show()

        plt.close("all")


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
    _args = tyro.cli(Args)
    run_name = f"{_args.env_id}__{_args.seed}__{_args.dynamic_rate}"
    if _args.logging:wandb.init(
        project=_args.wandb_project_name,
        config=vars(_args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(_args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env = gym.make(
        _args.env_id,
        arms=_args.arms,
        dynamic_rate=_args.dynamic_rate,
        seed=_args.seed,
        noisy=_args.noisy_reward
    )

    agent = DQNAgent(env, _args)

    print(f"[ Environment: '{_args.env_id}' | Type: {_args.noisy_net} | Seed: {_args.seed} | Device: {agent.device} ]")

    agent.train(_args.num_episodes)

    if _args.logging:
        wandb.finish()

####################################################################################################

def wandb_sweep():
    with wandb.init() as run:
        config = wandb.config

        new_args = {k: v for k, v in config.items()}
        _args = Args(**new_args)

        run.name = f"{_args.env_id}__{_args.seed}__{_args.dynamic_rate}"

        np.random.seed(_args.seed)
        torch.manual_seed(_args.seed)

        _env = gym.make(
            _args.env_id,
            arms=_args.arms,
            dynamic_rate=_args.dynamic_rate,
            seed=_args.seed,
            noisy=_args.noisy_reward
        )

        _agent = DQNAgent(_env, _args, is_sweep=True)
        print(f"[ Environment: '{_args.env_id}' | Seed: {_args.seed} | Device: {_agent.device} ]")
        _agent.train(_args.num_episodes)
