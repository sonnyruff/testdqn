"""
conbandit_dqn2_0.py

DQN implementation for ContextualBandit-v1 - Continuous input state
Single loop version

Author: Sonny Ruff
Date: 12-05-2025

Based on:
- NoisyNet-DQN implementation from https://nbviewer.org/github/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb
- Parts of https://github.com/knyazer/nanodqn/tree/main
- OpenAI Gym (https://github.com/openai/gym) & Buffalo Gym environment (https://github.com/foreverska/buffalo-gym)
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
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = np.random.randint(0, 10000)
    """seed of the experiment"""
    wandb_project_name: str = "noisynet-dqn"
    """the wandb's project name"""
    logging: bool = True
    """whether to log to wandb"""

    env_id: str = "ContextualBandit-v1"
    """the id of the environment"""
    num_episodes: int = 5000
    """the number of episodes to run"""
    memory_size: int = 1000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_update: int = 50
    """the timesteps it takes to update the target network"""
    batch_size: int = 50
    """the batch size of sample from the reply memory"""

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

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.feature = nn.Linear(in_dim, out_dim)
        self.noisy_layer1 = NoisyLinear(out_dim, out_dim)
        self.noisy_layer2 = NoisyLinear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = F.relu(self.feature(x))
        hidden = F.relu(self.noisy_layer1(feature))
        out = self.noisy_layer2(hidden)
        
        return out
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()

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
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        gamma: float = 0.99,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            gamma (float): discount factor
        """
        # NoisyNet: All attributes related to epsilon are removed
        obs_dim = env.observation_space.shape[0]
        # obs_dim = env.unwrapped.states # WRONG
        action_dim = env.action_space.n
        
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        print(self.dqn)
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, state: np.ndarray, action: np.ndarray) -> float:
        """Take an action and return the response of the env."""
        next_state, reward, _, _, _ = self.env.step(action)
        # done = terminated or truncated
        
        if not self.is_test:
            # self.transition += [reward, next_state, done]
            self.memory.store(state, action, reward)
    
        return next_state, reward

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
        
    def train(self, num_episodes: int):
        """Train the agent."""
        self.is_test = False
        
        update_cnt = 0
        losses = []
        scores = []
        arm_weights = []
        data = []

        state, _ = self.env.reset(seed=self.seed)

        # Double loop isn't necessary
        for step_id in tqdm(range(1, num_episodes + 1)):
            score = 0
            
            action = self.select_action(state)
            next_state, reward = self.step(state, action)

            data.append([step_id, float(state[0]), action, float(reward)])

            state = next_state
            score += reward

            q_values = self.dqn(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
            arm_weights.append((state.astype(int), q_values))

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                if args.logging: wandb.log({"loss": loss})
                noise_l1 = self.dqn.noisy_layer1.get_noise()
                noise_l2 = self.dqn.noisy_layer2.get_noise()
                if args.logging: wandb.log({
                    "noisy_layer1/weight_epsilon_std": np.std(noise_l1["weight_epsilon"]),
                    "noisy_layer1/bias_epsilon_std": np.std(noise_l1["bias_epsilon"]),
                    "noisy_layer2/weight_epsilon_std": np.std(noise_l2["weight_epsilon"]),
                    "noisy_layer2/bias_epsilon_std": np.std(noise_l2["bias_epsilon"])
                })
                
                update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
            
            scores.append(score)
            if args.logging: wandb.log({"score": score})
                
        self.env.close()
        self._plot(scores, losses, arm_weights, np.array(data))
        
    def test(self, episode_length) -> None:
        """Test the agent."""
        self.is_test = True
        
        # for recording a video
        naive_env = self.env
        # self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        state, _ = self.env.reset()
        score = 0
        
        for _ in range(episode_length):
            action = self.select_action(state)
            next_state, reward = self.step(state, action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        # next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        # done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        # next_q_value = self.dqn_target(next_state).max(
        #     dim=1, keepdim=True
        # )[0].detach()
        # mask = 1 - done
        # target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        # loss = F.smooth_l1_loss(curr_q_value, target)
        loss = F.mse_loss(curr_q_value, reward)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
        self,
        scores: List[float], 
        losses: List[float],
        arm_weights: List[np.ndarray],
        data: np.ndarray
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

        # context_dict = {}
        # for state, q_values in arm_weights:
        #     ctx_key = tuple(state)
        #     if ctx_key not in context_dict:
        #         context_dict[ctx_key] = []
        #     context_dict[ctx_key].append(q_values)

        # n_contexts = len(context_dict)
        # fig, axs = plt.subplots(1, n_contexts, figsize=(3 * n_contexts, 5), squeeze=False)
        # fig.suptitle("Arm selection heatmaps by Context", fontsize=16)

        # for idx, (ctx, q_values_list) in enumerate(context_dict.items()):
        #     ax = axs[0, idx]
        #     q_matrix = np.array(q_values_list)
        #     im = ax.imshow(q_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        #     ax.set_title(f'Context: {int(ctx[0])}')
        #     ax.set_xlabel('Action Index')
        #     ax.set_ylabel('Training Step')
        #     fig.colorbar(im, ax=ax)

        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # if args.logging: wandb.log({"Q-value Heatmaps": wandb.Image(fig)})

        plt.figure(figsize=(15, 10))
        plt.subplot(211)
        scatter1 = plt.scatter(data[:, 1], data[:, 0], c=data[:, 2], cmap="viridis", alpha=0.6)
        plt.colorbar(scatter1, label="Action")
        plt.xlabel("State")
        plt.ylabel("Reward")
        plt.grid(True)

        plt.subplot(212)
        scatter2 = plt.scatter(data[:, 1], data[:, 3], c=data[:, 2], cmap="viridis", alpha=0.6)
        plt.colorbar(scatter2, label="Action")
        plt.xlabel("State")
        plt.ylabel("Reward")
        plt.grid(True)
        # plt.show()
        if args.logging: wandb.log({"Reward Scatter": wandb.Image(scatter)})




        num_state_bins = 50
        num_step_bins = 50

        # Create 2D bins
        state_vals = data[:, 1]
        step_vals = data[:, 0]
        action_vals = data[:, 2]

        heatmap, xedges, yedges = np.histogram2d(
            state_vals, step_vals, bins=[num_state_bins, num_step_bins], weights=action_vals
        )
        counts, _, _ = np.histogram2d(state_vals, step_vals, bins=[xedges, yedges])

        # Avoid divide-by-zero
        heatmap_avg = np.divide(heatmap, counts, where=counts != 0)

        # Plot the heatmap
        plt.figure(figsize=(12, 6))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap_avg.T, extent=extent, origin='lower', aspect='auto', cmap="viridis")
        plt.colorbar(label="Average Action")
        plt.xlabel("State")
        plt.ylabel("Step")
        plt.title("Heatmap of Average Action (State vs Step)")
        plt.grid(False)
        plt.show()



####################################################################################################

if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args.seed)
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

    agent = DQNAgent(env, args.memory_size, args.batch_size, args.target_update, args.seed, args.gamma)
    agent.train(args.num_episodes)
    agent.test(100)

    if args.logging: wandb.finish()
