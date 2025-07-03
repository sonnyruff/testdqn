from dataclasses import dataclass

import wandb
import numpy as np
import tyro
from conbandit_combi1_0 import wandb_sweep

@dataclass
class Args:
    config: str = ""

project_name = "NoisyNeuralNet-v7"

# config_random = {
#     "method": "random",
#     "metric": {"name": "regret", "goal": "minimize"},
#     "parameters": {
#         "seed": {"distribution": "int_uniform", "min": 0, "max": 1000},
#         "wandb_project_name": {"value": project_name},
#         "noisy_net": {"value": False},
#         "env_id": {"value": "ContextualBandit-v2"},
#         "noisy_layer_distr_type": {"values": ["normal", "uniform"]},
#         "noisy_output_layer": {"values": [True, False]},
#         "noisy_reward": {"values": [True, False]},
#         "hidden_layer_size": {"values": [4, 8, 10, 16, 20, 24, 40, 80]},
#         "noisy_layer_init_std": {"values": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]},
#     }
# }

# config_grid = {
#     "method": "grid",
#     "metric": {"name": "regret", "goal": "minimize"},
#     "parameters": {
#         "seed": {"values": np.arange(10).tolist()},
#         "wandb_project_name": {"value": project_name},
#         "noisy_net": {"value": True},
#         "env_id": {"values": ["ContextualBandit-v2", "NNBandit-v0"]},
#         "noisy_layer_init_std": {"values": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]},
#     }
# }
# config_grid_nonstationary = {
#     "method": "grid",
#     "metric": {"name": "mean_regret", "goal": "minimize"},
#     "parameters": {
#         "seed": {"values": np.arange(6).tolist()},
#         "wandb_project_name": {"value": project_name},
#         "noisy_net": {"value": True},
#         "env_id": {"values": ["ContextualBandit-v2", "NNBandit-v0"]},
#         "memory_size": {"values": [500, 1000, 1500, 2000]},
#         "batch_size": {"values": [2, 5, 10, 20, 30, 50, 100, 200]},
#         "dynamic_rate": {"values": [200, 1000]}
#     }
# }
batch = {
    "method": "grid",
    "metric": {"name": "regret", "goal": "minimize"},
    "parameters": {
        # "seed": {"distribution": "int_uniform", "min": 0, "max": 1000},
        "seed": {"values": np.arange(10).tolist()},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [True, False]},
        "dynamic_rate": {"values": [None]},
        "env_id": {"values": ["NNBandit-v0"]},
    }
}

# mnist_grid = {
#     "method": "grid",
#     "metric": {"name": "regret", "goal": "minimize"},
#     "parameters": {
#         "seed": {"values": [2]},
#         "wandb_project_name": {"value": project_name},
#         "noisy_net": {"value": True},
#         "env_id": {"values": ["MNISTBandit-v0"]},
#         "noisy_layer_init_std": {"values": [0.2, 0.4, 0.6, 0.8, 1.0]},
#         "num_episodes": {"value": 300000},
#         "hidden_layer_size": {"value": 100},
#         "memory_size": {"value": 50000},
#         "batch_size": {"value": 64},
#     }
# }
mnist_extx1 = { # DON'T FORGET TO CHANGE THE INIT_STD
    "method": "grid",
    "metric": {"name": "regret", "goal": "minimize"},
    "parameters": {
        "seed": {"values": [0]},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": True},
        "env_id": {"values": ["MNISTBandit-v0"]},
        "num_episodes": {"value": 300000},
        "hidden_layer_size": {"value": 100},
        "memory_size": {"value": 50000},
        "batch_size": {"value": 64},
        "noisy_layer_init_std": {"values": [1e-4, 1e4]},
    }
}
mnist_extx2 = { # DON'T FORGET TO CHANGE THE INIT_STD
    "method": "grid",
    "metric": {"name": "regret", "goal": "minimize"},
    "parameters": {
        "seed": {"values": [1]},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": True},
        "env_id": {"values": ["MNISTBandit-v0"]},
        "num_episodes": {"value": 300000},
        "hidden_layer_size": {"value": 100},
        "memory_size": {"value": 50000},
        "batch_size": {"value": 64},
        "noisy_layer_init_std": {"values": [1e-4, 1e4]},
    }
}

mnist_ext1 = {
    "method": "grid",
    "metric": {"name": "regret", "goal": "minimize"},
    "parameters": {
        "seed": {"values": [0]},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": True},
        "env_id": {"values": ["MNISTBandit-v0"]},
        "num_episodes": {"value": 300000},
        "hidden_layer_size": {"value": 100},
        "memory_size": {"value": 50000},
        "batch_size": {"value": 64},
    }
}
mnist_ext2 = {
    "method": "grid",
    "metric": {"name": "regret", "goal": "minimize"},
    "parameters": {
        "seed": {"values": [1]},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": True},
        "env_id": {"values": ["MNISTBandit-v0"]},
        "num_episodes": {"value": 300000},
        "hidden_layer_size": {"value": 100},
        "memory_size": {"value": 50000},
        "batch_size": {"value": 64},
    }
}

if __name__ == "__main__":
    _args = tyro.cli(Args)
    print(f"Sweeping with config:\n{_args.config}")

    sweep_configs = {
        # "config_random": config_random,
        # "config_grid": config_grid,
        # "config_grid_nonstationary": config_grid_nonstationary,
        # "mnist_grid": mnist_grid,
        "batch": batch,
        "mnist_ext1": mnist_ext1,
        "mnist_ext2": mnist_ext2
    }

    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configs[_args.config], project=project_name)
    # wandb.agent(sweep_id, function=wandb_sweep, count=10)
    wandb.agent(sweep_id, function=wandb_sweep)