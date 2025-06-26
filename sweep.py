import wandb
import numpy as np
from conbandit_combi1_0 import wandb_sweep

wandb.login()

project_name = "NoisyNeuralNet-v6"

config_random = {
    "method": "random",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"distribution": "int_uniform", "min": 0, "max": 1000},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": True},
        "env_id": {"values": ["ContextualBandit-v2", "MNISTBandit-v0", "NNBandit-v0"]},
        # "memory_size": {"values": [500, 1000, 1500, 2000]},
        # "batch_size": {"values": [50, 100, 150, 200, 300, 500]},
        # "noisy_layer_distr_type": {"values": ["normal", "uniform"]},
        "noisy_layer_init_std": {"values": np.arange(0.55, 1.05, 0.05).tolist()},
    }
}
config_grid = {
    "method": "grid",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"values": np.arange(10).tolist()},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [True, False]},
        "env_id": {"values": ["ContextualBandit-v2", "MNISTBandit-v0", "NNBandit-v0"]},
        "hidden_layer_size": {"values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        # "noisy_layer_init_std": {"values": np.arange(0.05, 1.05, 0.05).tolist()},
    }
}
config_grid_nonstationary = {
    "method": "grid",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"values": np.arange(6).tolist()},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": True},
        "env_id": {"values": ["ContextualBandit-v2", "NNBandit-v0"]},
        "memory_size": {"values": [500, 1000, 1500, 2000]},
        "batch_size": {"values": [2, 5, 10, 20, 30, 50, 100, 200]},
        "dynamic_rate": {"values": [200, 1000]}
    }
}
batch = {
    "method": "random",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"distribution": "int_uniform", "min": 0, "max": 1000},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": False},
        "env_id": {"value": "NNBandit-v0"},
        "dynamic_rate": {"value": 1000},
        "batch_size": {"value": 50}
    }
}
no_noisy = {
    "method": "grid",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"values": np.arange(30).tolist()},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": False},
        "env_id": {"values": ["ContextualBandit-v2", "MNISTBandit-v0", "NNBandit-v0"]}
    }
}
mnist_grid = {
    "method": "grid",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"value": 1},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [False]},
        "env_id": {"values": ["MNISTBandit-v0"]},
        "hidden_layer_size": {"values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
        "num_episodes": {"values": [2000, 4000, 6000, 8000, 10000]},
        # "noisy_layer_init_std": {"values": np.arange(0.05, 1.05, 0.05).tolist()},
    }
}

sweep_id = wandb.sweep(sweep=mnist_grid, project=project_name)
# wandb.agent(sweep_id, function=wandb_sweep, count=30)
wandb.agent(sweep_id, function=wandb_sweep)

