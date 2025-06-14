import wandb
import numpy as np
from conbandit_combi1_0 import wandb_sweep

wandb.login()

project_name = "NoisyNeuralNet-v6"


config_1 = {
    "method": "random",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"values": np.arange(1000)},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [True, False]},
        "env_id": {"values": ["ContextualBandit-v2", "MNISTBandit-v0", "NNBandit-v0"]},
        "memory_size": {"values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
        "batch_size": {"values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        "noisy_layer_distr_type": {"values": ["normal", "uniform"]},
        "noisy_layer_init_std": {"values": [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4, 10e4]},
        # "noisy_layer_init_std": {"values": [10e-2, 10e-1, 10e0, 10e1]},
        "hidden_layer_size": {"values": [8, 16, 20, 24, 40, 80, 160, 300, 500]},
    }
}
config_2 = {
    "method": "random",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"values": np.arange(1000)},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [True, False]},
        "env_id": {"values": ["ContextualBandit-v2", "MNISTBandit-v0", "NNBandit-v0"]},
        "memory_size": {"values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
        "batch_size": {"values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        "noisy_layer_distr_type": {"values": ["normal", "uniform"]},
        "noisy_layer_init_std": {"values": [10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2]},
        # "noisy_layer_init_std": {"values": [10e-2, 10e-1, 10e0, 10e1]},
        "hidden_layer_size": {"values": [8, 16, 20, 24, 40, 80, 160, 300, 500]},
    }
}
config_3 = {
    "method": "random",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"distribution": "int_uniform", "min": 0, "max": 1000},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [True, False]},
        "env_id": {"values": ["ContextualBandit-v2", "MNISTBandit-v0", "NNBandit-v0"]},
        "memory_size": {"values": [500, 1000, 1500, 2000]},
        "batch_size": {"values": [50, 100, 150, 200, 300, 500]},
        "noisy_layer_distr_type": {"values": ["normal", "uniform"]},
        "noisy_layer_init_std": {"distribution": "log_uniform_values", "min": 0.1, "max": 5.0},
        "hidden_layer_size": {"values": [8, 16, 20, 24, 40, 80, 160, 300, 500]},
    }
}
config_4 = {
    # "method": "random",
    "method": "grid",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        # "seed": {"distribution": "int_uniform", "min": 0, "max": 1000},
        "seed": {"values": np.arange(6).tolist()},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": True},
        "env_id": {"values": ["ContextualBandit-v2", "MNISTBandit-v0", "NNBandit-v0"]},
        # "memory_size": {"values": [500, 1000, 1500, 2000]},
        # "batch_size": {"values": [50, 100, 150, 200, 300, 500]},
        # "noisy_layer_distr_type": {"values": ["normal", "uniform"]},
        # "noisy_layer_init_std": {"values": [0.01, 0.5, 1.0, 2.0]},
        # "noisy_layer_init_std": {"values": np.linspace(0.01, 2, 6).tolist()},
        "noisy_layer_init_std": {"values": np.arange(0.05, 0.65, 0.05).tolist()},
    }
}
batch = {
    "method": "random",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "seed": {"distribution": "int_uniform", "min": 0, "max": 1000},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [True, False]},
        "env_id": {"values": ["ContextualBandit-v2", "MNISTBandit-v0", "NNBandit-v0"]},
    }
}


sweep_id = wandb.sweep(sweep=batch, project=project_name)
wandb.agent(sweep_id, function=wandb_sweep, count=100)
# wandb.agent(sweep_id, function=wandb_sweep)

