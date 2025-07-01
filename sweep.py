import wandb
import numpy as np
from conbandit_combi1_0 import wandb_sweep

wandb.login()

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
        "noisy_net": {"value": True},
        "env_id": {"values": ["ContextualBandit-v2", "NNBandit-v0"]},
    }
}
batch_dyn200 = {
    "method": "grid",
    "metric": {"name": "regret", "goal": "minimize"},
    "parameters": {
        # "seed": {"distribution": "int_uniform", "min": 0, "max": 1000},
        "seed": {"values": np.arange(10).tolist()},
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"value": True},
        "dynamic_rate": {"value": 200},
        "env_id": {"values": ["ContextualBandit-v2", "NNBandit-v0"]},
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

sweep_id = wandb.sweep(sweep=batch_dyn200, project=project_name)
wandb.agent(sweep_id, function=wandb_sweep, count=10)
# wandb.agent(sweep_id, function=wandb_sweep)

# sweep_id = wandb.sweep(sweep=mnist_grid, project=project_name)
# # wandb.agent(sweep_id, function=wandb_sweep, count=100)
# wandb.agent(sweep_id, function=wandb_sweep) 