import wandb
from conbandit_combi1_0 import wandb_sweep

wandb.login()

project_name = "NoisyNeuralNet-v4"

all_config = {
    "method": "random",
    # "metric": {"name": "mean_rewards", "goal": "maximize"},
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [True, False]},
        "env_id": {"values": ["ContextualBandit-v2", "MNISTBandit-v0", "NNBandit-v0"]},
        # "env_id": {"values": ["NNBandit-v0"]},
        # "memory_size": {"values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
        # "batch_size": {"values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        # "noisy_layer_distr_type": {"values": ["normal", "uniform"]},
        # "noisy_layer_init_std": {"values": [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4, 10e5]},
        "noisy_layer_init_std": {"values": [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1]},
        "hidden_layer_size": {"values": [16, 20, 24, 40, 80]},
        "noisy_output_layer": {"values": [True, False]},
        # "dynamic_rate": {"values": [None, 100, 200, 400, 600, 800, 1000]},
        "noisy_reward": {"values": [True, False]},
    }
}
medium_config = {
    "method": "random",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [True, False]},
        "env_id": {"values": ["ContextualBandit-v2"]},
        "noisy_layer_init_std": {"values": [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1]},
        "hidden_layer_size": {"values": [16, 20, 24, 40, 80]},
    }
}
small_config = {
    "method": "random",
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        "wandb_project_name": {"value": project_name},
        "noisy_net": {"values": [True]},
        "env_id": {"values": ["ContextualBandit-v2"]},
        "memory_size": {"values": [100]},
        "batch_size": {"values": [10]},
        "noisy_layer_distr_type": {"values": ["normal"]},
        "noisy_layer_init_std": {"values": [10e0]},
        "hidden_layer_size": {"values": [10]},
        "noisy_output_layer": {"values": [True]},
        "dynamic_rate": {"values": [None]},
        "noisy_reward": {"values": [False]},
    }
}


# Initialize sweep
sweep_id = wandb.sweep(sweep=medium_config, project=project_name)

# Launch agent with training function
wandb.agent(sweep_id, function=wandb_sweep, count=100)

