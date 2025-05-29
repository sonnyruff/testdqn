import wandb
from conbandit_noisydqn3_0 import wandb_sweep

wandb.login()

sweep_config = {
    "method": "random",
    # "metric": {"name": "mean_rewards", "goal": "maximize"},
    "metric": {"name": "mean_regret", "goal": "minimize"},
    "parameters": {
        # "batch_size": {"values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        # "memory_size": {"values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
        # "hidden_layer_size": {"min": 4, "max": 9}, # e.g. 2**6 = 64
        "hidden_layer_size": {"values": [4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 60, 80]},
        "pace": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        "noisy_layer_distr_type": {"values": ["normal", "uniform"]},
        "noisy_layer_init_std": {"values": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "noisy_output": {"values": [True, False]},
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="noisynet-dqn-new")

# Launch agent with training function
wandb.agent(sweep_id, function=wandb_sweep, count=100)
