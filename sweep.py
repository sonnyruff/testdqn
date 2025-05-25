import wandb
from conbandit_noisydqn3_0 import wandb_sweep

wandb.login()

sweep_config = {
    "method": "random",
    "metric": {"name": "score", "goal": "maximize"},
    "parameters": {
        "seed": {"min": 0, "max": 10000},
        "batch_size": {"values": [32, 64]},
        "memory_size": {"values": [500, 1000]},
        "optimal_std": {"values": [0.5, 1.0]},
        "max_suboptimal_mean": {"values": [3, 5]}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="noisynet-dqn")

# Launch agent with training function
wandb.agent(sweep_id, function=wandb_sweep, count=2)
