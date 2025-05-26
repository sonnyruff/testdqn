import wandb
from conbandit_noisydqn3_0 import wandb_sweep

wandb.login()

sweep_config = {
    "method": "random",
    "metric": {"name": "mean_rewards", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        "memory_size": {"values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
        "hidden_layer_size": {"min": 4, "max": 9}, # e.g. 2**6 = 64
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="noisynet-dqn-new")

# Launch agent with training function
wandb.agent(sweep_id, function=wandb_sweep, count=100)
