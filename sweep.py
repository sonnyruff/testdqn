import wandb
from conbandit_noisydqn3_0 import wandb_sweep

wandb.login()

sweep_config = {
    "method": "random",
    "metric": {"name": "mean_rewards", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [32, 64]},
        "memory_size": {"values": [100, 1000]},
        "hidden_layer_size": {"min": 4, "max": 9}, # e.g. 2**6 = 64
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="noisynet-dqn-new")

# Launch agent with training function
wandb.agent(sweep_id, function=wandb_sweep, count=100)
