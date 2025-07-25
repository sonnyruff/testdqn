import wandb
run = wandb.init()
# run_0_name = 's-ruff-tu-delft/NoisyNeuralNet-v7/run-vyn8tc47-history:v0' # seed 0
# run_1_name = 's-ruff-tu-delft/NoisyNeuralNet-v7/run-8czromre-history:v1' # seed 1
# run_2_name = 's-ruff-tu-delft/NoisyNeuralNet-v7/run-2miwabkf-history:v0' # seed 2
reg_run_0_name = 's-ruff-tu-delft/NoisyNeuralNet-v7/run-emhk2e0g-history:v0' # regular seed 0
reg_run_1_name = 's-ruff-tu-delft/NoisyNeuralNet-v7/run-su1agzym-history:v0'
reg_run_2_name = 's-ruff-tu-delft/NoisyNeuralNet-v7/run-7jin0xg3-history:v0'
artifact = run.use_artifact(reg_run_1_name, type='wandb-history')
artifact_dir = artifact.download()
artifact = run.use_artifact(reg_run_2_name, type='wandb-history')
artifact_dir = artifact.download()

# import pandas as pd
# df01 = pd.read_parquet('artifacts/run-8czromre-history-v1/0000.parquet')
# df02 = pd.read_parquet('artifacts/run-8czromre-history-v1/0001.parquet')
# df1 = pd.read_parquet('artifacts/run-vyn8tc47-history-v0/0000.parquet')
# df2 = pd.read_parquet('artifacts/run-2miwabkf-history-v0/0000.parquet')
# # _df11 = df11[['reward']]
# print(df2.info())