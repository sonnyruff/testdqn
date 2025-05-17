@echo off
setlocal enabledelayedexpansion

for /L %%i in (1,1,5) do (
    set /a seed=!RANDOM!
    echo [%%i] Using seed: !seed!
    py conbandit_dqn2_0.py --wandb-project-name noisynet-dqn-new --seed !seed!
    py conbandit_noisydqn2_1.py --wandb-project-name noisynet-dqn-new --seed !seed!
)

endlocal