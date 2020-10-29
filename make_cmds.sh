num_trials=5
seeds=(0 1 2 3 4 5)
envs=(
    "bigfish" 
    "plunder" 
    "chaser" 
    "dodgeball"
    "miner"
)

for seed in "${seeds[@]}"
do
    for env in "${envs[@]}"
    do
echo "python train.py \\
--env_name=$env \\
--use_ucb=True \\
--num_processes=64 \\
--log_interval=10 \\
--save_interval=10 \\
--num_levels=200 \\
--save_dir='~/logs/autodrac' \\
--log_dir='~/logs/autodrac' \\
--seed=$seed \\
--run_name=ucb-drac \\
--wandb_entity=level-replay \\
--wandb_project=ucb-drac
"
    done
done