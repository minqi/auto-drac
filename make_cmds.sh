num_trials=5
seeds=(0 1 2 3 4)
envs=(
	"bigfish"
	"bossfight"
	"caveflyer"
	"chaser"
	"climber"
	"coinrun"
	"dodgeball"
	"fruitbot"
	"heist"
	"jumper"
	"leaper"
	"maze"
	"miner"
	"ninja"
	"plunder"
	"starpilot"
)

for seed in "${seeds[@]}"
do
    for env in "${envs[@]}"
    do
echo "python train.py \\
--env_name=$env \\
--use_ucb=True \\
--num_processes=64 \\
--log_interval=1 \\
--save_interval=10 \\
--num_levels=200 \\
--save_dir='~/logs/autodrac' \\
--log_dir='~/logs/autodrac' \\
--seed=$seed \\
--run_name=ucb-drac-log1 \\
--wandb_entity=level-replay \\
--wandb_project=ucb-drac
"
    done
done
