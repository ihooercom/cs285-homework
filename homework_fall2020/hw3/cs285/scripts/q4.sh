python=/home/amax/anaconda3/envs/cs285/bin/python

mkdir -p x/logs
$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1  --which_gpu 0 --seed 0 --log_suffix __seed:0 >x/logs/log1 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1  --which_gpu 0 --seed 1 --log_suffix __seed:1 >x/logs/log1 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1  --which_gpu 0 --seed 2 --log_suffix __seed:2 >x/logs/log1 2>&1 &

$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_100_1 -ntu 100 -ngsptu 1 --which_gpu 0 --seed 0 --log_suffix __seed:0 >x/logs/log2 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_100_1 -ntu 100 -ngsptu 1 --which_gpu 0 --seed 1 --log_suffix __seed:1 >x/logs/log2 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_100_1 -ntu 100 -ngsptu 1 --which_gpu 0 --seed 2 --log_suffix __seed:2 >x/logs/log2 2>&1 &

$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_1_100 -ntu 1 -ngsptu 100 --which_gpu 1 --seed 0 --log_suffix __seed:0 >x/logs/log3 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_1_100 -ntu 1 -ngsptu 100 --which_gpu 1 --seed 1 --log_suffix __seed:1 >x/logs/log3 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_1_100 -ntu 1 -ngsptu 100 --which_gpu 1 --seed 2 --log_suffix __seed:2 >x/logs/log3 2>&1 &

$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_10_10 -ntu 10 -ngsptu 10 --which_gpu 1 --seed 0 --log_suffix __seed:0 >x/logs/log4 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_10_10 -ntu 10 -ngsptu 10 --which_gpu 1 --seed 1 --log_suffix __seed:1 >x/logs/log4 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_10_10 -ntu 10 -ngsptu 10 --which_gpu 1 --seed 2 --log_suffix __seed:2 >x/logs/log4 2>&1 &
wait 
echo "done!"