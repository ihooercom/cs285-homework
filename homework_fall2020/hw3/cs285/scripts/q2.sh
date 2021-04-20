python=/home/amax/anaconda3/envs/cs285/bin/python
$python homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --which_gpu 0 --seed 1 > x/logs/log 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --which_gpu 0 --seed 2 > x/logs/log 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --which_gpu 0 --seed 3 > x/logs/log 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --which_gpu 1 --seed 1 > x/logs/log 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --which_gpu 1 --seed 2 > x/logs/log 2>&1 &
$python homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --which_gpu 1 --seed 3 > x/logs/log 2>&1 &
wait 
echo "done!"