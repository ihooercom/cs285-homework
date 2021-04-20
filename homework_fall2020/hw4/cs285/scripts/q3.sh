python=/home/amax/anaconda3/envs/cs285/bin/python
src=homework_fall2020/hw4

$python $src/cs285/scripts/run_hw4_mb.py --exp_name q3_obstacles --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --n_iter 12 --which_gpu 0   > x/logs/logx1 2>&1 &
$python $src/cs285/scripts/run_hw4_mb.py --exp_name q3_reacher --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 5000 --n_iter 15   --which_gpu 1   > x/logs/logx2 2>&1 &
$python $src/cs285/scripts/run_hw4_mb.py --exp_name q3_cheetah --env_name cheetah-cs285-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 20   --which_gpu 0   > x/logs/logx3 2>&1 &
wait 
echo "done!"