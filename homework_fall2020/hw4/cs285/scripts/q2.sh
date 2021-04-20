python=/home/amax/anaconda3/envs/cs285/bin/python
src=homework_fall2020/hw4

$python $src/cs285/scripts/run_hw4_mb.py --exp_name q2_obstacles_singleiteration --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --which_gpu 0 --seed 2 > x/logs/log1 2>&1 &

wait 
echo "done!"