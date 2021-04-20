function generator() {
    echo 'export PYTHONPATH=homework_fall2020/hw2/:$PYTHONPATH'
    prefix='/home/amax/anaconda3/envs/allennlp-v1.1/bin/python homework_fall2020/hw2/cs285/scripts/run_hw2.py'
    params_list=(
        '--env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa'
        '--env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa'
        '--env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na'
        '--env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa'
        '--env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa'
        '--env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na'
    )
    seeds=(1 2 3 4 5 6 7 8 9 10)

    i=0
    for params in "${params_list[@]}"; do
        for seed in "${seeds[@]}"; do
            gpu_id=$(($RANDOM % 2))
            echo "$prefix $params -gpu_id $gpu_id --seed $seed --log_suffix _seed$seed > log 2>&1 &"
            ((i += 1))
            if ((i % 16 == 0)); then
                echo "wait"
                echo "echo processed $i tasks"
            fi
        done
    done
    wait
    echo "echo processed $i tasks"
    echo "echo done!"
}
generator
