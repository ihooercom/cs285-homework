from collections import OrderedDict
import sys
import numpy as np


def merge_dicts(dict1, dict2):
    dict1.update(dict2)
    return dict1


def exps_gen(k_vlist_dict):
    ks = list(k_vlist_dict.keys())
    if len(ks) == 0:
        return [{}]
    k = ks[0]
    v_list = k_vlist_dict[k]
    del k_vlist_dict[k]
    exps = exps_gen(k_vlist_dict)

    new_exps = []
    for v in v_list:
        for exp in exps:
            new_exps.append(merge_dicts({k: v}, exp))
    return new_exps


def config_dict_to_cmd_str(config, script):
    args = []
    for k, v in config.items():
        assert v is not None
        if v is True:
            args.append('--' + k)
        elif v is False:
            continue
        else:
            args += ['--' + k, v]

    cmd = [sys.executable, script] + args + [f' >log 2>&1 &']
    cmd_str = ' '.join([str(x) for x in cmd])
    return cmd_str


config = OrderedDict({
    "exp_name": 'todo',
    "env_name": None,
    "ep_len": None,
    "reward_to_go": True,
    "nn_baseline": True,
    "dont_standardize_advantages": False,
    "discount": 1.0,
    "num_agent_train_steps_per_iter": 1,
    "n_iter": 200,
    "batch_size": 1000,
    "eval_batch_size": 400,
    "n_layers": 2,
    "size": 64,
    "learning_rate": 5e-3,
    "video_log_freq": -1,
    "scalar_log_freq": 1,
    "no_gpu": False,
    "which_gpu": 0,
    "save_params": False,
    "seed": 1
})


def config_to_log_str(config):
    k2abbr = OrderedDict({
        "env_name": 'env',
        "reward_to_go": 'rtg',
        "nn_baseline": 'nnbase',
        "dont_standardize_advantages": 'dsa',
        "discount": 'discount',
        "n_iter": "niter",
        "batch_size": 'bsize',
        "n_layers": 'nlayers',
        "size": 'size',
        "learning_rate": 'lr',
        "seed": 'seed'
    })

    log_str_list = []
    for k, v in k2abbr.items():
        log_str_list.append(v + ':' + str(config[k]))
    log_str = '#'.join(log_str_list)
    return '__' + log_str


def generate_run(script, config, **kwargs):
    config.update(kwargs)
    gpuid = np.random.randint(0, 2)
    config['which_gpu'] = gpuid
    config['log_suffix'] = config_to_log_str(config)

    cmd_str = config_dict_to_cmd_str(config, script)
    return cmd_str


def generate_batch_run(config_for_gridsearch):
    exps = exps_gen(config_for_gridsearch)
    filtered_exps = []

    for exp in exps:
        if exp['dont_standardize_advantages'] is False and exp['reward_to_go'] is False:
            continue
        filtered_exps.append(exp)
    exps = filtered_exps

    cmd_str_list = []
    for exp in exps:
        cmd_str = generate_run('homework_fall2020/hw2/cs285/scripts/run_hw2.py', config, **exp)
        cmd_str_list.append(cmd_str)
    cmd_str_list = sorted(list(set(cmd_str_list)))

    lst = ['export PYTHONPATH=homework_fall2020/hw2/:$PYTHONPATH']
    for i, cmd_str in enumerate(cmd_str_list):
        lst.append(cmd_str)
        if (i + 1) % 16 == 0:
            lst.append('wait')
            lst.append(f'echo "finished {i+1} exps"')
    lst.append('wait')
    lst.append(f'echo "finished {len(cmd_str_list)} exps"')
    return lst


if __name__ == "__main__":
    q1_config_for_gs = OrderedDict(env_name=['CartPole-v0'],
                                   reward_to_go=[True, False],
                                   nn_baseline=[False],
                                   dont_standardize_advantages=[True, False],
                                   discount=[1.0],
                                   n_iter=[100],
                                   batch_size=[1000, 5000],
                                   n_layers=[2],
                                   size=[64],
                                   learning_rate=[5e-3],
                                   seed=range(10, 110, 10))
    q2_config_for_gs = OrderedDict(env_name=['InvertedPendulum-v2'],
                                   ep_len=[1000],
                                   reward_to_go=[True],
                                   nn_baseline=[False],
                                   dont_standardize_advantages=[False],
                                   discount=[0.9],
                                   n_iter=[100],
                                   batch_size=[10000, 20000, 30000],
                                   n_layers=[2],
                                   size=[64],
                                   learning_rate=[5e-3, 1e-2, 2e-2],
                                   seed=range(1, 6))
    q3_config_for_gs = OrderedDict(env_name=['LunarLanderContinuous-v2'],
                                   ep_len=[1000],
                                   reward_to_go=[True],
                                   nn_baseline=[True],
                                   dont_standardize_advantages=[False],
                                   discount=[0.99],
                                   n_iter=[100],
                                   batch_size=[40000],
                                   n_layers=[2],
                                   size=[64],
                                   learning_rate=[5e-3],
                                   seed=range(1, 6))
    q4_1_config_for_gs = OrderedDict(env_name=['HalfCheetah-v2'],
                                     ep_len=[150],
                                     reward_to_go=[True],
                                     nn_baseline=[True],
                                     dont_standardize_advantages=[False],
                                     discount=[0.95],
                                     n_iter=[100],
                                     batch_size=[10000, 30000, 40000],
                                     n_layers=[2],
                                     size=[32],
                                     learning_rate=[5e-3, 1e-2, 2e-2],
                                     seed=range(1, 6))
    debug_config_for_gs = OrderedDict(env_name=['MaximizeFunc1DEnv-v0'],
                                      ep_len=[150],
                                      reward_to_go=[False],
                                      nn_baseline=[False],
                                      dont_standardize_advantages=[True],
                                      discount=[1.0],
                                      n_iter=[100],
                                      batch_size=[10000],
                                      n_layers=[2],
                                      size=[32],
                                      learning_rate=[5e-3],
                                      seed=[0])

    cmd_str_list = generate_batch_run(q4_1_config_for_gs)
    for cmd_str in cmd_str_list:
        print(cmd_str)