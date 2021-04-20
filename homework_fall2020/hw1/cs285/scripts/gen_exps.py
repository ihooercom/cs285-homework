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

    k = k.rstrip('_list')
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
    "exp_name": None,
    "env_name": "Ant-v2",
    "do_dagger": False,
    "expert_policy_file": None,
    "expert_data": None,
    "ep_len": 1000,
    "num_agent_train_steps_per_iter": 1000,
    "n_iter": 1,
    "batch_size": 1000,
    "eval_batch_size": 1000,
    "train_batch_size": 100,
    "n_layers": 2,
    "size": 64,
    "learning_rate": 5e-3,
    "video_log_freq": 5,
    "scalar_log_freq": 1,
    "no_gpu": False,
    "which_gpu": 0,
    "max_replay_buffer_size": int(1e6),
    "save_params": False,
    "seed": 1
})


def config_to_log_str(config):
    k2abbr = OrderedDict({
        "exp_name": 'exp_name',
        "num_agent_train_steps_per_iter": "update_steps",
        "n_iter": "niter",
        "batch_size": 'bsize',
        "train_batch_size": 'train_bsize',
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
    env_name = config['env_name']
    env_name_abbr = env_name.split('-')[0]
    config['expert_policy_file'] = f'homework_fall2020/hw1/cs285/policies/experts/{env_name_abbr}.pkl'
    config['expert_data'] = f'homework_fall2020/hw1/cs285/expert_data/expert_data_{env_name}.pkl'

    if config['do_dagger']:
        exp_name = f'dagger_{env_name_abbr}'
        assert config['n_iter'] > 1
    else:
        exp_name = f'bc_{env_name_abbr}'
        config['n_iter'] = 1
    config['exp_name'] = exp_name
    config['train_batch_size'] = min(config['train_batch_size'], config['batch_size'])
    config['eval_batch_size'] = config['batch_size']
    gpuid = np.random.randint(0, 2)
    config['which_gpu'] = gpuid
    config['log_suffix'] = config_to_log_str(config)

    cmd_str = config_dict_to_cmd_str(config, script)
    return cmd_str


def generate_batch_run():
    env_name_list = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Walker2d-v2']
    do_dagger_list = [True]
    num_agent_train_steps_per_iter_list = [256, 1000, 1500]
    n_iter_list = [15]
    batch_size_list = [1000, 3000]
    train_batch_size_list = [100, 256]
    n_layers_list = [2]
    size_list = [64]
    learning_rate_list = [1e-3, 5e-3, 1e-2]
    video_log_freq_list = [-1]
    seed_list = [10, 20, 30, 40, 50]
    cfg = locals()

    exps = exps_gen(cfg)
    cmd_str_list = []
    for exp in exps:
        cmd_str = generate_run('homework_fall2020/hw1/cs285/scripts/run_hw1.py', config, **exp)
        cmd_str_list.append(cmd_str)
    cmd_str_list = sorted(list(set(cmd_str_list)))

    lst = []
    for i, cmd_str in enumerate(cmd_str_list):
        lst.append(cmd_str)
        if (i + 1) % 16 == 0:
            lst.append('wait')
            lst.append(f'echo "finished {i+1} exps"')
    lst.append('wait')
    lst.append(f'echo "finished {len(cmd_str_list)} exps"')
    return lst


if __name__ == "__main__":
    cmd_str_list = generate_batch_run()
    for cmd_str in cmd_str_list:
        print(cmd_str)