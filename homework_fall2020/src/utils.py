import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import glob
from tqdm import tqdm
import numpy as np
import json
from collections import OrderedDict
from copy import deepcopy

sns.set(style='whitegrid')
pd.set_option('display.precision', 2)


def event_scalars_to_csv(event_file, csv_file):
    event_data = event_accumulator.EventAccumulator(event_file)
    event_data.Reload()
    keys = event_data.scalars.Keys()
    df = pd.DataFrame(columns=keys)
    for key in keys:
        df[key] = pd.DataFrame(event_data.Scalars(key)).value
    df.to_csv(csv_file, index=False)


def load_config(config_path, config_k2abbr, is_json=False):
    if is_json:
        assert os.path.exists(config_path), print(config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert len(config_k2abbr) > 0

        new_config = OrderedDict({})
        for k, v in config_k2abbr.items():
            new_config[v] = config[k]
        config = new_config
    else:
        exp_config_str = config_path.split('__')[1]
        config = dict([x.split(':') for x in exp_config_str.split('#')])
        new_config = {}
        for k, v in config.items():
            if v.replace('.', '').isdigit():
                new_config[k] = eval(v)
            else:
                new_config[k] = v
        config = new_config
    return config


def parse_tb_data_to_csv(exps_dirs, exps_csv_path, exp_config_k2abbr=None):
    """parse exps dirs which include tensorboard events to csv files

    Args:
        exps_dirs ([list]): list of directory including event file
        exps_csv_path ([type]): dest csv file path
        exp_config_k2abbr ([type]): exp config dict, keys are important column names of experiments, 
            values are the abbreviations of the keys
    """

    lst = []
    print('exps num: ', len(exps_dirs))
    for exp_dir in tqdm(exps_dirs, total=len(exps_dirs)):
        event_path = glob.glob(os.path.join(exp_dir, 'events.out.*'))
        assert len(event_path) == 1, print(event_path)
        event_path = event_path[0]
        csv_path = os.path.join(exp_dir, 'scalars.' + str(os.path.basename(event_path)) + '.csv')
        event_scalars_to_csv(event_path, csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(e)
            # os.system(f'rm -r {os.path.dirname(csv_path)}')
            print('\n' + csv_path + '\n')
            continue
        if exp_config_k2abbr:
            config = load_config(os.path.join(exp_dir, 'config.json'), exp_config_k2abbr, is_json=True)
            df = pd.concat([pd.DataFrame([config.values()], index=df.index, columns=config.keys()), df], axis=1)
        df['Iter'] = np.arange(df.index.shape[0])
        lst.append(df)
    df = pd.concat(lst)
    df.to_csv(exps_csv_path)


def check_and_avg_seed(df, config_columns, avg=False, debug=False):
    if avg:
        df_show = df.groupby(config_columns + ['Iter'], as_index=False).agg(np.mean)
    else:
        df_show = df.copy()
        seed_num = max([g.shape[0] for _, g in df_show.groupby(config_columns + ['Iter'])])
        for name, g in df_show.groupby(config_columns + ['Iter']):
            try:
                assert g.shape[0] == seed_num
            except:
                if debug:
                    debug_columns = list(set(g.columns.tolist()) & set(['exp_name', 'env', 'seed', 'Iter']))
                    print(name, '\n', g[debug_columns])
                else:
                    pass
    return df_show


def smooth_df(df, config_columns, xname, ynames, alpha=0.4):
    if 'seed' in df.columns and 'seed' not in config_columns:
        config_columns.append('seed')
    ynames = list(set(df.columns.tolist()) & set(ynames))
    lst = []
    for name, g in df.groupby(config_columns):
        d = dict(zip(config_columns, name))
        d[xname] = g[xname]
        for yname in ynames:
            d[yname] = g[yname].ewm(alpha=alpha).mean()
        for c in df.columns:
            if c not in d:
                d[c] = g[c]
        gdf = pd.DataFrame(d)
        lst.append(gdf)
    smoothed_df_show = pd.concat(lst, axis=0)
    return smoothed_df_show


def df_plot(df_show, fields_show_dict, title=None, save_path=None):
    fields_show_dict = deepcopy(fields_show_dict)
    xname = fields_show_dict.get('x')
    yname = fields_show_dict.pop('y')
    if 'hue' in fields_show_dict:
        c = fields_show_dict['hue']
        hue_num = df_show[c].unique().shape[0]
        palette = sns.color_palette("hls", hue_num)
    else:
        palette = None
    if 'size' in fields_show_dict:
        sizes = (1.0, 2.5)
    else:
        sizes = None
    g = sns.relplot(data=df_show, **fields_show_dict, y=yname, kind='line', ci='sd', sizes=sizes, palette=palette)

    g.set_axis_labels(xname, yname)
    if title:
        g.fig.suptitle(title)
    plt.subplots_adjust(top=0.85)
    if save_path:
        g.savefig(save_path)