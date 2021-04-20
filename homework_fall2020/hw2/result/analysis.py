# %%
import os
import pandas as pd
import seaborn as sns
import glob
from collections import OrderedDict
from homework_fall2020.src.utils import parse_tb_data_to_csv, check_and_avg_seed, smooth_df, df_plot

root_dir = 'homework_fall2020/hw2'
result_dir = os.path.join(root_dir, 'result')
data_dir = os.path.join(root_dir, 'data_keep')

config_k2abbr = OrderedDict({
    "exp_name": "exp_name",
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

# %%
# q1
env = 'CartPole-v0'
exps_csv_path = os.path.join(result_dir, 'q1_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/q1/*')
# parse_tb_data_to_csv(exps, exps_csv_path, config_k2abbr)
df = pd.read_csv(exps_csv_path)

config_columns = ['exp_name', 'nnbase', 'bsize', 'rtg', 'dsa', 'lr']

df_show = check_and_avg_seed(df, config_columns, avg=True, debug=False)
df_show = smooth_df(df_show, config_columns, 'Iter', ['Train_AverageReturn', 'Eval_AverageReturn'], alpha=1.0)

if 'Train_AverageReturn' in df_show.columns:
    print('maximum Train_AverageReturn: ', df_show['Train_AverageReturn'].max())
if 'Eval_AverageReturn' in df_show.columns:
    print('maximum Eval_AverageReturn: ', df_show['Eval_AverageReturn'].max())

fields_show_dict = dict(x='Iter', y='Train_AverageReturn', hue="bsize", size='rtg', style="dsa", row='nnbase', col='lr')
df_plot(df_show, fields_show_dict, title=f'q1_pg_{env}', save_path=os.path.join(result_dir, 'imgs/q1.png'))

# %%
# q2
env = 'InvertedPendulum-v2'
exps_csv_path = os.path.join(result_dir, 'q2_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/q2/*')
parse_tb_data_to_csv(exps, exps_csv_path, config_k2abbr)
df = pd.read_csv(exps_csv_path)

config_columns = ['exp_name', 'nnbase', 'bsize', 'rtg', 'dsa', 'lr']
df_show = check_and_avg_seed(df, config_columns, avg=True, debug=False)
df_show = smooth_df(df_show, config_columns, 'Iter', ['Train_AverageReturn', 'Eval_AverageReturn'], alpha=1.0)
df_show = df_show[(df_show['bsize'] >= 5000)]

fields_show_dict = dict(x='Iter', y='Train_AverageReturn', hue="bsize", size='rtg', style="dsa", row='nnbase', col='lr')
df_plot(df_show, fields_show_dict, title=f'q2_pg_{env}', save_path=os.path.join(result_dir, 'imgs/q2.png'))

# %%
# q3
env = 'LunarLanderContinuous-v2'
exps_csv_path = os.path.join(result_dir, 'q3_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/q3/*')
parse_tb_data_to_csv(exps, exps_csv_path, config_k2abbr)
df = pd.read_csv(exps_csv_path)

config_columns = ['exp_name', 'nnbase', 'bsize', 'rtg', 'dsa', 'lr']
df_show = check_and_avg_seed(df, config_columns, avg=True, debug=False)
df_show = smooth_df(df_show, config_columns, 'Iter', ['Train_AverageReturn', 'Eval_AverageReturn'], alpha=1.0)

fields_show_dict = dict(x='Iter', y='Eval_AverageReturn', hue="bsize", size='rtg', style="dsa", row='nnbase', col='lr')
df_plot(df_show, fields_show_dict, title=f'q3_pg_{env}', save_path=os.path.join(result_dir, 'imgs/q3.png'))

# %%
# q4.1
env = 'HalfCheetah-v2'
exps_csv_path = os.path.join(result_dir, 'q4.1_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/q4.1/*')
parse_tb_data_to_csv(exps, exps_csv_path, config_k2abbr)
df = pd.read_csv(exps_csv_path)

config_columns = ['exp_name', 'nnbase', 'bsize', 'rtg', 'dsa', 'lr']
df_show = check_and_avg_seed(df, config_columns, avg=True, debug=False)
df_show = smooth_df(df_show, config_columns, 'Iter', ['Train_AverageReturn', 'Eval_AverageReturn'], alpha=1.0)

fields_show_dict = dict(x='Iter', y='Eval_AverageReturn', hue="bsize", size='rtg', style="dsa", row='nnbase', col='lr')
df_plot(df_show, fields_show_dict, title=f'q4.1_pg_{env}', save_path=os.path.join(result_dir, 'imgs/q4.1.png'))

# %%
# q4.2
env = 'HalfCheetah-v2'
exps_csv_path = os.path.join(result_dir, 'q4.2_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/q4.2/*')
parse_tb_data_to_csv(exps, exps_csv_path, config_k2abbr)
df = pd.read_csv(exps_csv_path)

config_columns = ['exp_name', 'nnbase', 'bsize', 'rtg', 'dsa', 'lr']
df_show = check_and_avg_seed(df, config_columns, avg=True, debug=False)
df_show = smooth_df(df_show, config_columns, 'Iter', ['Train_AverageReturn', 'Eval_AverageReturn'], alpha=1.0)

fields_show_dict = dict(x='Iter', y='Eval_AverageReturn', hue="bsize", size='rtg', style="dsa", row='nnbase', col='lr')
df_plot(df_show, fields_show_dict, title=f'q4.2_pg_{env}', save_path=os.path.join(result_dir, 'imgs/q4.2.png'))
