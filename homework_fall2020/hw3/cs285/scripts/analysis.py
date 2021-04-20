# %%
import os
import pandas as pd
import seaborn as sns
import glob
from collections import OrderedDict
from homework_fall2020.src.utils import parse_tb_data_to_csv, check_and_avg_seed, smooth_df, df_plot

root_dir = 'homework_fall2020/hw3'
result_dir = os.path.join(root_dir, 'result')
data_dir = os.path.join(root_dir, 'data_keep')

#%%
#q1
env = 'MsPacman-v0'
exps_csv_path = os.path.join(result_dir, 'q1_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/hw3_q1_{env}*')
parse_tb_data_to_csv(exps, exps_csv_path)
df = pd.read_csv(exps_csv_path)
df_show = df

if 'Train_AverageReturn' in df_show.columns:
    print('maximum Train_AverageReturn: ', df_show['Train_AverageReturn'].max())
if 'Eval_AverageReturn' in df_show.columns:
    print('maximum Eval_AverageReturn: ', df_show['Eval_AverageReturn'].max())

fields_show_dict = dict(x='Iter', y='Train_AverageReturn')
df_plot(df_show, fields_show_dict, title=f'q1_dqn_{env}', save_path=os.path.join(result_dir, 'imgs/q1.png'))

#%%
#q2

config_k2abbr = OrderedDict({
    "exp_name": "exp_name",
    "env_name": 'env',
    "double_q": "double_q",
    "seed": 'seed',
})

env = 'LunarLander-v3'
exps_csv_path = os.path.join(result_dir, 'q2_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/hw3_q2_*')
parse_tb_data_to_csv(exps, exps_csv_path, exp_config_k2abbr=config_k2abbr)
df = pd.read_csv(exps_csv_path)

config_columns = ['exp_name', 'env', 'double_q']

df_show = check_and_avg_seed(df, config_columns, avg=False, debug=False)
df_show = smooth_df(df_show, config_columns, 'Iter', ['Train_AverageReturn', 'Eval_AverageReturn'], alpha=1.0)

if 'Train_AverageReturn' in df_show.columns:
    print('maximum Train_AverageReturn: ', df_show['Train_AverageReturn'].max())
if 'Eval_AverageReturn' in df_show.columns:
    print('maximum Eval_AverageReturn: ', df_show['Eval_AverageReturn'].max())

fields_show_dict = dict(x='Train_EnvstepsSoFar', y='Train_AverageReturn', hue='double_q')
df_plot(df_show, fields_show_dict, title=f'q2_dqn_{env}', save_path=os.path.join(result_dir, 'imgs/q2.png'))

# %%
# q4
config_k2abbr = OrderedDict({
    "exp_name": "exp_name",
    "env_name": 'env',
    "discount": 'discount',
    "n_iter": "niter",
    'num_target_updates': 'ntu',
    'num_grad_steps_per_target_update': 'ngsptu',
    "batch_size": 'bsize',
    "n_layers": 'nlayers',
    "size": 'size',
    "learning_rate": 'lr',
    "seed": 'seed',
})
exps_csv_path = os.path.join(result_dir, 'q4_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/hw3_q4*')
parse_tb_data_to_csv(exps, exps_csv_path, config_k2abbr)
df = pd.read_csv(exps_csv_path)

config_columns = ['exp_name', 'env', 'niter', 'bsize', 'ntu', 'ngsptu', 'lr']

df_show = check_and_avg_seed(df, config_columns, avg=False, debug=False)
df_show = smooth_df(df_show, config_columns, 'Iter', ['Train_AverageReturn', 'Eval_AverageReturn'], alpha=1.0)

print('maximum Train_AverageReturn: ', df_show['Train_AverageReturn'].max())
print('maximum Eval_AverageReturn: ', df_show['Eval_AverageReturn'].max())

fields_show_dict = dict(x='Iter',
                        y='Eval_AverageReturn',
                        hue="ntu",
                        size='bsize',
                        style="ngsptu",
                        row='lr',
                        col='niter')
df_plot(df_show,
        fields_show_dict,
        title='q4_actor-critic_CartPole-v0',
        save_path=os.path.join(result_dir, 'imgs/q4.png'))
# %%
# q5
config_k2abbr = OrderedDict({
    "exp_name": "exp_name",
    "env_name": 'env',
    "discount": 'discount',
    "n_iter": "niter",
    'num_target_updates': 'ntu',
    'num_grad_steps_per_target_update': 'ngsptu',
    "batch_size": 'bsize',
    "n_layers": 'nlayers',
    "size": 'size',
    "learning_rate": 'lr',
    "seed": 'seed',
})
exps_csv_path = os.path.join(result_dir, 'q5_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/hw3_q5*')
parse_tb_data_to_csv(exps, exps_csv_path, config_k2abbr)
df = pd.read_csv(exps_csv_path)

config_columns = ['exp_name', 'env', 'niter', 'bsize', 'ntu', 'ngsptu', 'lr']
df_show = check_and_avg_seed(df, config_columns, avg=False, debug=False)
df_show = smooth_df(df_show, config_columns, 'Iter', ['Train_AverageReturn', 'Eval_AverageReturn'], alpha=1.0)

print('maximum Train_AverageReturn: ', df_show['Train_AverageReturn'].max())
print('maximum Eval_AverageReturn: ', df_show['Eval_AverageReturn'].max())

fields_show_dict = dict(x='Iter', y='Eval_AverageReturn', hue="ntu", size='lr', style="ngsptu", row='niter', col='env')

df_plot(df_show[df_show['env'] == 'HalfCheetah-v2'],
        fields_show_dict,
        title='q5_actor-critic',
        save_path=os.path.join(result_dir, 'imgs/q5_halfcheetah-v2.png'))

df_plot(df_show[df_show['env'] == 'InvertedPendulum-v2'],
        fields_show_dict,
        title='q5_actor-critic',
        save_path=os.path.join(result_dir, 'imgs/q5_InvertedPendulum-v2.png'))
# %%
