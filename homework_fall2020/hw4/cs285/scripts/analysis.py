# %%
import os
import pandas as pd
import seaborn as sns
import glob
from collections import OrderedDict
from homework_fall2020.src.utils import parse_tb_data_to_csv, check_and_avg_seed, smooth_df, df_plot

root_dir = 'homework_fall2020/hw4'
result_dir = os.path.join(root_dir, 'result')
data_dir = os.path.join(root_dir, 'data_keep')
os.system(f'mkdir -p {result_dir}/imgs')

# %%
#q1
from IPython.display import Image, display

env = 'cheetah'
exps = glob.glob(f'{data_dir}/hw4_q1_{env}*')
for exp in exps:
    print(exp)
    display(Image(filename=f'{exp}/itr_0_losses.png'))
    display(Image(filename=f'{exp}/itr_0_predictions.png'))

#%%
#q2
data_dir = os.path.join(root_dir, 'data')
exps_csv_path = os.path.join(result_dir, 'q2_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/hw4_q2_*')
parse_tb_data_to_csv(exps, exps_csv_path)
df = pd.read_csv(exps_csv_path)
df_show = df.copy()

print(df_show)

# %%
#q3
config_k2abbr = OrderedDict({
    "exp_name": "exp_name",
    "env_name": 'env',
    "n_iter": "niter",
    "ensemble_size": "ensemble_size",
    "mpc_horizon": "H",
    "mpc_num_action_sequences": "N",
    "add_sl_noise": "add_sl_noise",
    "batch_size": 'bsize',
    "train_batch_size": 'train_bsize',
    "eval_batch_size": 'eval_bsize',
    "n_layers": 'nlayers',
    "size": "size",
    "learning_rate": 'lr',
    "seed": 'seed',
})
data_dir = os.path.join(root_dir, 'data')
exps_csv_path = os.path.join(result_dir, 'q3_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/hw4_q3_*')
parse_tb_data_to_csv(exps, exps_csv_path, exp_config_k2abbr=config_k2abbr)
df = pd.read_csv(exps_csv_path)
df_show = df.copy()

if 'Train_AverageReturn' in df_show.columns:
    print('maximum Train_AverageReturn: ', df_show['Train_AverageReturn'].max())
if 'Eval_AverageReturn' in df_show.columns:
    print('maximum Eval_AverageReturn: ', df_show['Eval_AverageReturn'].max())

for name, g in df.groupby(['exp_name']):
    fields_show_dict = dict(x='Train_EnvstepsSoFar', y='Eval_AverageReturn', row='exp_name')
    df_plot(g, fields_show_dict, title='q3', save_path=os.path.join(result_dir, f'imgs/{g["exp_name"].iloc[0]}.png'))

# %%
#q4
config_k2abbr = OrderedDict({
    "exp_name": "exp_name",
    "env_name": 'env',
    "n_iter": "niter",
    "ensemble_size": "ensemble_size",
    "mpc_horizon": "H",
    "mpc_num_action_sequences": "N",
    "add_sl_noise": "add_sl_noise",
    "batch_size": 'bsize',
    "train_batch_size": 'train_bsize',
    "eval_batch_size": 'eval_bsize',
    "n_layers": 'nlayers',
    "size": "size",
    "learning_rate": 'lr',
    "seed": 'seed',
})
data_dir = os.path.join(root_dir, 'data')
exps_csv_path = os.path.join(result_dir, 'q4_scalars_exps.csv')
exps = glob.glob(f'{data_dir}/hw4_q4_*')
parse_tb_data_to_csv(exps, exps_csv_path, exp_config_k2abbr=config_k2abbr)
df = pd.read_csv(exps_csv_path)
df_show = df.copy()

if 'Train_AverageReturn' in df_show.columns:
    print('maximum Train_AverageReturn: ', df_show['Train_AverageReturn'].max())
if 'Eval_AverageReturn' in df_show.columns:
    print('maximum Eval_AverageReturn: ', df_show['Eval_AverageReturn'].max())

fields_show_dict = dict(x='Train_EnvstepsSoFar', y='Eval_AverageReturn', hue='exp_name')

condition = df_show['exp_name'].map(lambda x: x in ['q4_reacher_horizon30', 'q4_reacher_horizon5', 'q4_reacher_horizon15'])
df_plot(df_show[condition], fields_show_dict, title=f'q4_horizon', save_path=os.path.join(result_dir, 'imgs/q4_horizon.png'))

condition = df_show['exp_name'].map(lambda x: x in ['q4_reacher_numseq100', 'q4_reacher_numseq1000'])
df_plot(df_show[condition], fields_show_dict, title=f'q4_numseq', save_path=os.path.join(result_dir, 'imgs/q4_numseq.png'))

condition = df_show['exp_name'].map(lambda x: x in ['q4_reacher_ensemble1', 'q4_reacher_ensemble3', 'q4_reacher_ensemble5'])
df_plot(df_show[condition], fields_show_dict, title=f'q4_ensemble', save_path=os.path.join(result_dir, 'imgs/q4_ensemble.png'))

# %%
