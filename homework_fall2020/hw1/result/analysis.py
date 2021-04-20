# %%
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import glob
from tqdm import tqdm
import numpy as np

pd.set_option('display.precision', 2)
sns.set(style="darkgrid")


def event_scalars_to_csv(event_file, csv_file):
    event_data = event_accumulator.EventAccumulator(event_file)
    event_data.Reload()
    keys = event_data.scalars.Keys()
    df = pd.DataFrame(columns=keys)
    for key in keys:
        df[key] = pd.DataFrame(event_data.Scalars(key)).value
    df.to_csv(csv_file, index=False)


def parse_exp_str(exp):
    config = dict([x.split(':') for x in exp.split('#')])
    new_config = {}
    for k, v in config.items():
        if v.replace('.', '').isdigit():
            new_config[k] = eval(v)
        else:
            new_config[k] = v
    return new_config


def parse_tb_data_to_csv(exps_data_dir, exps_csv_path):
    lst = []
    exps_data_dirs = os.listdir(exps_data_dir)
    print('exps num: ', len(exps_data_dirs))
    for exp in tqdm(exps_data_dirs, total=len(exps_data_dirs)):
        event_path = glob.glob(os.path.join(exps_data_dir, exp, 'events.out.*'))
        assert len(event_path) == 1, print(event_path)
        event_path = event_path[0]
        csv_path = os.path.join(exps_data_dir, exp, 'scalars.' + str(os.path.basename(event_path)) + '.csv')
        event_scalars_to_csv(event_path, csv_path)

        df = pd.read_csv(csv_path)
        exp_config_str = exp.split('__')[1]
        config = parse_exp_str(exp_config_str)
        df = pd.concat([pd.DataFrame([config.values()], index=df.index, columns=config.keys()), df], axis=1)
        df['Iter'] = np.arange(df.index.shape[0])
        lst.append(df)
    df = pd.concat(lst)
    df.to_csv(exps_csv_path)


def df_plot(df_show, fields_show_dict, save_path=None):
    # df_show = df_show[(df_show['update_steps'] == 1000)]
    yname = fields_show_dict.pop('y')
    g = sns.relplot(data=df_show,
                    **fields_show_dict,
                    y=yname,
                    kind='line',
                    ci='sd',
                    row_order=df_show['exp_name'].unique().tolist(),
                    palette=sns.color_palette("hls", 3))

    def add_hline(exp_name, **kwargs):
        v = df_show[df_show['exp_name'] == exp_name.iloc[0]]["Initial_DataCollection_AverageReturn"].unique()
        assert v.shape[0] == 1
        v = v[0]
        plt.axhline(y=v, **kwargs)
        # ax = plt.gca()
        # ax.xaxis.label.set_visible(False)

    # # sns_plot.map(plt.axhline, y=0.0, ls='--', c='red')
    g.map(add_hline, 'exp_name', ls='--', c='black')
    g.set_axis_labels("Iter", yname)
    if save_path:
        g.savefig(save_path)


# %%
root_dir = 'homework_fall2020/hw1'
result_dir = os.path.join(root_dir, 'result')
exps_csv_path = os.path.join(result_dir, 'scalars_exps.csv')
# parse_tb_data_to_csv(os.path.join(root_dir, 'data_keep'), exps_csv_path)
df = pd.read_csv(exps_csv_path)
# %%
# df2 = df.groupby(config_columns + ['Iter'], as_index=False).agg(np.mean)
# n1 = df['exp_name'].nunique()
# n2 = df['update_steps'].nunique()
# plt.figure(figsize=(16, 16))
# columns_show = ['Iter', 'seed', 'Eval_AverageReturn']
# for i, (name, group) in enumerate(df):
#     df_show = group.loc[:, columns_show]
#     plt.subplot(4, 2, i + 1)
#     sns.lineplot(data=df_show, x='Iter', y='Eval_AverageReturn',
#                  hue='seed').set_title(name)
#     if i + 1 == 8:
#         break

#%% [markdown]
# > num_agent_train_steps_per_iter_list = [256, 1000, 1500]
#
# batch_size_list = [1000, 3000]
#
# train_batch_size_list = [100, 256]
#
# learning_rate_list = [1e-3, 5e-3, 1e-2]

# %%
config_columns = ['exp_name', 'update_steps', 'niter', 'bsize', 'train_bsize', 'nlayers', 'size', 'lr']
show_seeds = False
if show_seeds:
    df_show = df.copy()
    for name, g in df_show.groupby(config_columns + ['Iter']):
        assert g.shape[0] == 5, print(g)
else:
    df_show = df.groupby(config_columns + ['Iter'], as_index=False).agg(np.mean)
# print(df_show[["exp_name", "Initial_DataCollection_AverageReturn"]].unique())
print(df_show.groupby('exp_name')["Initial_DataCollection_AverageReturn"].unique())
print(df_show.columns)

#%%
fields_show_dict = dict(x='Iter',
                        y='Train_AverageReturn',
                        hue="lr",
                        size='update_steps',
                        style="train_bsize",
                        row="exp_name",
                        col="bsize")
df_plot(df_show, fields_show_dict, save_path=os.path.join(result_dir, 'imgs/train_img1.png'))

# %%
fields_show_dict = dict(x='Iter',
                        y='Train_AverageReturn',
                        hue="lr",
                        size='update_steps',
                        style="bsize",
                        row="exp_name",
                        col="train_bsize")
df_plot(df_show, fields_show_dict, save_path=os.path.join(result_dir, 'imgs/train_img2.png'))

# %%
fields_show_dict = dict(x='Iter',
                        y='Eval_AverageReturn',
                        hue="lr",
                        size='update_steps',
                        style="train_bsize",
                        row="exp_name",
                        col="bsize")
df_plot(df_show, fields_show_dict, save_path=os.path.join(result_dir, 'imgs/eval_img1.png'))

# %%
fields_show_dict = dict(x='Iter',
                        y='Eval_AverageReturn',
                        hue="lr",
                        size='update_steps',
                        style="bsize",
                        row="exp_name",
                        col="train_bsize")
df_plot(df_show, fields_show_dict, save_path=os.path.join(result_dir, 'imgs/eval_img2.png'))

# %%

fields_show_dict = dict(x='Iter',
                        y='Eval_AverageReturn',
                        hue="lr",
                        size='update_steps',
                        style="bsize",
                        row="exp_name",
                        col="train_bsize")
df_plot(df_show, fields_show_dict, save_path=os.path.join(result_dir, 'imgs/eval_img2.png'))
# %%
