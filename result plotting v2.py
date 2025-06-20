import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np

sns.set(style="whitegrid")


def load_data(folder):
    data = {}
    for _env in envs:
        _data = []
        for _metric in metrics:
            path = f'{csv_path}{folder}{_env}/{_metric}.csv'
            df = pd.read_csv(path)
            _data.append((df, _metric))
        data[_env] = _data
    return data

def rename(dataset):
    for df, metric in dataset:
        cols = [
            (f'noisy_net: true - {metric}', 'yes'),
            (f'noisy_net: true - {metric}__MIN', 'yes__MIN'),
            (f'noisy_net: true - {metric}__MAX', 'yes__MAX'),
            (f'noisy_net: false - {metric}', 'no'),
            (f'noisy_net: false - {metric}__MIN', 'no__MIN'),
            (f'noisy_net: false - {metric}__MAX', 'no__MAX'),
        ]
    
        for old, new in cols:
            if old in df.columns:
                df[new] = df.pop(old)
        df.dropna(subset=['yes__MIN', 'yes__MAX'], inplace=True)

def plot_metric_grid(dataset, title_prefix=""):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix} Metrics", fontsize=16)

    metric_map = {
        'score':       (0, 0, 'Score', 'Score', 'lower right'),
        'loss':        (0, 1, 'Loss', 'Loss', 'upper right'),
        'regret':      (1, 0, 'Regret', 'Regret', 'lower right'),
        'exploration_rate': (1, 1, 'Exploration Rate', 'Exploration Rate', 'upper right'),
    }

    for df, metric in dataset:
        row, col, title, ylabel, legend_loc = metric_map[metric]
        ax = axes[row, col]

        sns.lineplot(data=df, x='Step', y='yes', label=f'{metric.capitalize()} NoisyNet', ax=ax)
        sns.lineplot(data=nonoisy_data[0], x='Step', y='no', label=f'{metric.capitalize()} Non-NoisyNet', ax=ax)
        ax.fill_between(df['Step'], df['yes__MIN'], df['yes__MAX'], alpha=0.3)
        # ax.fill_between(df['Step'], df['no__MIN'], df['no__MAX'], alpha=0.3)

        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(title='Network', loc=legend_loc)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

####################################################################################################

if __name__ == '__main__':
    csv_path = "wandb_data/"

    metrics = ['score', 'regret', 'loss', 'exploration_rate']
    envs = ['reg', 'mnist', 'nn']

    # Load noisynet data
    data_p1o = load_data('p1o/')
    data_qyg = load_data('qyg/')

    for dataset in data_p1o.values():
        rename(dataset)
    for dataset in data_qyg.values():
        rename(dataset)

    combined_data = {}

    for env in envs:
        combined_env_data = []
        for (df1, metric), (df2, _) in zip(data_p1o[env], data_qyg[env]):
            df_merged = df1.merge(df2, on='Step', suffixes=('_p1o', '_qyg'))

            df_combined = pd.DataFrame()
            df_combined['Step'] = df_merged['Step']
            df_combined['yes'] = df_merged[['yes_p1o', 'yes_qyg']].mean(axis=1)
            df_combined['yes__MIN'] = df_merged[['yes__MIN_p1o', 'yes__MIN_qyg']].min(axis=1)
            df_combined['yes__MAX'] = df_merged[['yes__MAX_p1o', 'yes__MAX_qyg']].max(axis=1)
            df_combined['no'] = df_merged[['no_p1o', 'no_qyg']].mean(axis=1)
            df_combined['no__MIN'] = df_merged[['no__MIN_p1o', 'no__MIN_qyg']].min(axis=1)
            df_combined['no__MAX'] = df_merged[['no__MAX_p1o', 'no__MAX_qyg']].max(axis=1)

            combined_env_data.append((df_combined, metric))
        combined_data[env] = combined_env_data

    # combined_data['mnist'], combined_data['reg'], combined_data['nn']
    # list of (df, metric)
    
    
    # Load nonoisy data
    nonoisy_data = []
    for _metric in metrics:
        path = f'{csv_path}no-noisy/{_metric}.csv'
        df = pd.read_csv(path)
        # print(df.head(2))
        nonoisy_data.append((df, _metric))

    # print(nonoisy_data[0][0]['conbandit_combi1_0__5820__2025-06-17_16-53 - score'])

    plot_metric_grid(combined_data['reg'], title_prefix="ContextualBandit-v2")
    plot_metric_grid(combined_data['mnist'], title_prefix="MNISTBandit-v0")
    plot_metric_grid(combined_data['nn'], title_prefix="NNBandit-v0")