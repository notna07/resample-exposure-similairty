# Description: Script for plotting the results
# Author: Anton D. Lautrup
# Date: 20-05-2025

from typing import List

import seaborn as sns
import matplotlib.pyplot as plt

rcp = {'font.size': 10, 'font.family': 'sans', "mathtext.fontset": "dejavuserif"}
plt.rcParams.update(**rcp)

def plot_cross_validation_results(df_name: str, exp_name: str, k_values: List[int], avg_scores_lst: List[float], std_scores_lst: List[float], var_name: str) -> None:
    """Plot the average and standard deviation of cross-validation scores."""
    plt.errorbar(k_values, avg_scores_lst, yerr=std_scores_lst, fmt='o', capsize=5)
    plt.title(f"{exp_name} k-Search Cross-Validation for '{df_name}' Dataset")
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel(var_name)
    plt.xticks(k_values)
    plt.minorticks_on()

    plt.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.5)
    plt.savefig(f'plots/{df_name}_{exp_name}_cross_val.png')
    plt.close()
    pass

def plot_distributions(df):
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 5))
    axes = axes.flatten()

    # Loop through each column and plot the distribution
    for i, column in enumerate(df.columns):
        sns.histplot(df[column], bins=30, ax=axes[i], stat='probability')
        axes[i].set_title(f'{column}')
        axes[i].set_xlabel("")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set only the first item in a row to have a y-label
        if i % 5 == 0:
            axes[i].set_ylabel('Probability')
        else:
            axes[i].set_ylabel('')

        # shorten the tick labels
        axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))

    plt.tight_layout()
    plt.show()