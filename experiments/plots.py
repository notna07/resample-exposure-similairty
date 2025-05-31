# Description: Script for plotting the results
# Author: Anton D. Lautrup
# Date: 20-05-2025

from pandas import DataFrame
from typing import List, Dict

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


import gower
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from rex_score.resample_exposure import ResampleExposure

def plot_multi_dataset_contour_comparison(datasets_dict, source_point_pca=[0, 0]):  
    """ 
    Plot contour maps of Euclidean distance, Gower distance, and Resample Exposure 
    in PCA space for multiple datasets.

    Args:
        datasets_dict (dict): Dictionary where keys are dataset names (str) 
                              and values are pandas DataFrames.
        source_point_pca (list): Coordinates of the source point in 2D PCA space.
    """
    num_datasets = len(datasets_dict)
    if num_datasets == 0:
        print("No datasets provided.")
        return

    fig, axes = plt.subplots(nrows=num_datasets, ncols=3, 
                             figsize=(9, 3 * num_datasets), # Adjust figsize as needed
                             sharex=True, sharey=True)

    # Ensure axes is always 2D for consistent indexing, even if num_datasets=1
    if num_datasets == 1:
        axes = axes.reshape(3, 1)

    dataset_names = list(datasets_dict.keys())
    metric_names = ["Euclidean Distance", "Gower Distance", "Resample Exposure"]

    for row_idx, dataset_name in enumerate(dataset_names):
        df = datasets_dict[dataset_name]
        
        pca = PCA(n_components=2)
        X = df.select_dtypes(include=[np.number])

        if X.shape[1] < 2:
            print(f"Dataset '{dataset_name}' has fewer than 2 numeric features. Skipping PCA and plotting.")
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, f"{dataset_name}\n(Skipped)\nToo few numeric features", 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.axis('off')
                if row_idx == 0: # Set metric name as title for skipped plots as well
                    ax.set_title(metric_names[col_idx], fontsize=11)
            continue
        
        X_std = (X - X.mean()) / X.std()
        X_std.fillna(0, inplace=True) 
        X_pca = pca.fit_transform(X_std)

        x_lspace = np.linspace(-3.5, 3.5, 50)
        y_lspace = np.linspace(-3.5, 3.5, 50)
        X_grid_pca, Y_grid_pca = np.meshgrid(x_lspace, y_lspace)
        grid_points_pca = np.c_[X_grid_pca.ravel(), Y_grid_pca.ravel()]

        grid_points_original_space = pca.inverse_transform(grid_points_pca)
        grid_df_original_space = pd.DataFrame(grid_points_original_space, columns=X.columns)
        
        source_point_original_space_array = pca.inverse_transform(np.array(source_point_pca).reshape(1, -1))
        source_point_df_original_space = pd.DataFrame(source_point_original_space_array, columns=X.columns)

        distances_euclidean = np.linalg.norm(grid_points_original_space - source_point_original_space_array, axis=1)
        distance_grid_euclidean = distances_euclidean.reshape(X_grid_pca.shape)

        gower_distances_val = gower.gower_matrix(source_point_df_original_space, grid_df_original_space)
        gower_grid_val = gower_distances_val.reshape(X_grid_pca.shape)
        
        rex = ResampleExposure(X_std)
        rex.memorised_distribution = grid_df_original_space
        exposure_val_matrix = rex.resample_exposure_matrix(source_point_df_original_space, True)
        exposure_val = 1 - exposure_val_matrix.ravel()
        exposure_grid_val = exposure_val.reshape(X_grid_pca.shape)

        plot_data_grids = [distance_grid_euclidean, gower_grid_val, exposure_grid_val]

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_aspect('equal')
            
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], 
                            hue=df.get('class'), 
                            palette='tab20c', s=50, ax=ax, legend=False)
            
            ax.scatter(source_point_pca[0], source_point_pca[1], color='red', s=200, marker='x')

            data_grid = plot_data_grids[col_idx]
            contour = ax.contour(X_grid_pca, Y_grid_pca, data_grid, levels=10, cmap='viridis', alpha=0.4)
            ax.clabel(contour, contour.levels, fontsize=7)
            ax.axis([-3.5, 3.5, -3.5, 3.5])

            if row_idx == 0: # Top row: Metric name as title for the column
                ax.set_title(metric_names[col_idx], fontsize=14)
            
            if col_idx == 0: # First column: Dataset name as Y-axis label
                ax.set_ylabel(f"{dataset_name}\nPC2", fontsize=11)
            
            if row_idx == num_datasets - 1: # Bottom row: PC1 as X-axis label
                ax.set_xlabel('PC1', fontsize=11)
            
            # Remove y-axis labels for plots not in the first column
            if col_idx > 0:
                ax.set_ylabel('')
            
            # Remove x-axis labels for plots not in the last row
            if row_idx < num_datasets - 1:
                ax.set_xlabel('')
            
            # Remove tick labels for inner plots to save space, if desired
            if col_idx > 0:
                ax.tick_params(axis='y', labelleft=False)
            if row_idx < num_datasets - 1:
                ax.tick_params(axis='x', labelbottom=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
    plt.savefig("plots/contour_comparison.pdf", bbox_inches='tight')
    pass

def plot_multi_dataset_heatmap_comparison(datasets_dict: dict):
    """
    Plot heatmaps of Euclidean distance, Gower distance, and Resample Exposure 
    in PCA space for multiple datasets. For each grid point in PCA space,
    the metrics are calculated against the nearest point in the actual dataset.
    
    Args:
        datasets_dict (dict): Dictionary where keys are dataset names (str) 
                              and values are pandas DataFrames.
    """
    num_datasets = len(datasets_dict)
    if num_datasets == 0:
        print("No datasets provided.")
        return

    metric_names = ["Euclidean Distance", "Gower Distance", "Resample Exposure"]

    fig, axes = plt.subplots(nrows=num_datasets, ncols=3,
                             figsize=(9, 3 * num_datasets), # Adjusted figsize
                             sharex=True, sharey=True)

    if num_datasets == 1: # Ensure axes is 2D for consistent indexing
        axes = axes.reshape(1, 3)

    dataset_names = list(datasets_dict.keys())

    for row_idx, dataset_name in enumerate(dataset_names):
        df = datasets_dict[dataset_name]
        
        pca = PCA(n_components=2)
        X = df.select_dtypes(include=[np.number])

        if X.shape[1] < 2:
            print(f"Dataset '{dataset_name}' has fewer than 2 numeric features. Skipping PCA and plotting.")
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, f"{dataset_name}\n(Skipped)\nToo few numeric features", 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.axis('off')
                if row_idx == 0:
                    ax.set_title(metric_names[col_idx], fontsize=11)
            continue
        
        X_std = (X - X.mean()) / X.std()
        X_std.fillna(0, inplace=True) # Handle potential NaNs from std=0 or original NaNs
        X_pca = pca.fit_transform(X_std)

        x_lspace = np.linspace(-3.5, 3.5, 30)
        y_lspace = np.linspace(-3.5, 3.5, 30)
        X_grid_pca, Y_grid_pca = np.meshgrid(x_lspace, y_lspace)
        grid_points_pca = np.c_[X_grid_pca.ravel(), Y_grid_pca.ravel()]
        grid_points_original_space = pca.inverse_transform(grid_points_pca)

        rex = ResampleExposure(X_std) # Initialize with the dataset's standardized numeric features
        
        exposure_scores = np.zeros(grid_points_original_space.shape[0])
        euclidean_distances_to_nearest = np.zeros(grid_points_original_space.shape[0])
        gower_distances_to_nearest = np.zeros(grid_points_original_space.shape[0])

        # X_std is already a DataFrame if X is.
        # Ensure columns are available for creating point_df_orig_space
        feature_columns = X.columns 

        for i, point_orig_space_row in enumerate(grid_points_original_space):
            point_df_orig_space = pd.DataFrame([point_orig_space_row], columns=feature_columns)
            
            # Calculate exposure of the grid point w.r.t. the dataset X_std
            # Assuming resample_exposure_matrix calculates similarity, so 1 - similarity = exposure
            # And memorized_distribution_is_grid=False means rex.memorised_distribution (X_std) is the reference dataset
            exposure_scores[i] = np.min(1 - rex.resample_exposure_matrix(point_df_orig_space, False)) 
            
            # Euclidean distance from grid point to nearest point in X_std
            euclidean_distances_to_nearest[i] = np.min(np.linalg.norm(X_std.values - point_df_orig_space.values, axis=1))
            
            # Gower distance from grid point to nearest point in X_std
            gower_distances_to_nearest[i] = np.min(gower.gower_matrix(point_df_orig_space, X_std))

        exposure_grid = exposure_scores.reshape(X_grid_pca.shape)
        distance_grid = euclidean_distances_to_nearest.reshape(X_grid_pca.shape)
        gower_grid = gower_distances_to_nearest.reshape(X_grid_pca.shape)

        plot_data_grids = [distance_grid, gower_grid, exposure_grid]

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_aspect('equal')
            
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], 
                            hue=df.get('class'), # Use .get for safety if 'class' column might be missing
                            palette='tab20c', s=50, ax=ax, legend=False)
            
            data_grid_to_plot = plot_data_grids[col_idx]
            ax.imshow(data_grid_to_plot, extent=(-3.5, 3.5, -3.5, 3.5), 
                      origin='lower', cmap='magma', alpha=0.9, interpolation='bicubic')
            
            ax.axis([-3.5, 3.5, -3.5, 3.5])

            if row_idx == 0:
                ax.set_title(metric_names[col_idx], fontsize=14)
            
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_name}\nPC2", fontsize=11)
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)
            
            if row_idx == num_datasets - 1:
                ax.set_xlabel('PC1', fontsize=11)
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

            if ax.get_legend() is not None: # Remove scatterplot legend if it appears
                ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
    plt.savefig("plots/heatmap_comparison.pdf", bbox_inches='tight')
    pass

def plot_clustering_subplots(df: DataFrame, metrics: Dict[str, callable], n_clusters: int = 4, seed: int = 42):
    
    df_x, df_y = df.drop(columns=['class']), df['class']

    fig, axes = plt.subplots(1, 6, figsize=(10,2), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (metric_name, metric_func) in enumerate(metrics.items()):
        labels, medoid_indices = metric_func(df_x, n_clusters=n_clusters, seed=seed)

        # plot the data in principal component space
        pca = PCA(n_components=2)
        X_std = (df_x - df_x.mean()) / df_x.std()
        pca_result = pca.fit_transform(X_std)

        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df_y, palette='tab20c', s=20, ax=axes[i])
        axes[i].set_title(metric_name, fontsize=10)
        axes[i].get_legend().remove()

        ## draw lines between the points and the medoids
        cols_palette = sns.color_palette('tab10', n_colors=n_clusters)
        for j, medoid in enumerate(medoid_indices):
            medoid_pca = pca_result[medoid].reshape(1, -1)

            for k in range(len(pca_result[labels == j])):
                axes[i].plot([pca_result[labels == j][k, 0], medoid_pca[0, 0]], 
                             [pca_result[labels == j][k, 1], medoid_pca[0, 1]], 
                             color=cols_palette[j], linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('plots/clustering_subplots.pdf', dpi=300)
    plt.show()
    pass

def plot_clustering_scores_artificial_data(res_dataframe: pd.DataFrame) -> None:

    # Set up the figure with 2 rows (for ARI and NMI) and 3 columns (for Large/Medium/Small)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey='row', sharex=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.15)  # Add spacing between triplets

    # Define consistent hue order
    hue_order = ['Reals', 'Cats', 'Balanced']
    features_distribution_dict = {'Reals':'Majority Quantitative', 'Cats':'Majority Nominal', 'Balanced':'Balanced'}

    metric_order = ['REX', 'GOW', 'GEM', 'HEOM', 'L2', 'L2_OHE']  # Fixed metric order

    Sizes = {'Large':"16", 'Medium':"8", 'Small':'4'}

    # Loop through measures and dataset sizes
    for i, measure in enumerate(['ARI', 'NMI']):
        for j, dim in enumerate(['Large', 'Medium', 'Small']):
            ax = axes[i, j]
            data_subset = res_dataframe[
                (res_dataframe['measure'] == measure) & 
                (res_dataframe['dims'] == dim)
            ]

            # Plot stripplot and pointplot with consistent metric order
            sns.stripplot(
                data=data_subset,
                x="metric", y="result", hue='size',
                hue_order=hue_order,
                order=metric_order,  # Ensure consistent metric order
                dodge=.2, alpha=.4, legend=False,
                jitter=False, ax=ax, palette="tab20c"
            )
            point_plot = sns.pointplot(
                data=data_subset,
                x="metric", y="result", hue='size',
                hue_order=hue_order,
                order=metric_order,  # Ensure consistent metric order
                dodge=.6, linestyle='none', 
                errorbar='sd', capsize=0.15,
                marker="_", markersize=10, markeredgewidth=2.5, 
                ax=ax, palette="tab20b", alpha=1,
                err_kws={'linewidth': 1}
            )
            
            # Add connecting lines between means
            for metric_val in metric_order:
                # Calculate x-position for current metric
                x_pos = metric_order.index(metric_val)
                
                # Calculate positions for line endpoints
                line_x = [x_pos - 0.3, x_pos, x_pos + 0.3]
                
                # Get y-values (means) from pointplot
                line_y = []
                for hue_val in hue_order:
                    # Filter data for specific metric and hue
                    filtered = data_subset[(data_subset['metric'] == metric_val) & 
                                        (data_subset['size'] == hue_val)]
                    # Calculate mean if data exists
                    if not filtered.empty:
                        line_y.append(filtered['result'].mean())
                    else:
                        line_y.append(np.nan)
                
                # Plot connecting line if we have all values
                if len(line_y) == 3 and not any(np.isnan(line_y)):
                    ax.plot(line_x, line_y, color='gray', linestyle='-', 
                            linewidth=1, alpha=0.5, zorder=0)
            
            # Formatting
            ax.spines[['top', 'right']].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            if i == 0:
                ax.set_title(f' # Number of Dimensions = {Sizes[dim]}', fontsize=11)
            
            # Remove left spine for middle and right columns
            if j > 0:
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', color='gray')
            
            # Add y-label for leftmost plots
            if j == 0:
                ax.set_ylabel(measure)
            
            # Add legend only to top-left plot
            if i == 1 and j == 2:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:len(hue_order)], 
                    [features_distribution_dict.get(k, None) for k in hue_order],
                    title='Feature types',
                    title_fontsize=9,
                    fontsize=8,
                    loc='lower right'
                )
            else:
                ax.get_legend().remove()
        
    # Add common x-axis label
    fig.text(0.5, 0.01, 'Distance Metrics', ha='center', va='center', fontsize=12)

    # Remove individual x-axis labels from all subplots
    for ax in axes.flat:
        ax.set_xlabel('')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make space for x-axis label
    plt.savefig("plots/clustering_scores_artificial.pdf", dpi=300)
    # plt.savefig("plots/plotBox Plots.png") 
    plt.show()
    pass