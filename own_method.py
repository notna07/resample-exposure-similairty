# Description: Script for holding our distance matrix calculations
# Author: Anton D. Lautrup
# Date: 20-05-2025

import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import Series, DataFrame

from prepare_data import scott_ref_rule, get_categorical_features

def prob_reroll_cat(counts: Series, target: int):
    """Function to calculate the probability of rolling a categorical variable into the target variable."""
    # Use .loc with a default value to avoid deprecation of .get
    return counts.loc[target] if target in counts.index else 0

def prob_reroll_num(histogram: ndarray, binning: ndarray,  target: float):
    """Calculate the probability of rolling a numerical variable into the target variable."""
    # Get the counts of each bin in the variable
    
    bin_index = np.digitize(target, binning) - 1

    # Handle cases where target is outside bin range or histogram is empty
    if bin_index < 0 or bin_index >= len(histogram) or len(histogram) == 0:
        return 0.0
    
    current_sum_hist = sum(histogram)
    if current_sum_hist == 0: # Avoid division by zero if histogram sum is zero
        return 0.0
        
    prob = histogram[bin_index] / current_sum_hist
    return prob

# def own_distance_matrix(df_x, df_y=None):
#     """
#     Compute the rerollers distance matrix between two dataframes.
#     (the chance that randomly resampling the variable will make it into the target variable)
#     """

#     # join the datasets
#     if df_y is None:
#         df_y = df_x
#     df = pd.concat((df_x,df_y), axis=0)

#     cat_features = get_categorical_features(df_y)
#     num_features = [col for col in df_y.columns if col not in cat_features]

#     bin_range, histogram, feat_range = {}, {}, {}
#     for col in num_features:
#         if np.issubdtype(df_y[col].dtype, np.floating):
#             bin_range[col] = scott_ref_rule(df_y[col])
#             histogram[col] = np.histogram(df_y[col], bins=bin_range[col])[0]
#         else:
#             bin_range[col] = np.arange(df_y[col].nunique()+1)
#             histogram[col] = np.histogram(df_y[col], bins=bin_range[col])[0]
            
#         feat_range[col] = df_y[col].max()-df_y[col].min()
    
#     cat_counts = {}
#     for col in cat_features:
#         cat_counts[col] = df_y[col].value_counts(normalize=True)

#     # create the distance matrix
#     dist_matrix = np.zeros((len(df_x), len(df_y)))

#     for i, row in enumerate(df_x.iterrows()):
#         for j, col in enumerate(df_y.iterrows()):
#             for col_name in num_features:
#                 dist_matrix[i,j] += (1-prob_reroll_num(histogram[col_name], bin_range[col_name], j))*np.abs(row[1][col_name] - col[1][col_name]) / feat_range[col_name]
#             for col_name in cat_features:
#                 if row[1][col_name] != col[1][col_name]:
#                     dist_matrix[i,j] += (1-prob_reroll_cat(cat_counts[col_name], j))
#     return dist_matrix

def own_distance_matrix(X: DataFrame, Y: DataFrame | None = None, unique_threshold: int = 10) -> ndarray:
    """ Function to calculate the distance matrix between two datasets using a custom distance metric.
    The distance metric is a combination of categorical and numerical features. (Optimized Version)

    Arguments:
        - X (DataFrame): The first dataset.
        - Y (DataFrame): The second dataset. If None, the distance matrix is calculated within X.
        - unique_threshold (int): The threshold for determining categorical features.
    
    Returns:
        - dist_matrix (ndarray): The distance matrix.
    """    
    Y_actual = Y if Y is not None else X
    
    len_X = len(X)
    len_Y_actual = len(Y_actual)

    # Concatenate for global statistics.
    # df = pd.concat((X, Y_actual), axis=0)

    cat_feature_names = get_categorical_features(Y_actual, unique_threshold)
    num_feature_names = [col for col in Y_actual.columns if col not in cat_feature_names]

    bin_range_hist_dict, histogram_dict, feat_range_dict = {}, {}, {}
    for col_name in num_feature_names:
        col_data = Y_actual[col_name]
        if col_data.empty or col_data.isnull().all():
            histogram_dict[col_name] = np.array([])
            bin_range_hist_dict[col_name] = np.array([])
            feat_range_dict[col_name] = np.nan 
            continue

        if np.issubdtype(col_data.dtype, np.floating):
            # Drop NaNs for scott_ref_rule and histogram calculation data
            col_data_dropna = col_data.dropna()
            if col_data_dropna.empty:
                histogram_dict[col_name] = np.array([])
                bin_range_hist_dict[col_name] = np.array([])
            else:
                bins_param = scott_ref_rule(col_data_dropna)
                hist_values, hist_bins = np.histogram(col_data_dropna, bins=bins_param)
                histogram_dict[col_name] = hist_values
                bin_range_hist_dict[col_name] = hist_bins
        else: # Integer types
            # Original behavior: use nunique on the column (which doesn't count NaNs) for bin definition
            # And use the column directly (with NaNs, which np.histogram ignores in data) for histogramming
            num_unique_col = col_data.nunique()
            bins_param = np.arange(num_unique_col + 1)
            bin_range_hist_dict[col_name] = bins_param
            
            # np.histogram ignores NaNs in the input data array `a`
            hist_values, _ = np.histogram(col_data, bins=bins_param)
            histogram_dict[col_name] = hist_values
            
        var_range = col_data.max() - col_data.min()
        feat_range_dict[col_name] = var_range if var_range > 0 else 1.0 # Avoid division by zero

    cat_counts_dict = {}
    for col_name in cat_feature_names:
        cat_counts_dict[col_name] = Y_actual[col_name].value_counts(normalize=True)

    # Precompute all weights for the distance matrix
    num_weights_matrix = np.zeros((len(num_feature_names), len_Y_actual))
    if num_feature_names:
        for k, col_name in enumerate(num_feature_names):
            if col_name in histogram_dict and len(histogram_dict[col_name]) > 0:
                for j_idx in range(len_Y_actual):
                    target_value = Y_actual.iloc[j_idx][col_name]
                    num_weights_matrix[k, j_idx] = 1.0 - prob_reroll_num(
                        histogram_dict[col_name], 
                        bin_range_hist_dict[col_name], 
                        target_value
                    )
            else: # Histogram was empty or not computed (e.g. all NaN column)
                  # prob_reroll_num returns 0 if histogram is empty, so weight is 1.
                num_weights_matrix[k, :] = 1.0

    cat_weights_matrix = np.zeros((len(cat_feature_names), len_Y_actual))
    if cat_feature_names:
        for k, col_name in enumerate(cat_feature_names):
            if col_name in cat_counts_dict:
                for j_idx in range(len_Y_actual):
                    target_value = Y_actual.iloc[j_idx][col_name]
                    cat_weights_matrix[k, j_idx] = 1.0 - prob_reroll_cat(
                        cat_counts_dict[col_name], 
                        target_value
                    )
            else: # Should not occur if cat_feature_names are valid
                cat_weights_matrix[k, :] = 1.0 # Default if counts somehow not found

    X_num_np = X[num_feature_names].values if num_feature_names else np.empty((len_X, 0))
    Y_num_np = Y_actual[num_feature_names].values if num_feature_names else np.empty((len_Y_actual, 0))
    
    X_cat_np = X[cat_feature_names].values if cat_feature_names else np.empty((len_X, 0))
    Y_cat_np = Y_actual[cat_feature_names].values if cat_feature_names else np.empty((len_Y_actual, 0))

    feat_ranges_np = np.array([feat_range_dict.get(col, np.nan) for col in num_feature_names], dtype=float)

    dist_matrix = np.zeros((len_X, len_Y_actual))

    for j_idx in range(len_Y_actual):
        current_dist_num = np.zeros(len_X)
        if num_feature_names:
            y_num_row = Y_num_np[j_idx, :]
            abs_diff_num = np.abs(X_num_np - y_num_row)

            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_abs_diff_num = abs_diff_num / feat_ranges_np
            
            current_col_num_weights = num_weights_matrix[:, j_idx]
            weighted_diff_terms_num = normalized_abs_diff_num * current_col_num_weights
            current_dist_num = np.nansum(weighted_diff_terms_num, axis=1) # Use nansum if terms can be NaN due to feat_range

        current_dist_cat = np.zeros(len_X)
        if cat_feature_names:
            y_cat_row = Y_cat_np[j_idx, :]
            diff_cat_mask = (X_cat_np != y_cat_row)
            current_col_cat_weights = cat_weights_matrix[:, j_idx]
            weighted_diff_terms_cat = diff_cat_mask * current_col_cat_weights
            current_dist_cat = np.sum(weighted_diff_terms_cat, axis=1) # Categorical weights shouldn't introduce NaNs
        
        # If current_dist_num became NaN (e.g. from 0/0 in normalization and then nansum didn't make it 0)
        # it will propagate here.
        dist_matrix[:, j_idx] = current_dist_num + current_dist_cat
            
    return dist_matrix

if __name__ == "__main__":
    import doctest
    doctest.ELLIPSIS_MARKER = '-etc-'
    doctest.testmod()