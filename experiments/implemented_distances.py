# Description: distance matrices mostly implemented by copilot, seems they do the correct things
# Author: Anton D. Lautrup
# Date: 20-05-2025

import numpy as np
import pandas as pd
from collections import defaultdict

from experiments.prepare_data import get_categorical_features

def heom_distance_matrix(X: pd.DataFrame, Y: pd.DataFrame | None = None, unique_threshold: int = 10) -> np.ndarray:
    """
    Calculate the HEOM (Heterogeneous Euclidean-Overlap Metric) distance matrix between two datasets.
    Arguments:
        X (DataFrame): The first dataset (n_samples_x, n_features)
        Y (DataFrame): The second dataset (n_samples_y, n_features). If None, uses X.
        unique_threshold (int): Threshold for determining categorical features.
    Returns:
        dist_matrix (ndarray): (n_samples_x, n_samples_y) distance matrix.
    """
    Y_actual = Y if Y is not None else X

    len_X = len(X)
    len_Y = len(Y_actual)

    # Identify feature types
    df = pd.concat((X, Y_actual), axis=0)
    cat_feature_names = get_categorical_features(df, unique_threshold)
    num_feature_names = [col for col in df.columns if col not in cat_feature_names]

    # Precompute ranges for numerical features (avoid division by zero)
    feat_range_dict = {}
    for col in num_feature_names:
        col_data = df[col]
        rng = col_data.max() - col_data.min()
        feat_range_dict[col] = rng if rng > 0 else 1.0

    # Prepare numpy arrays for fast computation
    X_num = X[num_feature_names].values if num_feature_names else np.empty((len_X, 0))
    Y_num = Y_actual[num_feature_names].values if num_feature_names else np.empty((len_Y, 0))
    X_cat = X[cat_feature_names].values if cat_feature_names else np.empty((len_X, 0))
    Y_cat = Y_actual[cat_feature_names].values if cat_feature_names else np.empty((len_Y, 0))

    feat_ranges_np = np.array([feat_range_dict[col] for col in num_feature_names], dtype=float)

    dist_matrix = np.zeros((len_X, len_Y))

    for j_idx in range(len_Y):
        # Numerical part: normalized absolute difference, missing values get distance 1
        current_dist_num = np.zeros(len_X)
        if num_feature_names:
            y_num_row = Y_num[j_idx, :]
            x_nan = np.isnan(X_num)
            y_nan = np.isnan(y_num_row)
            nan_mask = x_nan | y_nan  # shape (len_X, n_num_features)
            abs_diff = np.abs(X_num - y_num_row)
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_diff = abs_diff / feat_ranges_np
            norm_diff[nan_mask] = 1.0  # HEOM: missing values get distance 1
            current_dist_num = np.sqrt(np.nansum(norm_diff ** 2, axis=1))

        # Categorical part: 0 if equal, 1 if not, missing values get distance 1
        current_dist_cat = np.zeros(len_X)
        if cat_feature_names:
            y_cat_row = Y_cat[j_idx, :]
            x_nan = pd.isnull(X_cat)
            y_nan = pd.isnull(y_cat_row)
            nan_mask = x_nan | y_nan  # shape (len_X, n_cat_features)
            diff_mask = (X_cat != y_cat_row)
            diff_mask[nan_mask] = True  # missing values get distance 1
            current_dist_cat = np.sqrt(np.sum(diff_mask.astype(float), axis=1))

        # Combine numerical and categorical distances
        dist_matrix[:, j_idx] = np.sqrt(current_dist_num ** 2 + current_dist_cat ** 2)

    return dist_matrix

def ichino_yaguchi_distance_matrix(X: pd.DataFrame, Y: pd.DataFrame | None = None, unique_threshold: int = 10) -> np.ndarray:
    """
    Calculate the Ichino-Yaguchi Generalised Euclidean distance matrix between two datasets.
    Arguments:
        X (DataFrame): The first dataset (n_samples_x, n_features)
        Y (DataFrame): The second dataset (n_samples_y, n_features). If None, uses X.
        unique_threshold (int): Threshold for determining categorical features.
    Returns:
        dist_matrix (ndarray): (n_samples_x, n_samples_y) distance matrix.
    """
    Y_actual = Y if Y is not None else X

    len_X = len(X)
    len_Y = len(Y_actual)

    # Identify feature types
    df = pd.concat((X, Y_actual), axis=0)
    cat_feature_names = get_categorical_features(df, unique_threshold)
    num_feature_names = [col for col in df.columns if col not in cat_feature_names]

    # Compute dmn for each feature
    dmn = {}
    for col in num_feature_names:
        col_data = df[col]
        rng = col_data.max() - col_data.min()
        dmn[col] = rng if rng > 0 else 1.0
    for col in cat_feature_names:
        dmn[col] = df[col].nunique() if df[col].nunique() > 0 else 1.0

    # Prepare numpy arrays for fast computation
    X_num = X[num_feature_names].values if num_feature_names else np.empty((len_X, 0))
    Y_num = Y_actual[num_feature_names].values if num_feature_names else np.empty((len_Y, 0))
    X_cat = X[cat_feature_names].values if cat_feature_names else np.empty((len_X, 0))
    Y_cat = Y_actual[cat_feature_names].values if cat_feature_names else np.empty((len_Y, 0))

    dmn_num = np.array([dmn[col] for col in num_feature_names], dtype=float)
    dmn_cat = np.array([dmn[col] for col in cat_feature_names], dtype=float)

    dist_matrix = np.zeros((len_X, len_Y))

    for j_idx in range(len_Y):
        # Numerical/discrete part: normalized absolute difference
        current_dist_num = np.zeros(len_X)
        if num_feature_names:
            y_num_row = Y_num[j_idx, :]
            abs_diff = np.abs(X_num - y_num_row)
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_diff = abs_diff / dmn_num
            norm_diff[np.isnan(norm_diff)] = 0.0  # treat NaNs as 0 distance
            current_dist_num = np.sum(norm_diff ** 2, axis=1)

        # Categorical part: 1/dmn if not equal, 0 if equal
        current_dist_cat = np.zeros(len_X)
        if cat_feature_names:
            y_cat_row = Y_cat[j_idx, :]
            diff_mask = (X_cat != y_cat_row)
            norm_diff_cat = diff_mask.astype(float) / dmn_cat
            norm_diff_cat[np.isnan(norm_diff_cat)] = 0.0  # treat NaNs as 0 distance
            current_dist_cat = np.sum(norm_diff_cat ** 2, axis=1)

        # Combine numerical and categorical distances
        dist_matrix[:, j_idx] = np.sqrt(current_dist_num + current_dist_cat)

    return dist_matrix

def compute_vdm_tables(X: pd.DataFrame, y_labels: pd.Series, unique_threshold: int = 10):
    """Compute VDM tables, std dict, and related info from training data."""
    from collections import defaultdict
    import numpy as np
    import pandas as pd

    df_train = X.reset_index(drop=True)
    y_train_labels = y_labels.reset_index(drop=True)

    cat_feature_names = get_categorical_features(df_train, unique_threshold)
    num_feature_names = [col for col in df_train.columns if col not in cat_feature_names]
    
    # Ensure class_values has a consistent order
    class_values = np.unique(y_train_labels) 
    class_to_idx_map = {cls_val: i for i, cls_val in enumerate(class_values)}

    # Numerical std
    std_dict = {}
    for col in num_feature_names:
        std = df_train[col].std()
        std_dict[col] = std if std > 0.0 else 1.0

    # VDM tables using more efficient pandas operations
    vdm_probs_np_lookup = {}

    temp_df_for_counts = df_train[cat_feature_names].copy()
    temp_df_for_counts['__CLASS__'] = y_train_labels

    for col in cat_feature_names:
        # P(v,c) counts: occurrences of each value with each class
        # Using pivot_table is robust for getting all combinations
        counts_vc_df = pd.pivot_table(temp_df_for_counts, index=col, columns='__CLASS__', aggfunc='size', fill_value=0)
        
        # P(v) counts: total occurrences of each value
        counts_v_series = df_train[col].value_counts()

        unique_vals_in_col = counts_v_series.index.tolist()
        val_to_idx_map = {val: i for i, val in enumerate(unique_vals_in_col)}
        
        # Matrix: rows are unique_vals_in_col, columns are class_values
        # Entries are P(class | value)
        # +1 row for unknown values (all probs = 0)
        prob_matrix = np.zeros((len(unique_vals_in_col) + 1, len(class_values)))
        default_idx_for_unknown = len(unique_vals_in_col) # Index for the row of zeros

        for i, val_i in enumerate(unique_vals_in_col):
            total_count_for_val_i = counts_v_series.get(val_i, 0)
            if total_count_for_val_i > 0:
                for j, class_j in enumerate(class_values):
                    # Get count of (val_i, class_j)
                    count_val_i_class_j = 0
                    if val_i in counts_vc_df.index and class_j in counts_vc_df.columns:
                        count_val_i_class_j = counts_vc_df.loc[val_i, class_j]
                    
                    prob_matrix[i, j] = count_val_i_class_j / total_count_for_val_i
        
        vdm_probs_np_lookup[col] = {
            'val_to_idx': val_to_idx_map,
            'prob_matrix': prob_matrix,
            'default_idx': default_idx_for_unknown
        }

    return vdm_probs_np_lookup, std_dict, cat_feature_names, num_feature_names, class_values

def hvdm_distance_matrix(
    X: pd.DataFrame,
    Y: pd.DataFrame | None = None,
    vdm_probs_np_lookup: dict | None = None, # Changed from vdm_probs
    std_dict: dict | None = None,
    cat_feature_names: list | None = None,
    num_feature_names: list | None = None,
    class_values: list | None = None, # Make sure this is ordered consistently (e.g. np.unique)
    unique_threshold: int = 10,
    y_labels: pd.Series | None = None # y_labels for X, if computing tables
) -> np.ndarray:
    """
    Calculate the HVDM distance matrix. Optimized version.
    If vdm_probs_np_lookup/std_dict are not provided, they are computed from X and y_labels.
    """
    import numpy as np

    Y_actual = Y if Y is not None else X
    len_X = len(X)
    len_Y = len(Y_actual)

    if vdm_probs_np_lookup is None or std_dict is None or \
       cat_feature_names is None or num_feature_names is None or class_values is None:
        if y_labels is None:
            raise ValueError("If VDM/std tables are not provided, y_labels for X must be given.")
        # Note: compute_vdm_tables now expects X to be the training set for table computation
        vdm_probs_np_lookup, std_dict, cat_feature_names, num_feature_names, class_values = \
            compute_vdm_tables(X, y_labels, unique_threshold)

    # Prepare numpy arrays for X and Y
    X_num_np = X[num_feature_names].values if num_feature_names else np.empty((len_X, 0))
    Y_num_np = Y_actual[num_feature_names].values if num_feature_names else np.empty((len_Y, 0))
    X_cat_np = X[cat_feature_names].values if cat_feature_names else np.empty((len_X, 0))
    Y_cat_np = Y_actual[cat_feature_names].values if cat_feature_names else np.empty((len_Y, 0))
    
    std_num_np = np.array([std_dict[col] for col in num_feature_names], dtype=float) if num_feature_names else np.empty(0)

    dist_matrix = np.zeros((len_X, len_Y))

    for j_idx in range(len_Y):
        # Numerical part (already reasonably optimized)
        current_dist_num_sq = np.zeros(len_X)
        if num_feature_names:
            y_num_row = Y_num_np[j_idx, :]
            abs_diff = np.abs(X_num_np - y_num_row)
            # Handle cases where std_num_np might be zero, though compute_vdm_tables tries to make it 1.0
            # Ensure std_num_np has same length as abs_diff.shape[1]
            if std_num_np.size > 0:
                 with np.errstate(divide='ignore', invalid='ignore'):
                    norm_diff = abs_diff / std_num_np
                 norm_diff[np.isnan(norm_diff) | np.isinf(norm_diff)] = 0.0 # If std was 0 or value was NaN
            else: # No numerical features, or std_num_np is empty
                norm_diff = np.zeros_like(abs_diff)

            current_dist_num_sq = np.sum(norm_diff ** 2, axis=1)

        # Categorical part: VDM (optimized)
        current_dist_cat_sq = np.zeros(len_X)
        if cat_feature_names:
            y_cat_row_for_j = Y_cat_np[j_idx, :] # Values for all cat features for current Y sample

            for k, col in enumerate(cat_feature_names):
                col_lookup = vdm_probs_np_lookup[col]
                val_to_idx_map_col = col_lookup['val_to_idx']
                prob_matrix_col = col_lookup['prob_matrix']
                default_idx_col = col_lookup['default_idx']

                y_val_k = y_cat_row_for_j[k]
                
                y_prob_idx = val_to_idx_map_col.get(y_val_k, default_idx_col)
                P_y_c_vec = prob_matrix_col[y_prob_idx, :] # Shape (n_classes,)

                # Map all X values for this column to their probability vectors
                x_vals_for_col_k = X_cat_np[:, k] # Shape (len_X,)
                
                # Vectorized mapping of x_vals to their row indices in prob_matrix_col
                # This list comprehension is mapping values to int indices, relatively fast.
                x_prob_indices = np.array([val_to_idx_map_col.get(x_val, default_idx_col) for x_val in x_vals_for_col_k])
                
                P_X_c_matrix = prob_matrix_col[x_prob_indices, :] # Shape (len_X, n_classes)
                
                # Difference in probabilities and sum of squares
                diff_P_c = P_X_c_matrix - P_y_c_vec # Broadcasting P_y_c_vec
                vdm_sq_for_feature = np.sum(diff_P_c**2, axis=1) # Shape (len_X,)
                
                current_dist_cat_sq += vdm_sq_for_feature
        
        dist_matrix[:, j_idx] = np.sqrt(current_dist_num_sq + current_dist_cat_sq)

    return dist_matrix