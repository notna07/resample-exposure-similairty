# Description: Script for calculating the resample exposure index
# Author: Anton D. Lautrup
# Date: 21-05-2025
# Version: 1.1
# License: MIT

import numpy as np

from numpy import ndarray
from typing import Dict, List, Tuple, Union
from pandas import DataFrame, Series

from joblib import Parallel, delayed

from .utils.preprocessing import get_cat_variables, scott_ref_rule

class ResampleExposure:
    """Class to calculate the resample exposure index between two datasets."""

    def __init__(self, target_distribution: DataFrame, 
                 categorical_features: List[str] = None, 
                 unique_threshold: int = 10,
                 feature_weights: Union[Dict[str, float], np.ndarray, List[float]] = None):
        """ Initialize the ResampleExposure with a memorised distribution.

        Arguments:
            - target_distribution (DataFrame): The target distribution used for modelling the distributions (unless overwritten).
            - categorical_features (List[str]): List of categorical features in the dataset, 
                if None the categorical features will be determined from the memorised distribution.
            - unique_threshold (int): Threshold for determining if a numerical feature is categorical.
            - feature_weights (Union[Dict[str, float], np.ndarray, List[float]]): Weights for each feature.
                If a dictionary, keys are feature names and values are weights.
                If a NumPy array or list, it must be in the same order as target_distribution.columns.
                If None, all features are weighted equally with 1.0.
        """
        self.memorised_distribution = target_distribution.copy()
        self.unique_threshold = unique_threshold # Store for potential use in overwrite_memory

        detected_cat_features = get_cat_variables(self.memorised_distribution, unique_threshold)
        if categorical_features is None:
            self.categorical_features = detected_cat_features
        else: # combine detected and provided categorical features
            self.categorical_features = list(set(detected_cat_features) | set(categorical_features))
        print("Categorical features detected:", self.categorical_features)

        self.numerical_features = [col for col in self.memorised_distribution.columns if col not in self.categorical_features]

        # Initialize feature weights
        all_columns = list(self.memorised_distribution.columns)
        self.weights: Dict[str, float] = {}
        if feature_weights is None:
            self.weights = {col: 1.0 for col in all_columns}
        elif isinstance(feature_weights, dict):
            self.weights = {col: feature_weights.get(col, 1.0) for col in all_columns}
            # Optional: Warn about keys in feature_weights not in all_columns
            # for key in feature_weights:
            #     if key not in self.weights:
            #         print(f"Warning: Weight provided for feature '{key}' not present in target_distribution.")
        elif isinstance(feature_weights, (np.ndarray, list)):
            if len(feature_weights) != len(all_columns):
                raise ValueError(
                    "Length of feature_weights array/list must match the number of columns in target_distribution."
                )
            self.weights = {col: weight for col, weight in zip(all_columns, feature_weights)}
        else:
            raise TypeError(
                "feature_weights must be a dictionary, numpy array, list, or None."
            )
        
        self._setup_done = False # Ensure setup is marked as not done until _setup() completes
        self._setup()
        pass

    def _setup(self) -> None:
        """ Setup the ResampleExposure by computing the ranges and histograms of the features. """
        self.numerical_ranges = self._compute_numerical_feature_ranges()
        self.cat_counts = self._get_cat_feature_counts()
        self.bin_ranges = self._get_bin_ranges_of_numerical_features()
        self.histograms = self._get_histogram_of_numerical_features(self.bin_ranges)
        self.height_diff = self._get_histogram_total_height_difference(self.histograms)
        self._setup_done = True
        pass

    def _compute_numerical_feature_ranges(self) -> Dict[str, float]:
        """ Compute the range of a numerical feature in the memorised distribution.
        Returns:
            - Dictionary with the range of each numerical feature.

        Example:
            >>> ranges = resample_exposure_index._infer_numerical_feature_ranges()
            >>> print(ranges)
            {'feature1': 1.0, 'feature2': 10.0}
        """
        ranges = {}
        for col in self.numerical_features:
            if self.memorised_distribution[col].nunique() > 1:
                ranges[col] = self.memorised_distribution[col].max() - self.memorised_distribution[col].min()
            else:
                ranges[col] = 1.0
        return ranges
    
    def _get_cat_feature_counts(self) -> Dict[str, Series]:
        """ Get the counts of each categorical feature in the memorised distribution.
        Returns:
            - Dictionary with the counts of each categorical feature.
        
        Example:
            >>> cat_counts = resample_exposure_index._get_cat_feature_counts()
            >>> print(cat_counts)
            {'feature1': Series([0.5, 0.5]), 'feature2': Series([0.3, 0.7])}
        """
        cat_counts = {}
        for col in self.categorical_features:
            cat_counts[col] = self.memorised_distribution[col].value_counts(normalize=True)
        return cat_counts
    
    def _get_bin_ranges_of_numerical_features(self) -> Dict[str, List[float]]:
        """ Infer the bin ranges for numerical features in the memorised distribution.
        Returns:
            - Dictionary with the bin ranges for each numerical feature.

        Example:
            >>> bin_ranges = resample_exposure_index._get_bin_ranges_of_numerical_features()
            >>> print(bin_ranges)
            {'feature1': [0.0, 1.0, 2.0], 'feature2': [0.0, 5.0, 10.0]}
        """
        bin_ranges = {}
        for col in self.numerical_features:
            bin_ranges[col] = scott_ref_rule(self.memorised_distribution[col])
        return bin_ranges
    
    def _get_histogram_of_numerical_features(self, bin_ranges: Dict[str, List[float]]) -> Dict[str, ndarray]:
        """ Get the histogram of numerical features in the memorised distribution.
        Returns:
            - Dictionary with the histogram of each numerical feature.
        
        Example:
            >>> histograms = resample_exposure_index._get_histogram_of_numerical_features()
            >>> print(histograms)
            {'feature1': array([0.5, 0.5]), 'feature2': array([0.3, 0.7])}
        """
        histograms = {}
        for col in self.numerical_features:
            histograms[col] = np.histogram(self.memorised_distribution[col], bins=bin_ranges[col])[0]
        return histograms

    def _get_histogram_total_height_difference(self, histograms: Dict[str, ndarray]) -> Dict[str, Tuple[float, float]]:
        """ Compute the height differences going from one column to the next over the full histogram.
        
        Arguments:
            - histograms (Dict[str, ndarray]): The histograms of the numerical features.
        Returns:
            - height difference (float): The total height difference of the histograms.
        Example:
            >>> histograms = {'feature1': array([0.5, 0.5]), 'feature2': array([0.3, 0.7])}
            >>> height_diff = resample_exposure_index._get_histogram_total_height_difference(histograms)
            >>> print(height_diff)
            {'feature1': 1.0, 'feature2': 1.4}
        """
        height_diff = {}
        for col in self.numerical_features:
            hist = histograms[col]
            if len(hist) > 1:
                bin_heights = hist / np.sum(hist)  # Normalise the histogram
                bin_heights = np.concatenate(([0], bin_heights, [0]))  # Add zero height at the edges
                height_diffs = np.diff(bin_heights)
                left_to_right_height_diffs = height_diffs[height_diffs > 0]
                right_to_left_height_diffs = height_diffs[height_diffs < 0]
                height_diff[col] = (np.sum(np.abs(left_to_right_height_diffs)), np.sum(np.abs(right_to_left_height_diffs)))
            else:
                height_diff[col] = (0.0, 0.0)
        return height_diff

    def _get_height_of_histogram_descent(self, query_val: str, target_val: str,
                                         bin_ranges: List[float], histogram: ndarray) -> float:
        """ Compute the height of the histogram descent for a given query value and target value.
        Arguments:
            - query_val (str): The query value.
            - target_val (str): The target value.
            - bin_ranges (List[float]): The bin ranges of the histogram.
            - histograms (ndarray): The histogram of the numerical feature.
        Returns:
            - height of histogram descent (float): amount of distance we climb down when 
                traversing the histogram between the query and target values.
        Example:
            >>> query_val = 2.0
            >>> target_val = 1.0
            >>> bin_ranges = [0.5, 1.5, 2.5]
            >>> histograms = np.array([0.3, 0.7])
            >>> height = resample_exposure_index._get_height_of_histogram_descent(query_val, target_val, bin_ranges, histograms)
            >>> print(height)
            0.4
        """
        bin_index_query = np.digitize(query_val, bin_ranges) - 1
        bin_index_target = np.digitize(target_val, bin_ranges) - 1

        bin_heights = histogram / np.sum(histogram)  # Normalise the histogram
        # Get the height of the histogram at the bin index
        if bin_index_query == bin_index_target:
            return 0.0
        elif bin_index_query < bin_index_target:
            bin_height_diffs = np.diff(bin_heights[bin_index_query:bin_index_target + 1])
        else:
            bin_height_diffs = np.diff(bin_heights[bin_index_target:bin_index_query + 1])
        # return only the negative differences
        bin_height_diffs[bin_height_diffs > 0] = 0
        bin_height_diffs = np.abs(bin_height_diffs)
        return np.sum(bin_height_diffs)

    def resample_exposure_sim(self, query_point: Series, target_point: Series, normalised: bool = False) -> float:
        """ Compute the resample exposure similarity for a given query point and target point.

        Arguments:
            - query_point (Series): A query point to compute the resample exposure index for.
            - target_point (Series): The target point in the memorised distribution to compare against.
            - normalised (bool): If True, the resample exposure index will be normalised by the sum of weights of active features.

        Returns:
            - resample exposure index (float): The resample exposure index for the query point to be made into the target point.

        Example:
            >>> query_point = pd.Series({'feature1': 1.0, 'feature2': 5.0})
            >>> target_point = pd.Series({'feature1': 2.0, 'feature2': 7.0})
            >>> feature_weights = {'feature1': 0.7, 'feature2': 0.3}
            >>> resample_exposure_index = ResampleExposure(memorised_distribution, feature_weights=feature_weights)
            >>> index = resample_exposure_index.compute_resample_exposure_index(query_point, target_point)
            >>> print(index)
            # Value will depend on weights and internal calculations
        """
        if not self._setup_done:
            raise ValueError("ResampleExposure is not set up. Call _setup() first.")

        if target_point.shape[0] != self.memorised_distribution.shape[1]:
            raise ValueError("Target point must have the same columns as the memorised distribution.")
        
        if query_point.shape[0] != target_point.shape[0]:
            raise ValueError("Query and target points must have the same columns.")
        
        resample_exposure_index = 0.0
        feature_weight: float

        for col in self.categorical_features:
            feature_weight = self.weights.get(col, 1.0) # Default to 1.0 if somehow not in weights
            if feature_weight == 0: continue # Skip if weight is zero

            target = target_point[col]
            if query_point[col] != target:
                resample_exposure_index += (self.cat_counts[col].loc[target] if target in self.cat_counts[col].index else 0) * feature_weight
            else:
                resample_exposure_index += 1.0 * feature_weight

        for col in self.numerical_features:
            feature_weight = self.weights.get(col, 1.0) # Default to 1.0 if somehow not in weights
            if feature_weight == 0: continue # Skip if weight is zero

            neg_descent = self._get_height_of_histogram_descent(query_point[col], target_point[col], self.bin_ranges[col], self.histograms[col])

            diff_num = query_point[col] - target_point[col]

            height_key = 0 if diff_num < 0 else 1
                
            sim = 0.0 # Default similarity
            if self.numerical_ranges[col] == 0: # Avoid division by zero if range is zero
                if diff_num == 0: # If range is zero and values are same, similarity is 1
                    sim = 1.0
                else: # If range is zero and values differ, similarity is 0
                    sim = 0.0
            elif self.height_diff[col][height_key] == 0: # Avoid division by zero for height diff
                 # If max descent is 0, second term is 1 if neg_descent is also 0, else 0
                term1 = (1 - (abs(diff_num) / self.numerical_ranges[col]))
                term2 = 1.0 if neg_descent == 0 else 0.0
                sim = term1 * term2
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    sim_val = (1 - (abs(diff_num) / self.numerical_ranges[col])) * (1-(neg_descent/self.height_diff[col][height_key]))
                if np.isnan(sim_val): sim_val = 0.0 # Handle potential NaN from 0/0 if not caught by prior checks
                sim = sim_val


            bin_index = np.digitize(target_point[col], self.bin_ranges[col]) - 1
            if bin_index < 0 or bin_index >= len(self.histograms[col]) or len(self.histograms[col]) == 0:
                resample_exposure_index += 0.0 # No contribution if target is outside histogram bins
            else:
                resample_exposure_index += sim * feature_weight #* self.histograms[col][bin_index] / sum(self.histograms[col])

        if normalised:
            active_features = self.categorical_features + self.numerical_features
            sum_of_weights = sum(self.weights.get(f, 0.0) for f in active_features if self.weights.get(f, 0.0) > 0) # Sum of positive weights
            
            if sum_of_weights > 0:
                resample_exposure_index /= sum_of_weights
            elif resample_exposure_index == 0: # If sum_of_weights is 0 and index is 0, result is 0
                resample_exposure_index = 0.0
            else: # sum_of_weights is 0 (or negative, though weights assumed non-negative) but index isn't.
                resample_exposure_index = np.nan # Or 0.0, depending on desired behavior for 0 sum of weights.
        return resample_exposure_index
    
    def resample_exposure_matrix(self, query_df: DataFrame = None, normalised: bool = False, 
                                 reverse_direction: bool = False, overwrite_memory: bool = False,
                                 n_jobs: int = -1) -> ndarray: # Added n_jobs
        """ Compute the resample exposure matrix between two dataframes.

        Arguments:
            - query_df (DataFrame): The query dataframe to compute the resample exposure matrix for,
                if None the resample exposure matrix will be computed within the memorised distribution.
            - normalised (bool): If True, the resample exposure index will be normalised to [0, 1]
                by dividing by the total number of features.
            - reverse_direction (bool): If True, the roles of query and memorised distribution are swapped 
                for the comparison, but statistics are still drawn from the memorised distribution.
                E.g., for calculating synthetic-to-real exposure using only knowledge of 
                the synthetic distribution (which is self.memorised_distribution).
            - overwrite_memory (bool): If True, the memorised distribution will be overwritten with the query dataframe.
                This is useful for calculating the resample exposure matrix between two dataframes.
            - n_jobs (int): The number of jobs to run in parallel. -1 means using all processors.

        Returns:
            - resample exposure matrix (ndarray): The resample exposure matrix between the two dataframes.

        Example:
            >>> memorised_distribution = pd.DataFrame({'feature1': [1.0, 2.0, 1.0, 3.0], 'feature2': ['A', 'B', 'A', 'A']})
            >>> rex = ResampleExposure(memorised_distribution, categorical_features=['feature2'])
            >>> query_data = pd.DataFrame({'feature1': [1.5], 'feature2': ['A']})
            >>> matrix = rex.resample_exposure_matrix(query_data, normalised=False, n_jobs=1) # Example with n_jobs
            # This is a conceptual example; actual values depend on internal calculations like Scott's rule.
            # For feature1 (num): query=1.5. Suppose target 1.0 (range e.g. 2.0, hist_prob e.g. 0.5) -> (1-0.5/2)*0.5 = 0.375
            # For feature2 (cat): query='A', target='A' -> 1.0
            # Total for query_data[0] vs memorised_distribution[0] (1.0, 'A') could be around 1.0 + (1 - 0.5/range)*P(bin_1.0)
            # Example output structure:
            >>> # print(matrix) 
            >>> # [[value_q0_t0, value_q0_t1, value_q0_t2, value_q0_t3]]
        """
        if not self._setup_done:
            raise ValueError("ResampleExposure is not set up. Call _setup() first.")

        _processed_query_df: DataFrame
        if query_df is None:
            _processed_query_df = self.memorised_distribution.copy() 
        else:
            _processed_query_df = query_df.copy() 

        expected_columns = list(self.memorised_distribution.columns) # Ensure it's a list for consistent indexing
        try:
            _processed_query_df = _processed_query_df[expected_columns]
        except KeyError as e:
            raise ValueError(
                f"Query DataFrame is missing one or more columns expected from the memorised distribution: {e}. Expected: {expected_columns}, Got: {list(_processed_query_df.columns)}"
            ) from e

        original_memorised_distribution = None
        # These variables will hold the state of self's attributes for the parallel computation
        current_categorical_features = self.categorical_features
        current_numerical_features = self.numerical_features
        current_weights = self.weights
        current_numerical_ranges = self.numerical_ranges
        current_cat_counts = self.cat_counts
        current_bin_ranges = self.bin_ranges
        current_histograms = self.histograms
        current_height_diff = self.height_diff
        current_memorised_distribution_for_setup = self.memorised_distribution # For effective_target_df if not overwritten

        if overwrite_memory:
            original_memorised_distribution = self.memorised_distribution.copy()
            original_categorical_features = list(self.categorical_features)
            original_numerical_features = list(self.numerical_features)
            original_numerical_ranges = self.numerical_ranges.copy()
            original_cat_counts = {k: v.copy() for k, v in self.cat_counts.items()}
            original_bin_ranges = {k: list(v) for k, v in self.bin_ranges.items()}
            original_histograms = {k: v.copy() for k, v in self.histograms.items()}
            original_height_diff = self.height_diff.copy()
            original_setup_done = self._setup_done

            self.memorised_distribution = _processed_query_df.copy() # Overwrite memorised distribution
            
            # Re-derive features and re-run setup based on the new memorised_distribution
            detected_cat_features_overwrite = get_cat_variables(self.memorised_distribution, self.unique_threshold)
            self.categorical_features = [col for col in detected_cat_features_overwrite if col in self.memorised_distribution.columns]
            self.numerical_features = [col for col in self.memorised_distribution.columns if col not in self.categorical_features]
            self._setup() # This will use the new self.memorised_distribution

            # Update 'current_' variables to reflect the overwritten state for computation
            current_categorical_features = self.categorical_features
            current_numerical_features = self.numerical_features
            # current_weights remains self.weights (not changed by overwrite_memory)
            current_numerical_ranges = self.numerical_ranges
            current_cat_counts = self.cat_counts
            current_bin_ranges = self.bin_ranges
            current_histograms = self.histograms
            current_height_diff = self.height_diff
            # current_memorised_distribution_for_setup is updated to the new self.memorised_distribution (which is _processed_query_df)
            current_memorised_distribution_for_setup = self.memorised_distribution


        effective_query_df: DataFrame
        effective_target_df: DataFrame

        # Determine the actual DataFrames for comparison.
        # If overwrite_memory is True, statistics are derived from _processed_query_df,
        # but the "memorised" component of the comparison should be the *original* memorised_distribution.
        memorised_component_for_comparison: DataFrame
        if overwrite_memory:
            memorised_component_for_comparison = original_memorised_distribution
        else:
            memorised_component_for_comparison = self.memorised_distribution # This is current_memorised_distribution_for_setup if not overwritten

        if not reverse_direction:
            effective_query_df = _processed_query_df
            effective_target_df = memorised_component_for_comparison
        else: 
            effective_query_df = memorised_component_for_comparison
            effective_target_df = _processed_query_df
            
        num_effective_query_rows = len(effective_query_df)
        num_effective_target_rows = len(effective_target_df)
        
        result_matrix = np.zeros((num_effective_query_rows, num_effective_target_rows))

        # Precompute categorical probabilities using current (potentially overwritten) state
        cat_probs_for_effective_targets = np.zeros((num_effective_target_rows, len(current_categorical_features)))
        if current_categorical_features and num_effective_target_rows > 0:
            et_cat_values_all_local = effective_target_df[current_categorical_features].values
            for k_cat, col_name in enumerate(current_categorical_features):
                if col_name in current_cat_counts:
                    counts = current_cat_counts[col_name] 
                    for j in range(num_effective_target_rows):
                        target_val = et_cat_values_all_local[j, k_cat]
                        cat_probs_for_effective_targets[j, k_cat] = counts.get(target_val, 0.0)
                else: # Should not happen if features are consistent
                    cat_probs_for_effective_targets[:, k_cat] = 0.0
        
        # Precompute negative descent matrices for numerical features using current (potentially overwritten) state
        all_neg_descent_matrices = {}
        if current_numerical_features:
            for col_name_precompute in current_numerical_features:
                if col_name_precompute not in current_histograms or \
                   col_name_precompute not in current_bin_ranges or \
                   not current_histograms[col_name_precompute].size: # Check if histogram is empty
                    all_neg_descent_matrices[col_name_precompute] = np.empty((0,0), dtype=float) 
                    continue

                current_histogram_pre = current_histograms[col_name_precompute]
                num_bins_for_col = len(current_histogram_pre)

                if num_bins_for_col == 0: # Should be caught by .size check above
                    all_neg_descent_matrices[col_name_precompute] = np.empty((0,0), dtype=float)
                    continue

                sum_hist = np.sum(current_histogram_pre)
                if sum_hist == 0: 
                    all_neg_descent_matrices[col_name_precompute] = np.full((num_bins_for_col, num_bins_for_col), np.nan, dtype=float)
                    continue
                
                bin_heights = current_histogram_pre / sum_hist
                neg_descent_matrix_col = np.zeros((num_bins_for_col, num_bins_for_col), dtype=float)

                for b_q in range(num_bins_for_col): 
                    for b_t in range(num_bins_for_col): 
                        if b_q == b_t:
                            neg_descent_matrix_col[b_q, b_t] = 0.0
                            continue
                        
                        path_segment: ndarray
                        if b_q < b_t: 
                            path_segment = bin_heights[b_q : b_t + 1]
                        else: 
                            path_segment = bin_heights[b_t : b_q + 1][::-1]
                        
                        if len(path_segment) < 2: 
                            diffs = np.array([])
                        else:
                            diffs = np.diff(path_segment) 
                        
                        neg_diffs = diffs[diffs < 0] 
                        neg_descent_matrix_col[b_q, b_t] = np.sum(np.abs(neg_diffs))
                all_neg_descent_matrices[col_name_precompute] = neg_descent_matrix_col

        eq_cat_values_all = effective_query_df[current_categorical_features].values if current_categorical_features else np.empty((num_effective_query_rows, 0))
        eq_num_values_all = effective_query_df[current_numerical_features].values if current_numerical_features else np.empty((num_effective_query_rows, 0))
        
        et_cat_values_for_comp = effective_target_df[current_categorical_features].values if current_categorical_features else np.empty((num_effective_target_rows, 0))
        et_num_values_for_comp = effective_target_df[current_numerical_features].values if current_numerical_features else np.empty((num_effective_target_rows, 0))

        # Define helper function for parallel execution (captures variables from outer scope)
        def _process_query_row_parallel(i_query_row):
            current_query_total_score_vs_all_targets = np.zeros(num_effective_target_rows)

            # Categorical features contribution
            if current_categorical_features and num_effective_target_rows > 0:
                query_cat_row_vals = eq_cat_values_all[i_query_row, :] 
                matches = (query_cat_row_vals == et_cat_values_for_comp)
                cat_contribution_per_target = np.zeros(num_effective_target_rows)
                for k_cat, col_name in enumerate(current_categorical_features):
                    feature_weight = current_weights.get(col_name, 1.0)
                    if feature_weight == 0: continue

                    feature_k_cat_scores = np.where(matches[:, k_cat], 1.0, cat_probs_for_effective_targets[:, k_cat])
                    cat_contribution_per_target += feature_k_cat_scores * feature_weight
                current_query_total_score_vs_all_targets += cat_contribution_per_target

            # Numerical features contribution (vectorized inner loop)
            if current_numerical_features and num_effective_target_rows > 0:
                query_num_row_vals = eq_num_values_all[i_query_row, :]
                num_contribution_per_target_for_query_i = np.zeros(num_effective_target_rows)

                for k_num, col_name in enumerate(current_numerical_features):
                    feature_weight = current_weights.get(col_name, 1.0)
                    if feature_weight == 0: continue
                    

                    if col_name not in current_numerical_ranges or \
                       col_name not in current_bin_ranges or \
                       col_name not in current_histograms or \
                       col_name not in current_height_diff or \
                       col_name not in all_neg_descent_matrices or \
                       all_neg_descent_matrices[col_name].shape[0] == 0: 
                        continue 

                    q_val_scalar = query_num_row_vals[k_num]
                    target_vals_for_feature_k = et_num_values_for_comp[:, k_num] 
                    
                    current_feature_range_val = current_numerical_ranges[col_name]
                    current_bin_ranges_val = current_bin_ranges[col_name]
                    current_histogram_val = current_histograms[col_name] # Used for target_bin_index check
                    current_height_diff_tuple_val = current_height_diff[col_name]
                    precomputed_descent_matrix_col_val = all_neg_descent_matrices[col_name]
                    num_bins_for_col_runtime = len(current_histogram_val)

                    # Vectorized calculation for all targets for this feature
                    diff_num_vector = q_val_scalar - target_vals_for_feature_k
                    

                    neg_descent_vector = np.zeros_like(target_vals_for_feature_k, dtype=float)
                    if num_bins_for_col_runtime > 0:
                        bin_idx_q_scalar_clipped = np.clip(np.digitize(q_val_scalar, current_bin_ranges_val) - 1, 0, num_bins_for_col_runtime - 1)
                        bin_idx_t_vector_clipped = np.clip(np.digitize(target_vals_for_feature_k, current_bin_ranges_val) - 1, 0, num_bins_for_col_runtime - 1)
                        neg_descent_vector = precomputed_descent_matrix_col_val[bin_idx_q_scalar_clipped, bin_idx_t_vector_clipped]
                    elif np.sum(current_histogram_val) == 0 and num_bins_for_col_runtime > 0: # Should be caught by precomp if hist sum is 0
                        neg_descent_vector.fill(np.nan)
                    else: # num_bins_for_col_runtime == 0 or sum_hist != 0
                        neg_descent_vector.fill(0.0)
                    
                    height_key_vector = np.where(diff_num_vector < 0, 0, 1)
                    
                    term1_vector = np.zeros_like(diff_num_vector, dtype=float)
                    if np.isclose(current_feature_range_val, 0):
                        term1_vector = np.where(np.isclose(diff_num_vector, 0), 1.0, 0.0)
                    else:
                        term1_vector = (1.0 - (np.abs(diff_num_vector) / current_feature_range_val))
                    
                    term2_vector = np.zeros_like(diff_num_vector, dtype=float)
                    term2_denominator_vector = np.array([current_height_diff_tuple_val[key] for key in height_key_vector])
                    
                    mask_denom_zero = np.isclose(term2_denominator_vector, 0)
                    mask_num_zero = np.isclose(neg_descent_vector, 0) # Numerator for term2 is neg_descent_vector

                    term2_vector[mask_denom_zero & mask_num_zero] = 1.0  # 0/0 case for (neg_desc / max_desc) -> term2 is 1.0
                    term2_vector[mask_denom_zero & ~mask_num_zero] = 0.0 # x/0 case for (neg_desc / max_desc) -> term2 is 0.0
                    
                    mask_denom_nonzero = ~mask_denom_zero
                    if np.any(mask_denom_nonzero):
                        term2_vector[mask_denom_nonzero] = (1.0 - (neg_descent_vector[mask_denom_nonzero] / term2_denominator_vector[mask_denom_nonzero]))
                    
                    calculated_sim_vector = term1_vector * term2_vector
                    calculated_sim_vector[np.isnan(calculated_sim_vector)] = 0.0 
                    
                    sim_for_feature_target_pair_vector = np.where(np.isclose(q_val_scalar, target_vals_for_feature_k), 1.0, calculated_sim_vector)
                    
                    target_bin_indices = np.digitize(target_vals_for_feature_k, current_bin_ranges_val) - 1
                    valid_target_bin_mask = (target_bin_indices >= 0) & \
                                            (target_bin_indices < len(current_histogram_val)) & \
                                            (len(current_histogram_val) > 0)
                    
                    scores_for_current_feature_all_targets = np.where(valid_target_bin_mask, sim_for_feature_target_pair_vector, 0.0)
                    num_contribution_per_target_for_query_i += scores_for_current_feature_all_targets * feature_weight
                
                current_query_total_score_vs_all_targets += num_contribution_per_target_for_query_i
            return current_query_total_score_vs_all_targets

        if num_effective_query_rows > 0:
            # Parallel execution
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(_process_query_row_parallel)(i) for i in range(num_effective_query_rows)
            )
            if results_list: # Ensure results_list is not empty
                 result_matrix = np.array(results_list)
                 # Ensure correct shape if one dimension is 0 or 1
                 if result_matrix.ndim == 1:
                     if num_effective_target_rows == 1:
                         result_matrix = result_matrix.reshape(-1, 1)
                     elif num_effective_query_rows == 1: # Should already be (1, N) if targets > 0
                         result_matrix = result_matrix.reshape(1, -1)
                 # If num_effective_target_rows is 0, result_matrix should be (num_query_rows, 0)
                 # np.array([np.array([]), np.array([])]) gives shape (2,0)
            else: # Should not happen if num_effective_query_rows > 0, but as safeguard
                 result_matrix = np.zeros((num_effective_query_rows, num_effective_target_rows))

        else: # No query rows
            result_matrix = np.zeros((0, num_effective_target_rows))


        if overwrite_memory and original_memorised_distribution is not None:
            self.memorised_distribution = original_memorised_distribution
            self.categorical_features = original_categorical_features
            self.numerical_features = original_numerical_features
            self.numerical_ranges = original_numerical_ranges
            self.cat_counts = original_cat_counts
            self.bin_ranges = original_bin_ranges
            self.histograms = original_histograms
            self.height_diff = original_height_diff
            self._setup_done = original_setup_done 

        if normalised:
            active_features_for_norm = self.categorical_features + self.numerical_features # Use restored self state
            sum_of_weights = sum(self.weights.get(f, 0.0) for f in active_features_for_norm if self.weights.get(f, 0.0) > 0)

            if sum_of_weights > 0:
                result_matrix /= sum_of_weights
            else: 
                result_matrix[result_matrix != 0] = np.nan 
                result_matrix[np.isclose(result_matrix, 0)] = 0.0 
        
        return result_matrix