# Description: Script for calculating the resample exposure index
# Author: Anton D. Lautrup
# Date: 21-05-2025
# Version: 0.1
# License: MIT

import numpy as np

from numpy import ndarray
from typing import Dict, List, Tuple
from pandas import DataFrame, Series

from .utils.preprocessing import get_cat_variables, scott_ref_rule

class ResampleExposure:
    """Class to calculate the resample exposure index between two datasets."""

    def __init__(self, target_distribution: DataFrame, 
                 categorical_features: List[str] = None, 
                 unique_threshold: int = 10):
        """ Initialize the ResampleExposure with a memorised distribution.

        Arguments:
            - target_distribution (DataFrame): The target distribution used for modelling the distributions (unless overwritten).
            - categorical_features (List[str]): List of categorical features in the dataset, 
                if None the categorical features will be determined from the memorised distribution.
            - unique_threshold (int): Threshold for determining if a numerical feature is categorical.
        """
        self.memorised_distribution = target_distribution.copy()
        self.unique_threshold = unique_threshold # Store for potential use in overwrite_memory

        detected_cat_features = get_cat_variables(self.memorised_distribution, unique_threshold)
        if categorical_features is None:
            self.categorical_features = detected_cat_features
        else: # combine detected and provided categorical features
            self.categorical_features = list(set(detected_cat_features) | set(categorical_features))

        self.numerical_features = [col for col in self.memorised_distribution.columns if col not in self.categorical_features]
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
            if np.issubdtype(self.memorised_distribution[col].dtype, np.floating):
                if self.memorised_distribution[col].nunique() > 1:
                    bin_ranges[col] = scott_ref_rule(self.memorised_distribution[col])
                else:
                    bin_ranges[col] = [self.memorised_distribution[col].min(), self.memorised_distribution[col].max()]
            else:
                bin_ranges[col] = np.arange(self.memorised_distribution[col].min(), self.memorised_distribution[col].max() + 1)
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
                height_diff[col] = (np.sum(left_to_right_height_diffs), np.sum(right_to_left_height_diffs))
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

    def compute_resample_exposure_index(self, query_point: Series, target_point: Series, normalised: bool = False) -> float:
        """ Compute the resample exposure index for a given query point and target point.

        Arguments:
            - query_point (Series): A query point to compute the resample exposure index for.
            - target_point (Series): The target point in the memorised distribution to compare against.
            - normalised (bool): If True, the resample exposure index will be normalised to [0, 1].

        Returns:
            - resample exposure index (float): The resample exposure index for the query point to be made into the target point.

        Example:
            >>> query_point = pd.Series({'feature1': 1.0, 'feature2': 5.0})
            >>> target_point = pd.Series({'feature1': 2.0, 'feature2': 7.0})
            >>> resample_exposure_index = ResampleExposure(memorised_distribution)
            >>> index = resample_exposure_index.compute_resample_exposure_index(query_point, target_point)
            >>> print(index)
            0.5
        """
        if not self._setup_done:
            raise ValueError("ResampleExposure is not set up. Call _setup() first.")

        if target_point.shape[0] != self.memorised_distribution.shape[1]:
            raise ValueError("Target point must have the same columns as the memorised distribution.")
        
        if query_point.shape[0] != target_point.shape[0]:
            raise ValueError("Query and target points must have the same columns.")
        
        resample_exposure_index = 0.0

        for col in self.categorical_features:
            target = target_point[col]
            if query_point[col] != target:
                resample_exposure_index += self.cat_counts[col].loc[target] if target in self.cat_counts[col].index else 0
            else:
                resample_exposure_index += 1.0

        for col in self.numerical_features:
            neg_descent = self._get_height_of_histogram_descent(query_point[col], target_point[col], self.bin_ranges[col], self.histograms[col])

            diff_num = query_point[col] - target_point[col]

            height_key = 1 if diff_num < 0 else 0
                
            with np.errstate(divide='ignore', invalid='ignore'):
                sim = (1 - (abs(diff_num) / self.numerical_ranges[col])) * (1-(neg_descent/self.height_diff[col][height_key]))

            bin_index = np.digitize(target_point[col], self.bin_ranges[col]) - 1
            if bin_index < 0 or bin_index >= len(self.histograms[col]) or len(self.histograms[col]) == 0:
                resample_exposure_index += 0.0
            else:
                resample_exposure_index += sim #* self.histograms[col][bin_index] / sum(self.histograms[col])

        if normalised: resample_exposure_index /= len(self.categorical_features) + len(self.numerical_features)
        return resample_exposure_index
    
    def resample_exposure_matrix(self, query_df: DataFrame = None, normalised: bool = False, 
                                 reverse_direction: bool = False, overwrite_memory: bool = False) -> ndarray:
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

        Returns:
            - resample exposure matrix (ndarray): The resample exposure matrix between the two dataframes.

        Example:
            >>> memorised_distribution = pd.DataFrame({'feature1': [1.0, 2.0, 1.0, 3.0], 'feature2': ['A', 'B', 'A', 'A']})
            >>> rex = ResampleExposure(memorised_distribution, categorical_features=['feature2'])
            >>> query_data = pd.DataFrame({'feature1': [1.5], 'feature2': ['A']})
            >>> matrix = rex.resample_exposure_matrix(query_data, normalised=False)
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
            _processed_query_df = query_df.copy() # Use a copy to avoid modifying original query_df

        expected_columns = self.memorised_distribution.columns
        try:
            _processed_query_df = _processed_query_df[expected_columns]
        except KeyError as e:
            raise ValueError(
                f"Query DataFrame is missing one or more columns expected from the memorised distribution: {e}"
            ) from e

        original_memorised_distribution = None
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

            # self.memorised_distribution = _processed_query_df.copy()
            
            detected_cat_features_overwrite = get_cat_variables(self.memorised_distribution, self.unique_threshold)
            # Respect original categorical_features if they were explicitly provided and still exist
            # For simplicity, we re-derive based on the new memorised_distribution,
            # consistent with how __init__ would behave if this was the target_distribution.
            self.categorical_features = [col for col in detected_cat_features_overwrite if col in self.memorised_distribution.columns]
            self.numerical_features = [col for col in self.memorised_distribution.columns if col not in self.categorical_features]
            self._setup()

        effective_query_df: DataFrame
        effective_target_df: DataFrame

        if not reverse_direction:
            effective_query_df = _processed_query_df
            effective_target_df = self.memorised_distribution 
        else: 
            effective_query_df = self.memorised_distribution
            effective_target_df = _processed_query_df
            
        num_effective_query_rows = len(effective_query_df)
        num_effective_target_rows = len(effective_target_df)
        
        result_matrix = np.zeros((num_effective_query_rows, num_effective_target_rows))

        # Precompute categorical probabilities
        cat_probs_for_effective_targets = np.zeros((num_effective_target_rows, len(self.categorical_features)))
        if self.categorical_features:
            et_cat_values_all_local = effective_target_df[self.categorical_features].values
            for k_cat, col_name in enumerate(self.categorical_features):
                if col_name in self.cat_counts:
                    counts = self.cat_counts[col_name] 
                    for j in range(num_effective_target_rows):
                        target_val = et_cat_values_all_local[j, k_cat]
                        cat_probs_for_effective_targets[j, k_cat] = counts.get(target_val, 0.0)
                else:
                    cat_probs_for_effective_targets[:, k_cat] = 0.0
        
        # Precompute negative descent matrices for numerical features
        all_neg_descent_matrices = {}
        if self.numerical_features:
            for col_name_precompute in self.numerical_features:
                if col_name_precompute not in self.histograms or \
                   col_name_precompute not in self.bin_ranges:
                    all_neg_descent_matrices[col_name_precompute] = np.empty((0,0), dtype=float) # Mark as unavailable
                    continue

                current_histogram_pre = self.histograms[col_name_precompute]
                num_bins_for_col = len(current_histogram_pre)

                if num_bins_for_col == 0:
                    all_neg_descent_matrices[col_name_precompute] = np.empty((0,0), dtype=float)
                    continue

                sum_hist = np.sum(current_histogram_pre)
                if sum_hist == 0: # Avoid division by zero if histogram is all zeros
                    all_neg_descent_matrices[col_name_precompute] = np.full((num_bins_for_col, num_bins_for_col), np.nan, dtype=float)
                    continue
                
                bin_heights = current_histogram_pre / sum_hist
                neg_descent_matrix_col = np.zeros((num_bins_for_col, num_bins_for_col), dtype=float)

                for b_q in range(num_bins_for_col): # Query bin index
                    for b_t in range(num_bins_for_col): # Target bin index
                        if b_q == b_t:
                            neg_descent_matrix_col[b_q, b_t] = 0.0
                            continue
                        
                        path_segment: ndarray
                        if b_q < b_t: # Moving from left to right (e.g., b_q=1, b_t=3)
                            # Path is heights[b_q], heights[b_q+1], ..., heights[b_t]
                            path_segment = bin_heights[b_q : b_t + 1]
                        else: # b_q > b_t, Moving from right to left (e.g., b_q=3, b_t=1)
                            # Path is heights[b_q], heights[b_q-1], ..., heights[b_t]
                            # Slice from b_t to b_q (inclusive) and then reverse it
                            path_segment = bin_heights[b_t : b_q + 1][::-1]
                        
                        if len(path_segment) < 2: # Should not happen if b_q != b_t and num_bins_for_col > 0
                            diffs = np.array([])
                        else:
                            diffs = np.diff(path_segment) # Differences along the path from b_q to b_t
                        
                        neg_diffs = diffs[diffs < 0] # Keep only negative differences (descents)
                        neg_descent_matrix_col[b_q, b_t] = np.sum(np.abs(neg_diffs))
                all_neg_descent_matrices[col_name_precompute] = neg_descent_matrix_col

        eq_cat_values_all = effective_query_df[self.categorical_features].values if self.categorical_features else np.empty((num_effective_query_rows, 0))
        eq_num_values_all = effective_query_df[self.numerical_features].values if self.numerical_features else np.empty((num_effective_query_rows, 0))
        
        et_cat_values_for_comp = effective_target_df[self.categorical_features].values if self.categorical_features else np.empty((num_effective_target_rows, 0))
        et_num_values_for_comp = effective_target_df[self.numerical_features].values if self.numerical_features else np.empty((num_effective_target_rows, 0))

        for i in range(num_effective_query_rows): 
            current_query_total_score_vs_all_targets = np.zeros(num_effective_target_rows)

            if self.categorical_features and num_effective_target_rows > 0:
                query_cat_row_vals = eq_cat_values_all[i, :] 
                matches = (query_cat_row_vals == et_cat_values_for_comp)
                cat_contribution_per_target = np.zeros(num_effective_target_rows)
                for k_cat in range(len(self.categorical_features)):
                    feature_k_cat_scores = np.where(matches[:, k_cat], 1.0, cat_probs_for_effective_targets[:, k_cat])
                    cat_contribution_per_target += feature_k_cat_scores
                current_query_total_score_vs_all_targets += cat_contribution_per_target

            if self.numerical_features and num_effective_target_rows > 0:
                query_num_row_vals = eq_num_values_all[i, :]
                num_contribution_per_target_for_query_i = np.zeros(num_effective_target_rows)

                for k_num, col_name in enumerate(self.numerical_features):
                    if col_name not in self.numerical_ranges or \
                       col_name not in self.bin_ranges or \
                       col_name not in self.histograms or \
                       col_name not in self.height_diff or \
                       col_name not in all_neg_descent_matrices or \
                       all_neg_descent_matrices[col_name].shape[0] == 0: # Check if precomputation was skipped or resulted in empty
                        continue 

                    q_val_scalar = query_num_row_vals[k_num]
                    target_vals_for_feature_k = et_num_values_for_comp[:, k_num] 
                    
                    current_feature_range = self.numerical_ranges[col_name]
                    current_bin_ranges = self.bin_ranges[col_name]
                    current_histogram = self.histograms[col_name] # Used for target_bin_index check
                    current_height_diff_tuple = self.height_diff[col_name]
                    
                    precomputed_descent_matrix_col = all_neg_descent_matrices[col_name]
                    num_bins_for_col_runtime = len(current_histogram)


                    scores_for_current_feature_all_targets = np.zeros(num_effective_target_rows, dtype=float)

                    for j_target in range(num_effective_target_rows):
                        t_val_scalar = target_vals_for_feature_k[j_target]
                        sim_for_feature_target_pair: float

                        if np.isclose(q_val_scalar, t_val_scalar):
                            sim_for_feature_target_pair = 1.0
                        else:
                            diff_num_scalar = q_val_scalar - t_val_scalar
                            neg_descent_scalar: float
                            if num_bins_for_col_runtime == 0: # Should align with precomputation skip
                                neg_descent_scalar = 0.0 if np.sum(current_histogram) != 0 else np.nan
                            else:
                                bin_idx_q = np.digitize(q_val_scalar, current_bin_ranges) - 1
                                bin_idx_t = np.digitize(t_val_scalar, current_bin_ranges) - 1
                                

                                bin_idx_q_clipped = np.clip(bin_idx_q, 0, num_bins_for_col_runtime - 1)
                                bin_idx_t_clipped = np.clip(bin_idx_t, 0, num_bins_for_col_runtime - 1)
                                neg_descent_scalar = precomputed_descent_matrix_col[bin_idx_q_clipped, bin_idx_t_clipped]
                            

                            height_key_scalar = 1 if diff_num_scalar < 0 else 0
                            
                            with np.errstate(divide='ignore', invalid='ignore'):
                                term1 = (1.0 - (abs(diff_num_scalar) / current_feature_range))
                                term2_numerator = neg_descent_scalar
                                term2_denominator = current_height_diff_tuple[height_key_scalar]
                                

                                if term2_denominator == 0: # Avoid 0/0 or x/0 if neg_descent is also 0 or non-zero
                                    term2 = 1.0 if term2_numerator == 0 else 0.0 # if descent is 0, factor is 1. If descent non-0 and max_descent 0, factor is 0.
                                else:
                                    term2 = (1.0 - (term2_numerator / term2_denominator))

                                calculated_sim = term1 * term2
                                if np.isnan(calculated_sim): calculated_sim = 0.0 # Default for NaN results from divisions
                            sim_for_feature_target_pair = float(calculated_sim)
                        
                        target_bin_index = np.digitize(t_val_scalar, current_bin_ranges) - 1
                        if target_bin_index < 0 or target_bin_index >= len(current_histogram) or len(current_histogram) == 0:
                            scores_for_current_feature_all_targets[j_target] = 0.0
                        else:
                            scores_for_current_feature_all_targets[j_target] = sim_for_feature_target_pair
                    

                    num_contribution_per_target_for_query_i += scores_for_current_feature_all_targets
                current_query_total_score_vs_all_targets += num_contribution_per_target_for_query_i
            
            result_matrix[i, :] = current_query_total_score_vs_all_targets

        if overwrite_memory and original_memorised_distribution is not None:
            self.memorised_distribution = original_memorised_distribution
            self.categorical_features = original_categorical_features
            self.numerical_features = original_numerical_features
            self.numerical_ranges = original_numerical_ranges
            self.cat_counts = original_cat_counts
            self.bin_ranges = original_bin_ranges
            self.histograms = original_histograms
            self.height_diff = original_height_diff
            self._setup_done = original_setup_done # Restore setup status

        if normalised:
            num_total_features = len(self.categorical_features) + len(self.numerical_features)
            if num_total_features > 0:
                result_matrix /= num_total_features
        
        return result_matrix