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
                bin_ranges[col] = np.arange(self.memorised_distribution[col].nunique() + 1)
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
            abs_diff_num = np.abs(query_point[col] - target_point[col])

            with np.errstate(divide='ignore', invalid='ignore'):
                sim = 1 - (abs_diff_num / self.numerical_ranges[col])

            bin_index = np.digitize(target_point[col], self.bin_ranges[col]) - 1
            if bin_index < 0 or bin_index >= len(self.histograms[col]) or len(self.histograms[col]) == 0:
                resample_exposure_index += 0.0
            else:
                resample_exposure_index += sim * self.histograms[col][bin_index] / sum(self.histograms[col])

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

        # _processed_query_df is the DataFrame that is not self.memorised_distribution,
        # or a copy of self.memorised_distribution if query_df is None.
        _processed_query_df: DataFrame
        if query_df is None:
            _processed_query_df = self.memorised_distribution.copy() 
        else:
            _processed_query_df = query_df

        # Ensure _processed_query_df has the same columns in the same order as self.memorised_distribution
        expected_columns = self.memorised_distribution.columns
        try:
            _processed_query_df = _processed_query_df[expected_columns]
        except KeyError as e:
            raise ValueError(
                f"Query DataFrame is missing one or more columns expected from the memorised distribution: {e}"
            ) from e

        effective_query_df: DataFrame
        effective_target_df: DataFrame

        if not reverse_direction:
            effective_query_df = _processed_query_df
            effective_target_df = self.memorised_distribution 
        else: # reverse_direction is True
            effective_query_df = self.memorised_distribution
            effective_target_df = _processed_query_df
            
        num_effective_query_rows = len(effective_query_df)
        num_effective_target_rows = len(effective_target_df)
        
        result_matrix = np.zeros((num_effective_query_rows, num_effective_target_rows))

        # Precompute probabilities for values in effective_target_df, using stats from self.memorised_distribution.
        if overwrite_memory:
            self.memorised_distribution = _processed_query_df.copy()
            self._setup()

        # For Categorical Features:
        # cat_probs_for_effective_targets[target_row_idx, cat_feature_idx] = P_memorised(effective_target_df.iloc[target_row_idx][cat_feature])
        cat_probs_for_effective_targets = np.zeros((num_effective_target_rows, len(self.categorical_features)))
        if self.categorical_features:
            et_cat_values_all = effective_target_df[self.categorical_features].values
            for k_cat, col_name in enumerate(self.categorical_features):
                counts = self.cat_counts[col_name] # From self.memorised_distribution
                for j in range(num_effective_target_rows):
                    target_val = et_cat_values_all[j, k_cat]
                    cat_probs_for_effective_targets[j, k_cat] = counts.get(target_val, 0.0)

        # For Numerical Features:
        # num_hist_probs_for_effective_targets[target_row_idx, num_feature_idx] = P_memorised(bin of effective_target_df.iloc[target_row_idx][num_feature])
        num_hist_probs_for_effective_targets = np.zeros((num_effective_target_rows, len(self.numerical_features)))
        if self.numerical_features:
            et_num_values_all = effective_target_df[self.numerical_features].values
            for k_num, col_name in enumerate(self.numerical_features):
                hist = self.histograms[col_name] # From self.memorised_distribution
                bins = self.bin_ranges[col_name] # From self.memorised_distribution
                sum_hist = np.sum(hist)

                if sum_hist == 0 or len(hist) == 0:
                    num_hist_probs_for_effective_targets[:, k_num] = 0.0
                    continue
                
                for j in range(num_effective_target_rows):
                    target_val = et_num_values_all[j, k_num]
                    bin_index = np.digitize(target_val, bins) - 1
                    if 0 <= bin_index < len(hist):
                        num_hist_probs_for_effective_targets[j, k_num] = hist[bin_index] / sum_hist
                    else:
                        num_hist_probs_for_effective_targets[j, k_num] = 0.0
        
        # --- Main calculation loop over effective_query_df rows ---
        eq_cat_values_all = effective_query_df[self.categorical_features].values if self.categorical_features else np.empty((num_effective_query_rows, 0))
        eq_num_values_all = effective_query_df[self.numerical_features].values if self.numerical_features else np.empty((num_effective_query_rows, 0))
        
        # Values from effective_target_df are also needed for direct comparison (matches, abs_diff)
        et_cat_values_for_comp = effective_target_df[self.categorical_features].values if self.categorical_features else np.empty((num_effective_target_rows, 0))
        et_num_values_for_comp = effective_target_df[self.numerical_features].values if self.numerical_features else np.empty((num_effective_target_rows, 0))

        for i in range(num_effective_query_rows): # Iterate over rows of effective_query_df
            current_query_total_score_vs_all_targets = np.zeros(num_effective_target_rows)

            # Categorical features contribution
            if self.categorical_features:
                query_cat_row_vals = eq_cat_values_all[i, :] 
                matches = (query_cat_row_vals == et_cat_values_for_comp)
                
                cat_contribution_per_target = np.zeros(num_effective_target_rows)
                for k_cat in range(len(self.categorical_features)):
                    feature_k_cat_scores = np.where(matches[:, k_cat], 1.0, cat_probs_for_effective_targets[:, k_cat])
                    cat_contribution_per_target += feature_k_cat_scores
                current_query_total_score_vs_all_targets += cat_contribution_per_target

            # Numerical features contribution
            if self.numerical_features:
                query_num_row_vals = eq_num_values_all[i, :]
                num_contribution_per_target = np.zeros(num_effective_target_rows)

                for k_num, col_name in enumerate(self.numerical_features):
                    q_val = query_num_row_vals[k_num]
                    t_vals_col = et_num_values_for_comp[:, k_num] 
                    
                    abs_diff_num = np.abs(q_val - t_vals_col)
                    feature_range = self.numerical_ranges[col_name] # From self.memorised_distribution

                    # Determine where query and target values are identical for this feature
                    # np.isclose is used for robust floating-point comparison.
                    is_identical = np.isclose(q_val, t_vals_col)

                    # This sim_calc is used only when q_val and t_val are not identical.
                    # If feature_range is close to 0 (e.g., all values in memorised dist for this feature were the same),
                    # and q_val is not t_val, sim_calc will be 0.0.
                    sim_calc = 1.0 - np.divide(abs_diff_num, feature_range, out=np.full_like(abs_diff_num, np.nan, dtype=float), where=~np.isclose(feature_range, 0.0))
                    sim_calc = np.nan_to_num(sim_calc, nan=0.0) # Handles NaN from division by zero if range was ~0.
                    sim_calc = np.clip(sim_calc, 0.0, 1.0)

                    hist_probs_for_col = num_hist_probs_for_effective_targets[:, k_num]
                    
                    # Calculate scores for non-identical cases: sim * probability
                    scores_if_not_identical = sim_calc #* hist_probs_for_col
                    
                    feature_k_num_scores = np.where(is_identical, 1.0, scores_if_not_identical)
                    num_contribution_per_target += feature_k_num_scores
                current_query_total_score_vs_all_targets += num_contribution_per_target
            
            result_matrix[i, :] = current_query_total_score_vs_all_targets

        if normalised:
            num_total_features = len(self.categorical_features) + len(self.numerical_features)
            if num_total_features > 0:
                result_matrix /= num_total_features
        
        return result_matrix