# Description: Script for holding preprocessing functions
# Author: Anton D. Lautrup
# Date: 21-05-2025

import numpy as np

from typing import List
from pandas import DataFrame

def get_cat_variables(df: DataFrame, threshold: int) -> List[str]:
    """Function to get the categorical variables in a dataframe.

    Arguments:
        df (DataFrame): The dataframe to check for categorical variables.
        threshold (int): The threshold for determining if a numerical feature is categorical.

    Returns:
        list: A list of categorical variable names.
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [1.0, 2.0, 3.0]})
        >>> cat_vars = get_cat_variables(df, 2)
        >>> print(cat_vars)
        ['B']
    """
    cat_variables = []

    for col in df.columns:
        if df[col].dtype == "object":
            cat_variables.append(col)
        elif (
            np.issubdtype(df[col].dtype, np.integer) or np.issubdtype(df[col].dtype, np.floating)
        ) and df[col].nunique() < threshold:
            cat_variables.append(col)
    return cat_variables

def scott_ref_rule(samples: List[float]) -> List[float]:
    """Function for doing the Scott reference rule to calcualte number of bins needed to 
    represent the nummerical values.
    
    Args:
        samples (array-like) : The data to be binned.
    
    Returns:
        array : bin edges
    
    Example:
        >>> scott_ref_rule([1,2,3,4,5])
        array([1., 2., 3., 4., 5.])
    """
    n = len(samples)

    if n == 0:
        return np.array([])
    
    samples_np = np.asarray(samples) # Ensure it's a numpy array for calculations

    min_edge = np.min(samples_np)
    max_edge = np.max(samples_np)

    # If all samples are identical (or only one sample), std and iqr would be 0.
    # Create a single bin covering this value.
    if min_edge == max_edge:
        return np.array([min_edge, max_edge]) 

    std = np.std(samples_np, ddof=1) 
    q75, q25 = np.percentile(samples_np, [75, 25])
    iqr = q75 - q25

    if iqr < 1e-9:  # IQR is zero or very close to zero, potential division by zero.
        # Fallback: use a common heuristic sqrt(n) for the number of bins.
        N = int(np.ceil(np.sqrt(n)))
    else:
        denominator_scott = 3.5 * iqr
        numerator_scott = (n**(1.0/3.0)) * std
        
        # Calculate the intermediate factor
        # If std is 0 (and iqr > 0), numerator_scott is 0, factor is 0.
        # If std > 0 and iqr > 0, argument to ceil is > 0, so ceil >= 1.
        intermediate_factor = np.ceil(numerator_scott / denominator_scott).astype(int)

        if intermediate_factor <= 0:
            # This can happen if std is 0 (numerator is 0).
            # Safeguard: ensure the factor is at least 1 to prevent division by zero or N=0 later.
            intermediate_factor = 1
        
        # Calculate N using this factor.
        # The int() truncates, so if (max_edge - min_edge) / intermediate_factor is < 1, N could be 0.
        N_candidate = (max_edge - min_edge) / intermediate_factor
        N = int(N_candidate)

    N = max(1, N)  # Ensure N is at least 1.
    N = min(N, 10000)  # Apply cap on the number of bins
    Nplus1 = N + 1
    
    return np.linspace(min_edge, max_edge, Nplus1)