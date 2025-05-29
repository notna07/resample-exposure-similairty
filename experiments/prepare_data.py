# Description: Script to hold utility functions
# Author: Anton D. Lautrup
# Date: 20-05-2025

import numpy as np

from pandas import DataFrame
from typing import List, Tuple

from ucimlrepo import fetch_ucirepo

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def uci_dataset_id_import(dataset_id: int, silent_import: bool = True) -> DataFrame:
    """ Import a dataset from the UCI repository using its ID. """
    ds_obj = fetch_ucirepo(id=dataset_id)

    df = ds_obj.data.features.copy()
    df['class'] = ds_obj.data.targets

    if not silent_import: print(f"Dataset Name: {ds_obj.metadata.name}")
    return df


def encode_categorical_features(df: DataFrame, cat_features: List[str]) -> DataFrame:
    encoder = OrdinalEncoder()
    df[cat_features] = encoder.fit_transform(df[cat_features])
    return df

def get_categorical_features(df: DataFrame, unique_threshold: int = 10) -> List[str]:
    cat_variables = []

    for col in df.columns:
        if df[col].dtype == "object":
            cat_variables.append(col)
        elif df[col].dtype == "bool": # Add this condition to detect boolean types
            cat_variables.append(col)
        elif (
            np.issubdtype(df[col].dtype, np.integer) or np.issubdtype(df[col].dtype, np.floating)
        ) and df[col].nunique() < unique_threshold:
            cat_variables.append(col)
    return cat_variables

def _standardise_numerical_features(df: DataFrame, cat_feats: List[str]) -> DataFrame:
    """ Standardize numerical features in the DataFrame. """
    num_feats = [feat for feat in df.columns if feat not in cat_feats]
    if len(num_feats) == 0:
        return df

    scaler = StandardScaler()
    df[num_feats] = scaler.fit_transform(df[num_feats])
    return df

def _get_dataset_type(df: DataFrame, cat_feats: List[str]) -> str:
    """ Determine the type of dataset based on the number of categorical features. """
    num_cols = len(list(df.columns))

    if len(cat_feats)/num_cols > 0.75:
        return 'most_cats'
    elif len(cat_feats)/num_cols < 0.25:
        return 'most_nums'
    else:
        return 'balanced'
    
def _remove_index_columns(df: DataFrame) -> DataFrame:
    """ Remove index columns from the DataFrame if the column name exactly matches an identifier (case-insensitively). """
    identifier_names = ['id', 'ID', 'Id', 'index', 'Unnamed: 0','timestamp', 'name','spread1','date']
    # Create a set of lowercase identifiers for efficient lookup
    identifier_names_lower = {name.lower() for name in identifier_names}
    
    # Identify columns to remove by checking for exact match (case-insensitive)
    cols_to_remove = [col for col in df.columns if col.lower() in identifier_names_lower]
    
    return df.drop(columns=cols_to_remove)

def _clean_up_data(df: DataFrame, label: str) -> DataFrame:
    """ Clean the dataset by removing rows with missing values. """
    df = df.dropna(axis=0)
    df = _remove_index_columns(df)
    df = df.rename(columns={label: 'class'})
    return df

def _remove_severly_underrepresented_classes(df: DataFrame, min_samples: int = 5) -> DataFrame:
    """ Remove class instances with fewer than a specified number of samples. """
    class_counts = df['class'].value_counts()
    classes_to_remove = class_counts[class_counts < min_samples].index
    if len(classes_to_remove) > 0:
        df = df[~df['class'].isin(classes_to_remove)]
    return df

def preprocess_data(df: DataFrame, label: str) -> Tuple[DataFrame, List[str], str]:
    """ Clean the dataset by removing rows with missing values. """
    cat_feats = get_categorical_features(df)
    metadata = _get_dataset_type(df, cat_feats)
    df = encode_categorical_features(df, cat_feats)
    df = _standardise_numerical_features(df, cat_feats)
    df = _clean_up_data(df, label)
    df = _remove_severly_underrepresented_classes(df)
    return df, cat_feats, metadata

def scott_ref_rule(samples: List[float]) -> List[float]:
    """Function for doing the Scott reference rule to calcualte number of bins needed to 
    represent the nummerical values.
    
    Args:
        samples (array-like) : The data to be binned.
    
    Returns:
        array : bin edges
    
    Example:
        >>> _scott_ref_rule([1,2,3,4,5])
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
    N = min(N, 1000)  # Apply cap on the number of bins
    Nplus1 = N + 1
    
    return np.linspace(min_edge, max_edge, Nplus1)