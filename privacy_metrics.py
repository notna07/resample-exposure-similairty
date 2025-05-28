# Description: Script for holding the privacy metrics integrated with the KNNAdapter.
# Author: Anton D. Lautrup
# Date: 23-05-2025

import numpy as np
import pandas as pd

from pandas import DataFrame
from KNN_adapters import KNNAdapter

from scipy.stats import entropy

def _column_entropy(labels):
    """ Compute the entropy of a column of data
    Args:
        labels (np.array): A column of data

    Returns:
        float: The entropy of the column
    
    Example:
        >>> import numpy as np
        >>> ent = _column_entropy(np.array([1, 1, 2, 2, 3, 3]))
        >>> isinstance(ent, float)
        True
    """
    # If the column data type is floating point, round it to discretize.
    if np.issubdtype(labels.dtype, np.floating):
        processed_labels = np.round(labels)
    # For other types (integer, string, object), convert to string to ensure
    # np.unique can handle mixed types within object arrays or pure string arrays.
    else:
        processed_labels = labels.astype(str) 
    
    value, counts = np.unique(processed_labels, return_counts=True)
    return entropy(counts)

def distance_to_closest_record(queries: DataFrame, targets: DataFrame, adapter: KNNAdapter) -> DataFrame:
    """
    For each query, find the closest record in the targets DataFrame using the provided KNNAdapter.
    
    Arguments:
        queries (DataFrame): DataFrame containing query records.
        targets (DataFrame): DataFrame containing target records.
        adapter (KNNAdapter): An instance of KNNAdapter to compute distances.
    Returns:
        float: The average distance to the closest record in targets for each query.
    """
    knn_model = adapter()
    # Ensure the adapter is fitted to the targets
    knn_model.fit_nn(targets)

    if queries.equals(targets):
        #find the next-nearest neighbor
        dists = knn_model.get_neighbors(queries, n_neighbors=2)[0]
        dists = dists[:, 1]  # Exclude the first column which is the distance to itself
    else:
        #find the nearest neighbor
        dists = knn_model.get_neighbors(queries, n_neighbors=1)[0]

    res_df = pd.DataFrame(dists, columns=['DCR'])
    res = res_df.aggregate(
        ['mean', 'std', 'min', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75),'max']
    )
    res.index = ['mean', 'std', 'min', '25%', 'median', '75%', 'max']
    return res.T

def epsilon_identifiability_risk(queries: DataFrame, targets: DataFrame, adapter: KNNAdapter) -> DataFrame:
    """
    Calculate the epsilon identifiability risk for the given queries and targets using the provided KNNAdapter.
    
    Arguments:
        queries (DataFrame): DataFrame containing query records.
        targets (DataFrame): DataFrame containing target records.
        adapter (KNNAdapter): An instance of KNNAdapter to compute distances.
    Returns:
        float: The epsilon identifiability risk.
    """    
    real_dist = np.asarray(targets.copy())

    no, x_dim = np.shape(real_dist)
    W = [_column_entropy(real_dist[:, i]) for i in range(x_dim)]
    W_adjust = 1/(np.array(W)+1e-16)

    knn_model = adapter(weights=W_adjust)
    knn_model.fit_nn(targets)
    in_dists = knn_model.get_neighbors(targets, n_neighbors=2)[0]

    knn_model.fit_nn(queries)
    ext_dists = knn_model.get_neighbors(targets, n_neighbors=1)[0]

    R_Diff = ext_dists[:,0] - in_dists[:,1]

    identifiability_value = np.sum(R_Diff < 0) / float(no)

    return identifiability_value
