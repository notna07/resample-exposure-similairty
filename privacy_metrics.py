# Description: Script for holding the privacy metrics integrated with the KNNAdapter.
# Author: Anton D. Lautrup
# Date: 23-05-2025

import pandas as pd

from pandas import DataFrame

from KNN_adapters import KNNAdapter

def distance_to_closest_record(queries: DataFrame, targets: DataFrame, adapter: KNNAdapter) -> float:
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

def epsilon_identifiability_risk(queries: DataFrame, targets: DataFrame, adapter: KNNAdapter) -> float:
    """
    Calculate the epsilon identifiability risk for the given queries and targets using the provided KNNAdapter.
    
    Arguments:
        queries (DataFrame): DataFrame containing query records.
        targets (DataFrame): DataFrame containing target records.
        adapter (KNNAdapter): An instance of KNNAdapter to compute distances.
    Returns:
        float: The epsilon identifiability risk.
    """
    

    # knn_model = adapter()
    # # Ensure the adapter is fitted to the targets
    # knn_model.fit_nn(targets)

    # if queries.equals(targets):
    #     #find the next-nearest neighbor
    #     dists = knn_model.get_neighbors(queries, n_neighbors=2)[0]
    #     dists = dists[:, 1]  # Exclude the first column which is the distance to itself
    # else:
    #     #find the nearest neighbor
    #     dists = knn_model.get_neighbors(queries, n_neighbors=1)[0]

    # res_df = pd.DataFrame(dists, columns=['DCR'])
    # res = res_df.aggregate(
    #     ['mean', 'std', 'min', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75),'max']
    # )
    # res.index = ['mean', 'std', 'min', '25%', 'median', '75%', 'max']
    # return res.T