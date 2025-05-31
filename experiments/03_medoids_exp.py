# Description: Script holding the experimenter module for the KMedoids clustering algorithm
# Author: Hafiz Saud Arshad
# Date: 30-05-2025

import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from typing import List

from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from experiments.prepare_data import get_categorical_features, replace_features_with_gmm_category
from experiments.data_generators import (data_generator21, data_generator25, data_generator17,
                                         data_generator10, data_generator14, data_generator12,
                                         data_generator6, data_generator7, data_generator5)

### Results Data Structure
Dataset_Scores = {  # For Each Measure
    'Large' :  {
        'Reals'   : {},
        'Balanced': {},
        'Cats'    : {}
    },
    'Medium' : {
        'Reals'   : {},
        'Balanced': {},
        'Cats'    : {}
    },
    'Small' :  {
        'Reals'   : {},
        'Balanced': {},
        'Cats'    : {}
    }
}

ARIs = Dataset_Scores
NMIs = Dataset_Scores

# Combined Dict
All_Scores_dict = {
    'ARI' : ARIs,
    'NMI' : NMIs
}

#region ### Generate the datasets for the experiments
Large_Balanced_X, Large_Balanced_y = data_generator21(n_blobs=3, n_moons=2, dim=21, n_samples_per_cluster=400)
Large_Balanced_X, Large_Balanced_y = replace_features_with_gmm_category(Large_Balanced_X, Large_Balanced_y, 2, 3, n_components=6)
Large_Balanced_X, Large_Balanced_y = replace_features_with_gmm_category(Large_Balanced_X, Large_Balanced_y, 2, 3, n_components=5)
Large_Balanced_X, Large_Balanced_y = replace_features_with_gmm_category(Large_Balanced_X, Large_Balanced_y, 2, 3, n_components=4)
Large_Balanced_X, Large_Balanced_y = replace_features_with_gmm_category(Large_Balanced_X, Large_Balanced_y, 2, 3, n_components=3)
Large_Balanced_X, Large_Balanced_y = replace_features_with_gmm_category(Large_Balanced_X, Large_Balanced_y, 2, 3, n_components=2)
Large_Balanced_X, Large_Balanced_y = replace_features_with_gmm_category(Large_Balanced_X, Large_Balanced_y, 2, 3, n_components=7)
Large_Balanced_X, Large_Balanced_y = replace_features_with_gmm_category(Large_Balanced_X, Large_Balanced_y, 2, 3, n_components=3)
Large_Balanced_X = pd.DataFrame(Large_Balanced_X)

Large_Reals_X, Large_Reals_y = data_generator17(n_blobs=3, n_moons=2, dim=17, n_samples_per_cluster=400)
Large_Reals_X, Large_Reals_y = replace_features_with_gmm_category(Large_Reals_X, Large_Reals_y, 5, 6, n_components=3)
Large_Reals_X, Large_Reals_y = replace_features_with_gmm_category(Large_Reals_X, Large_Reals_y, 2, 3, n_components=4)
Large_Reals_X, Large_Reals_y = replace_features_with_gmm_category(Large_Reals_X, Large_Reals_y, 2, 3, n_components=5)
Large_Reals_X = pd.DataFrame(Large_Reals_X)

Large_Cats_X, Large_Cats_y = data_generator25(n_blobs=3, n_moons=2, dim=25, n_samples_per_cluster=400)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=2)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=3)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=4)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=5)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=6)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=7)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=8)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=2)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=3)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=4)
Large_Cats_X, Large_Cats_y = replace_features_with_gmm_category(Large_Cats_X, Large_Cats_y, 1, 2, n_components=5)
Large_Cats_X = pd.DataFrame(Large_Cats_X)

Medium_Balanced_X, Medium_Balanced_y = data_generator12(n_blobs=3, n_moons=2, dim=12, n_samples_per_cluster=200)
Medium_Balanced_X, Medium_Balanced_y = replace_features_with_gmm_category(Medium_Balanced_X, Medium_Balanced_y, 1, 2, n_components=6)
Medium_Balanced_X, Medium_Balanced_y = replace_features_with_gmm_category(Medium_Balanced_X, Medium_Balanced_y, 1, 2, n_components=3)
Medium_Balanced_X, Medium_Balanced_y = replace_features_with_gmm_category(Medium_Balanced_X, Medium_Balanced_y, 1, 2, n_components=4)
Medium_Balanced_X, Medium_Balanced_y = replace_features_with_gmm_category(Medium_Balanced_X, Medium_Balanced_y, 1, 2, n_components=5)
Medium_Balanced_X = pd.DataFrame(Medium_Balanced_X)

Medium_Reals_X, Medium_Reals_y = data_generator10(n_blobs=3, n_moons=2, dim=10, n_samples_per_cluster=200)
Medium_Reals_X, Medium_Reals_y = replace_features_with_gmm_category(Medium_Reals_X, Medium_Reals_y, 1, 2, n_components=3)
Medium_Reals_X, Medium_Reals_y = replace_features_with_gmm_category(Medium_Reals_X, Medium_Reals_y, 1, 2, n_components=5)
Medium_Reals_X = pd.DataFrame(Medium_Reals_X)

Medium_Cats_X, Medium_Cats_y = data_generator14(n_blobs=3, n_moons=2, dim=14, n_samples_per_cluster=200)
Medium_Cats_X, Medium_Cats_y = replace_features_with_gmm_category(Medium_Cats_X, Medium_Cats_y, 1, 2, n_components=2)
Medium_Cats_X, Medium_Cats_y = replace_features_with_gmm_category(Medium_Cats_X, Medium_Cats_y, 1, 2, n_components=3)
Medium_Cats_X, Medium_Cats_y = replace_features_with_gmm_category(Medium_Cats_X, Medium_Cats_y, 1, 2, n_components=4)
Medium_Cats_X, Medium_Cats_y = replace_features_with_gmm_category(Medium_Cats_X, Medium_Cats_y, 1, 2, n_components=5)
Medium_Cats_X, Medium_Cats_y = replace_features_with_gmm_category(Medium_Cats_X, Medium_Cats_y, 1, 2, n_components=3)
Medium_Cats_X, Medium_Cats_y = replace_features_with_gmm_category(Medium_Cats_X, Medium_Cats_y, 1, 2, n_components=5)
Medium_Cats_X = pd.DataFrame(Medium_Cats_X)

Small_Balanced_X, Small_Balanced_y = data_generator6(n_blobs=3, n_moons=2, dim=6, n_samples_per_cluster=100)
Small_Balanced_X, Small_Balanced_y = replace_features_with_gmm_category(Small_Balanced_X, Small_Balanced_y, 1, 2, n_components=3)
Small_Balanced_X, Small_Balanced_y = replace_features_with_gmm_category(Small_Balanced_X, Small_Balanced_y, 1, 2, n_components=5)
Small_Balanced_X = pd.DataFrame(Small_Balanced_X)

Small_Reals_X, Small_Reals_y = data_generator5(n_blobs=3, n_moons=2, dim=5, n_samples_per_cluster=100)
Small_Reals_X, Small_Reals_y = replace_features_with_gmm_category(Small_Reals_X, Small_Reals_y, 1, 2, n_components=3)
Small_Reals_X = pd.DataFrame(Small_Reals_X)

Small_Cats_X, Small_Cats_y = data_generator7(n_blobs=3, n_moons=2, dim=7, n_samples_per_cluster=100)
Small_Cats_X, Small_Cats_y = replace_features_with_gmm_category(Small_Cats_X, Small_Cats_y, 1, 2, n_components=3)
Small_Cats_X, Small_Cats_y = replace_features_with_gmm_category(Small_Cats_X, Small_Cats_y, 1, 2, n_components=4)
Small_Cats_X, Small_Cats_y = replace_features_with_gmm_category(Small_Cats_X, Small_Cats_y, 1, 2, n_components=5)
Small_Cats_X = pd.DataFrame(Small_Cats_X)
#endregion 

def run_kmedoids_multiple_seeds(X: ndarray, n_clusters: int, seeds: List[int], metric: str = 'euclidean') -> List[ndarray]:
    """Run KMedoids with multiple seeds and return list of labels"""
    labels_list = []
    
    for seed in seeds:
        kmedoids = KMedoids(
            n_clusters=n_clusters, 
            random_state=seed, 
            metric=metric,
            init='random' if metric == 'precomputed' else 'random'
        )
        kmedoids.fit(X)
        labels_list.append(kmedoids.labels_)
    
    return labels_list

def eucledian_kmedoids_multiple(df: DataFrame, n_clusters: int = 4, seeds: List[int] = range(30, 80)) -> List[ndarray]:
    """Euclidean KMedoids on original data with multiple seeds"""
    return run_kmedoids_multiple_seeds(df.values, n_clusters, seeds, 'euclidean')

def eucledian_kmedoids_OHE_multiple(df: DataFrame, n_clusters: int = 4, seeds: List[int] = range(30, 80)) -> List[ndarray]:
    """Euclidean KMedoids on One-Hot Encoded data with multiple seeds"""
    df = df.copy()
    cat_feats = get_categorical_features(df)
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_ohe = ohe.fit_transform(df[cat_feats])
    df_ohe = np.hstack((df_ohe, df.drop(columns=cat_feats).values))
    
    return run_kmedoids_multiple_seeds(df_ohe, n_clusters, seeds, 'euclidean')

def gower_kmedoids_multiple(df: DataFrame, n_clusters: int = 4, seeds: List[int] = range(30, 80)) -> List[ndarray]:
    """Gower distance KMedoids with multiple seeds"""
    import gower
    gower_matrix = gower.gower_matrix(df)
    return run_kmedoids_multiple_seeds(gower_matrix, n_clusters, seeds, 'precomputed')

def gem_kmedoids_multiple(df: DataFrame, n_clusters: int = 4, seeds: List[int] = range(30, 80)) -> List[ndarray]:
    """GEM (Ichino-Yaguchi) distance KMedoids with multiple seeds"""
    from implemented_distances import ichino_yaguchi_distance_matrix
    distance_matrix = ichino_yaguchi_distance_matrix(df)
    return run_kmedoids_multiple_seeds(distance_matrix, n_clusters, seeds, 'precomputed')

def heom_kmedoids_multiple(df: DataFrame, n_clusters: int = 4, seeds: List[int] = range(30, 80)) -> List[ndarray]:
    """HEOM distance KMedoids with multiple seeds"""
    from implemented_distances import heom_distance_matrix
    distance_matrix = heom_distance_matrix(df)
    return run_kmedoids_multiple_seeds(distance_matrix, n_clusters, seeds, 'precomputed')

def resample_exposure_kmedoids_multiple(df: DataFrame, n_clusters: int = 4, seeds: List[int] = range(30, 80)) -> List[ndarray]:
    """Resample Exposure KMedoids with multiple seeds"""
    from rex_score.resample_exposure import ResampleExposure
    rex = ResampleExposure(df)
    exposure_matrix = rex.resample_exposure_matrix(normalised=True, reverse_direction=True)
    exposure_matrix = np.ones_like(exposure_matrix) - exposure_matrix
    
    return run_kmedoids_multiple_seeds(exposure_matrix.T, n_clusters, seeds, 'precomputed')

# Function to compute NMI
def compute_nmis(true_labels, pred_labels_list):
    nmis = []
    for pred_labels in pred_labels_list:
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        nmis.append(nmi)
    return nmis

# Function to compute  ARI
def compute_aris(true_labels, pred_labels_list):
    aris = []
    for pred_labels in pred_labels_list:
        ari = adjusted_rand_score(true_labels, pred_labels)
        aris.append(ari)
    return aris

# Run all Clusterings, Compute Scores and return dictionaries
def compute_scores(X_new, y, n_clusters = 5):
    
    # Run each algorithm with multiple seeds
    gem_labels_list = gem_kmedoids_multiple(X_new, n_clusters=n_clusters, )
    euc_labels_list = eucledian_kmedoids_multiple(X_new, n_clusters=n_clusters)
    eucOHE_labels_list = eucledian_kmedoids_OHE_multiple(X_new, n_clusters=n_clusters)
    gower_labels_list = gower_kmedoids_multiple(X_new, n_clusters=n_clusters)
    heom_labels_list = heom_kmedoids_multiple(X_new, n_clusters=n_clusters)
    rex_labels_list = resample_exposure_kmedoids_multiple(X_new, n_clusters=n_clusters)
    
    ari_dict = {}
    # Compute average ARI for each method
    ari_dict["GEM"] = compute_aris(y, gem_labels_list)
    ari_dict["L2"] = compute_aris(y, euc_labels_list)
    ari_dict["L2_OHE"] = compute_aris(y, eucOHE_labels_list)
    ari_dict["GOW"] = compute_aris(y, gower_labels_list)
    ari_dict["HEOM"] = compute_aris(y, heom_labels_list)
    ari_dict["REX"] = compute_aris(y, rex_labels_list)

    nmi_dict = {}
    # Compute average ARI for each method
    nmi_dict["GEM"] = compute_nmis(y, gem_labels_list)
    nmi_dict["L2"] = compute_nmis(y, euc_labels_list)
    nmi_dict["L2_OHE"] = compute_nmis(y, eucOHE_labels_list)
    nmi_dict["GOW"] = compute_nmis(y, gower_labels_list)
    nmi_dict["HEOM"] = compute_nmis(y, heom_labels_list)
    nmi_dict["REX"] = compute_nmis(y, rex_labels_list)

    return ari_dict, nmi_dict

if __name__ == "__main__":
    import time, json

    start_time = time.time()
    All_Scores_dict['ARI']['Small']['Reals'], All_Scores_dict['NMI']['Small']['Reals'] = compute_scores(Small_Reals_X, Small_Reals_y, n_clusters = 5)
    print ("All_Scores_dict['ARI']['Small']['Reals'] is Done")
    All_Scores_dict['ARI']['Small']['Balanced'], All_Scores_dict['NMI']['Small']['Balanced'] = compute_scores(Small_Balanced_X, Small_Balanced_y, n_clusters = 5)
    print ("All_Scores_dict['ARI']['Small']['Balanced'] is Done")
    All_Scores_dict['ARI']['Small']['Cats'], All_Scores_dict['NMI']['Small']['Cats'] = compute_scores(Small_Cats_X, Small_Cats_y, n_clusters = 5)
    print ("All_Scores_dict['ARI']['Small']['Cats'] is Done")
    print(f"Time taken for Small datasets: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    All_Scores_dict['ARI']['Medium']['Reals'], All_Scores_dict['NMI']['Medium']['Reals'] = compute_scores(Medium_Reals_X, Medium_Reals_y, n_clusters = 5)
    print ("All_Scores_dict['ARI']['Medium']['Reals'] is Done")
    All_Scores_dict['ARI']['Medium']['Balanced'], All_Scores_dict['NMI']['Medium']['Balanced'] = compute_scores(Medium_Balanced_X, Medium_Balanced_y, n_clusters = 5)
    print ("All_Scores_dict['ARI']['Medium']['Balanced'] is Done")
    All_Scores_dict['ARI']['Medium']['Cats'], All_Scores_dict['NMI']['Medium']['Cats'] = compute_scores(Medium_Cats_X, Medium_Cats_y, n_clusters = 5)
    print ("All_Scores_dict['ARI']['Medium']['Cats'] is Done")
    print(f"Time taken for Medium datasets: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    All_Scores_dict['ARI']['Large']['Reals'], All_Scores_dict['NMI']['Large']['Reals'] = compute_scores(Large_Reals_X, Large_Reals_y, n_clusters = 5)
    print ("All_Scores_dict['ARI']['Large']['Reals'] is Done")
    All_Scores_dict['ARI']['Large']['Balanced'], All_Scores_dict['ARI']['Large']['Balanced'] = compute_scores(Large_Balanced_X, Large_Balanced_y, n_clusters = 5)
    print ("All_Scores_dict['ARI']['Large']['Balanced'] is Done")
    All_Scores_dict['ARI']['Large']['Cats'], All_Scores_dict['NMI']['Large']['Cats'] = compute_scores(Large_Cats_X, Large_Cats_y, n_clusters = 5)
    print ("All_Scores_dict['ARI']['Large']['Balanced'] is Done")
    print(f"Time taken for Large datasets: {time.time() - start_time:.2f} seconds")

    # Writing Dictionary into a json file for the case of re-runs
    with open('experiments/results/03_clustering_scores.json', 'w') as f:
        json.dump(All_Scores_dict, f)