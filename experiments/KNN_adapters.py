# Description: KNN adapters for different distance metrics
# Author: Anton D. Lautrup
# Date: 20-05-2025

import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, List
from pandas import DataFrame
from numpy import ndarray

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import accuracy_score
from experiments.prepare_data import get_categorical_features

### Adapter pattern 
class KNNAdapter(ABC):
    """Abstract base class for KNN adapters."""
    def __init__(self, n_neighbors: int):
        self.n_neighbors = n_neighbors
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @abstractmethod
    def fit_cls(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def predict_proba(self, X):
        raise NotImplementedError("This method is not implemented in the base class.")

    def fit_nn(self, X):
        pass
    
    def get_neighbors(self, X, n_neighbors: int) -> np.ndarray:
        pass

### Concrete adapter for KNN with L2 distance
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
class EuclideanKNN(KNNAdapter):
    def __init__(self, n_neighbors: int = 5, weights: np.ndarray = None):
        self.n_neighbors = n_neighbors
        self.cls_model = KNeighborsClassifier(n_neighbors, metric_params={'w': weights} if weights is not None else None)
        self.nn_model = NearestNeighbors(metric_params={'w': weights} if weights is not None else None)
        pass

    def name() -> str:
        return "L2"

    def fit_cls(self, X, y):
        self.cls_model.fit(X, y)
    
    def fit_nn(self, X):
        self.nn_model.fit(X)

    def predict(self, X):
        return self.cls_model.predict(X)
    
    def predict_proba(self, X):
        return self.cls_model.predict_proba(X)

    def get_neighbors(self, X, n_neighbors: int = 5):
        dists, neighbors = self.nn_model.kneighbors(X, n_neighbors, return_distance=True)
        return dists, neighbors
    
def eucledian_kmedoids(df: DataFrame, n_clusters: int = 4, seed: int = 42) -> Tuple[ndarray, ndarray]:
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=seed, metric='euclidean')
    kmedoids.fit(df)
    
    return kmedoids.labels_, kmedoids.medoid_indices_


from sklearn.preprocessing import OneHotEncoder
class EuclideanKNN_OneHot(KNNAdapter):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.unique_threshold = 10
        self.cls_model = KNeighborsClassifier(n_neighbors, metric='euclidean')
        self.nn_model = NearestNeighbors(metric='euclidean')
        self.cls_encoder = OneHotEncoder(handle_unknown='ignore')
        self.nn_encoder = OneHotEncoder(handle_unknown='ignore')
        pass

    def name() -> str:
        return "L2_OHE"
    
    def fit_cls(self, X, y):
        # One-hot encode categorical features
        self.cat_cols = get_categorical_features(X, self.unique_threshold)
        self.cls_encoder.fit(X[self.cat_cols])
        X_encoded = self.cls_encoder.transform(X[self.cat_cols])
        X = np.hstack((X.drop(columns=self.cat_cols).values, X_encoded.toarray()))

        self.cls_model.fit(X, y)
    
    def fit_nn(self, X):
        # One-hot encode categorical features
        self.cat_cols = get_categorical_features(X, self.unique_threshold)
        self.nn_encoder.fit(X[self.cat_cols])
        X_encoded = self.nn_encoder.transform(X[self.cat_cols])
        X = np.hstack((X.drop(columns=self.cat_cols).values, X_encoded.toarray()))

        self.nn_model.fit(X)

    def predict(self, X):
        # One-hot encode categorical features
        X_encoded = self.cls_encoder.transform(X[self.cat_cols])
        X = np.hstack((X.drop(columns=self.cat_cols).values, X_encoded.toarray()))
        return self.cls_model.predict(X)
    
    def predict_proba(self, X):
        # One-hot encode categorical features
        X_encoded = self.cls_encoder.transform(X[self.cat_cols])
        X = np.hstack((X.drop(columns=self.cat_cols).values, X_encoded.toarray()))
        return self.cls_model.predict_proba(X)
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        X_encoded = self.nn_encoder.transform(X[self.cat_cols])
        X = np.hstack((X.drop(columns=self.cat_cols).values, X_encoded.toarray()))
        
        dists, neighbors = self.nn_model.kneighbors(X, n_neighbors, return_distance=True)
        return dists, neighbors

def eucledian_kmedoids_OHE(df: DataFrame, n_clusters: int = 4, seed: int = 42) -> Tuple[ndarray, ndarray]:
    df = df.copy()
    cat_feats = get_categorical_features(df)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_ohe = ohe.fit_transform(df[cat_feats])
    df_ohe = np.hstack((df_ohe, df.drop(columns=cat_feats).values))  # Combine OHE with numerical features

    kmedoids = KMedoids(n_clusters=n_clusters, random_state=seed, metric='euclidean')
    kmedoids.fit(df_ohe)

    return kmedoids.labels_, kmedoids.medoid_indices_


### REX distance metric
from rex_score.resample_exposure import ResampleExposure
class REX_KNN(KNNAdapter):
    def __init__(self, n_neighbors: int = 5, weights: np.ndarray = None):
        self.n_neighbors = n_neighbors
        self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
        self.nn_model = NearestNeighbors(metric='precomputed')
        self.weights = weights
        pass

    def name() -> str:
        return "REX"
    
    def fit_cls(self, X, y):
        self.memory = X
        self.REX = ResampleExposure(X, feature_weights=self.weights)
        rex_train = self.REX.resample_exposure_matrix()
        rex_train = np.abs(np.ones_like(rex_train) - rex_train)
        self.cls_model.fit(rex_train, y)

    def fit_nn(self, X):
        self.memory = X
        self.REX = ResampleExposure(X, feature_weights=self.weights)
        rex_train = self.REX.resample_exposure_matrix()
        rex_train = np.abs(np.ones_like(rex_train) - rex_train)
        self.nn_model.fit(rex_train)

    def predict(self, X):
        rex_test = self.REX.resample_exposure_matrix(X, True, reverse_direction=True)
        rex_test = np.abs(np.ones_like(rex_test) - rex_test).T
        return self.cls_model.predict(rex_test)
    
    def predict_proba(self, X):
        rex_test = self.REX.resample_exposure_matrix(X, True, reverse_direction=True)
        rex_test = np.abs(np.ones_like(rex_test) - rex_test).T
        return self.cls_model.predict_proba(rex_test)
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        rex_test = self.REX.resample_exposure_matrix(X, True, overwrite_memory=True)
        rex_test = np.abs(np.ones_like(rex_test) - rex_test)
        dists, neighbors = self.nn_model.kneighbors(rex_test, n_neighbors, return_distance=True)
        return dists, neighbors

def resample_exposure_kmedoids(df: DataFrame, n_clusters: int = 4, seed: int = 42, reverse: bool = False) -> Tuple[ndarray, ndarray]:
    rex = ResampleExposure(df)
    exposure_matrix = rex.resample_exposure_matrix(normalised=True, reverse_direction=reverse)
    exposure_matrix = np.ones_like(exposure_matrix) - exposure_matrix  # Invert the exposure matrix for PCA
    np.fill_diagonal(exposure_matrix, 0)  # Set diagonal to 0 to avoid self-distance
    if reverse:
        exposure_matrix = np.transpose(exposure_matrix)

    kmedoids = KMedoids(n_clusters=n_clusters, random_state=seed, metric='precomputed')
    kmedoids.fit(exposure_matrix)

    return kmedoids.labels_, kmedoids.medoid_indices_

### Gowers distance 
import gower
class GowerKNN(KNNAdapter):
    def __init__(self, n_neighbors: int = 5, weights: np.ndarray = None):
        self.n_neighbors = n_neighbors
        self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
        self.nn_model = NearestNeighbors(metric='precomputed')
        self.weights = weights#/weights.sum() if weights is not None else None

    def name() -> str:
        return "GOW"
    
    def fit_cls(self, X, y):
        self.memory = X
        gower_train = gower.gower_matrix(X, X, weight=self.weights if self.weights is not None else None)
        self.cls_model.fit(gower_train, y)

    def fit_nn(self, X):
        self.memory = X
        gower_train = gower.gower_matrix(X, X, weight=self.weights if self.weights is not None else None)
        self.nn_model.fit(gower_train)

    def predict(self, X):
        gower_test = gower.gower_matrix(X, self.memory, weight=self.weights if self.weights is not None else None)
        return self.cls_model.predict(gower_test)
    
    def predict_proba(self, X):
        gower_test = gower.gower_matrix(X, self.memory, weight=self.weights if self.weights is not None else None)
        return self.cls_model.predict_proba(gower_test)
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        gower_test = gower.gower_matrix(X, self.memory, weight=self.weights if self.weights is not None else None)
        dists, neighbors = self.nn_model.kneighbors(gower_test, n_neighbors, return_distance=True)
        return dists, neighbors
    
def gower_kmedoids(df: DataFrame, n_clusters: int = 4, seed: int = 42) -> Tuple[ndarray, ndarray]:
    gower_matrix = gower.gower_matrix(df)

    kmedoids = KMedoids(n_clusters=n_clusters, random_state=seed, metric='precomputed')
    kmedoids.fit(gower_matrix)

    return kmedoids.labels_, kmedoids.medoid_indices_

   
### Heterogeneous Euclidean-overlap 
from experiments.implemented_distances import heom_distance_matrix
class HeomKNN(KNNAdapter):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.unique_threshold = 10
        self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
        self.nn_model = NearestNeighbors(metric='precomputed')

    def name() -> str:
        return "HEOM"
    
    def fit_cls(self, X, y):
        self.memory = X
        heom_train = heom_distance_matrix(X)
        self.cls_model.fit(heom_train, y)
    
    def fit_nn(self, X):
        self.memory = X
        heom_train = heom_distance_matrix(X)
        self.nn_model.fit(heom_train)
    
    def predict(self, X):
        heom_test = heom_distance_matrix(X, self.memory)
        return self.cls_model.predict(heom_test)
    
    def predict_proba(self, X):
        heom_test = heom_distance_matrix(X, self.memory)
        return self.cls_model.predict_proba(heom_test)
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        heom_test = heom_distance_matrix(X, self.memory)
        dists, neighbors = self.nn_model.kneighbors(heom_test, n_neighbors, return_distance=True)
        return dists, neighbors

def heom_kmedoids(df: DataFrame, n_clusters: int = 4, seed: int = 42) -> Tuple[ndarray, ndarray]:
    distance_matrix = heom_distance_matrix(df)
    
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=seed, metric='precomputed')
    kmedoids.fit(distance_matrix)

    return kmedoids.labels_, kmedoids.medoid_indices_

### Generalised Euclidean distance
from experiments.implemented_distances import ichino_yaguchi_distance_matrix
class GeneralisedEuclideanKNN(KNNAdapter):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.unique_threshold = 10
        self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
        self.nn_model = NearestNeighbors(metric='precomputed')

    def name() -> str:
        return "GEM"
    
    def fit_cls(self, X, y):
        self.memory = X
        gem_train = ichino_yaguchi_distance_matrix(X)
        self.cls_model.fit(gem_train, y)
    
    def fit_nn(self, X):
        self.memory = X
        gem_train = ichino_yaguchi_distance_matrix(X)
        self.nn_model.fit(gem_train)
    
    def predict(self, X):
        gem_test = ichino_yaguchi_distance_matrix(X, self.memory)
        return self.cls_model.predict(gem_test)
    
    def predict_proba(self, X):
        gem_test = ichino_yaguchi_distance_matrix(X, self.memory)
        return self.cls_model.predict_proba(gem_test)
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        gem_test = ichino_yaguchi_distance_matrix(X, self.memory)
        dists, neighbors = self.nn_model.kneighbors(gem_test, n_neighbors, return_distance=True)
        return dists, neighbors

def gem_kmedoids(df: DataFrame, n_clusters: int = 4, seed: int = 42) -> Tuple[ndarray, ndarray]:
    distance_matrix = ichino_yaguchi_distance_matrix(df)
    
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=seed, metric='precomputed')
    kmedoids.fit(distance_matrix)

    return kmedoids.labels_, kmedoids.medoid_indices_


### Heterogeneous value difference metric
from experiments.implemented_distances import hvdm_distance_matrix, compute_vdm_tables
class HvdmKNN(KNNAdapter):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.unique_threshold = 10
        self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
        self.nn_model = NearestNeighbors(metric='precomputed')

    def name() -> str:
        return "HVDM"
    
    def fit_cls(self, X, y):
        self.memory = (X, y)
        hvdm_train = hvdm_distance_matrix(X, y_labels=y)
        self.cls_model.fit(hvdm_train, y)
        
    def predict(self, X):
        vdm_probs, std_dict, cat_features, num_features, class_values = compute_vdm_tables(self.memory[0], self.memory[1])
        hvdm_test = hvdm_distance_matrix(
            X, Y=self.memory[0],
            vdm_probs_np_lookup=vdm_probs,
            std_dict=std_dict,
            cat_feature_names=cat_features,
            num_feature_names=num_features,
            class_values=class_values
        )
        return self.cls_model.predict(hvdm_test)
    
    def predict_proba(self, X):
        vdm_probs, std_dict, cat_features, num_features, class_values = compute_vdm_tables(self.memory[0], self.memory[1])
        hvdm_test = hvdm_distance_matrix(
            X, Y=self.memory[0],
            vdm_probs_np_lookup=vdm_probs,
            std_dict=std_dict,
            cat_feature_names=cat_features,
            num_feature_names=num_features,
            class_values=class_values
        )
        return self.cls_model.predict_proba(hvdm_test)
