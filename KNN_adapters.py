# Description: KNN adapters for different distance metrics
# Author: Anton D. Lautrup
# Date: 20-05-2025

import numpy as np

from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score
from prepare_data import get_categorical_features

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

    def fit_nn(self, X):
        pass
    
    def get_neighbors(self, X, n_neighbors: int) -> np.ndarray:
        pass

### Concrete adapter for KNN with L2 distance
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
class EuclideanKNN(KNNAdapter):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.cls_model = KNeighborsClassifier(n_neighbors, metric='euclidean')
        self.nn_model = NearestNeighbors(metric='euclidean')
        pass

    def name() -> str:
        return "L2"

    def fit_cls(self, X, y):
        self.cls_model.fit(X, y)
    
    def fit_nn(self, X):
        self.nn_model.fit(X)

    def predict(self, X):
        return self.cls_model.predict(X)

    def get_neighbors(self, X, n_neighbors: int = 5):
        dists, neighbors = self.nn_model.kneighbors(X, n_neighbors, return_distance=True)
        return dists, neighbors

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
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        X_encoded = self.nn_encoder.transform(X[self.cat_cols])
        X = np.hstack((X.drop(columns=self.cat_cols).values, X_encoded.toarray()))
        
        dists, neighbors = self.nn_model.kneighbors(X, n_neighbors, return_distance=True)
        return dists, neighbors

### REX distance metric
from rex_score.resample_exposure import ResampleExposure
class REX_KNN(KNNAdapter):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
        self.nn_model = NearestNeighbors(metric='precomputed')
        pass

    def name() -> str:
        return "REX"
    
    def fit_cls(self, X, y):
        self.memory = X
        self.REX = ResampleExposure(X)
        rex_train = self.REX.resample_exposure_matrix()
        rex_train = np.abs(np.ones_like(rex_train) - rex_train)
        self.cls_model.fit(rex_train, y)

    def fit_nn(self, X):
        self.memory = X
        self.REX = ResampleExposure(X)
        rex_train = self.REX.resample_exposure_matrix()
        rex_train = np.abs(np.ones_like(rex_train) - rex_train)
        self.nn_model.fit(rex_train)

    def predict(self, X):
        rex_test = self.REX.resample_exposure_matrix(X, True, reverse_direction=True)
        rex_test = np.abs(np.ones_like(rex_test) - rex_test).T
        return self.cls_model.predict(rex_test)
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        rex_test = self.REX.resample_exposure_matrix(X, True, overwrite_memory=True)
        rex_test = np.abs(np.ones_like(rex_test) - rex_test)
        dists, neighbors = self.nn_model.kneighbors(rex_test, n_neighbors, return_distance=True)
        return dists, neighbors

### Own distance metric
# from own_method import own_distance_matrix
# class OwnKNN(KNNAdapter):
#     def __init__(self, n_neighbors: int = 5):
#         self.n_neighbors = n_neighbors
#         self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
#         self.nn_model = NearestNeighbors(metric='precomputed')
#         pass

#     def name() -> str:
#         return "Ours"
    
#     def fit_cls(self, X, y):
#         self.memory = X
#         own_train = own_distance_matrix(X)
#         self.cls_model.fit(own_train, y)

#     def fit_nn(self, X):
#         self.memory = X
#         own_train = own_distance_matrix(X)
#         self.nn_model.fit(own_train)

#     def predict(self, X):
#         own_test = own_distance_matrix(X, self.memory)
#         return self.cls_model.predict(own_test)
    
#     def get_neighbors(self, X, n_neighbors: int = 5):
#         own_test = own_distance_matrix(X, self.memory)
#         dists, neighbors = self.nn_model.kneighbors(own_test, n_neighbors, return_distance=True)
#         return dists, neighbors

### Gowers distance 
import gower
class GowerKNN(KNNAdapter):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
        self.nn_model = NearestNeighbors(metric='precomputed')

    def name() -> str:
        return "Gower"
    
    def fit_cls(self, X, y):
        self.memory = X
        gower_train = gower.gower_matrix(X, X)
        self.cls_model.fit(gower_train, y)

    def fit_nn(self, X):
        self.memory = X
        gower_train = gower.gower_matrix(X, X)
        self.nn_model.fit(gower_train)

    def predict(self, X):
        gower_test = gower.gower_matrix(X, self.memory)
        return self.cls_model.predict(gower_test)
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        gower_test = gower.gower_matrix(X, self.memory)
        dists, neighbors = self.nn_model.kneighbors(gower_test, n_neighbors, return_distance=True)
        return dists, neighbors
    
### Cosine similarity
# from sklearn.metrics.pairwise import cosine_similarity
# class CosineKNN(KNNAdapter):
#     def __init__(self, n_neighbors: int = 5):
#         self.n_neighbors = n_neighbors
#         self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
#         self.nn_model = NearestNeighbors(metric='precomputed')

#     def name() -> str:
#         return "Cosine"
    
#     def fit(self, X, y):
#         self.memory = X
#         cosine_train = cosine_similarity(X)
#         cosine_train = np.abs(np.ones_like(cosine_train) - cosine_train)
#         self.cls_model.fit(cosine_train, y)
#         self.nn_model.fit(cosine_train)

#     def predict(self, X):
#         cosine_test = cosine_similarity(X, self.memory)
#         cosine_test = np.abs(np.ones_like(cosine_test) - cosine_test)
#         return self.cls_model.predict(cosine_test)
    
#     def get_neighbors(self, X, n_neighbors: int = 5):
#         cosine_test = cosine_similarity(X, self.memory)
#         cosine_test = np.abs(np.ones_like(cosine_test) - cosine_test)
#         dists, neighbors = self.nn_model.kneighbors(cosine_test, n_neighbors, return_distance=True)
#         return dists, neighbors
    
# ### Chi-squared distance
# from sklearn.metrics.pairwise import chi2_kernel
# class Chi2KNN(KNNAdapter):
#     def __init__(self, n_neighbors: int = 5):
#         self.n_neighbors = n_neighbors
#         self.cls_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
#         self.nn_model = NearestNeighbors(metric='precomputed')

#     def name() -> str:
#         return "Chi2"
    
#     def fit(self, X, y):
#         self.memory = X
#         chi2_train = chi2_kernel(X)
#         chi2_train = np.abs(np.ones_like(chi2_train) - chi2_train)
#         self.cls_model.fit(chi2_train, y)
#         self.nn_model.fit(chi2_train)

#     def predict(self, X):
#         chi2_test = chi2_kernel(X, self.memory)
#         chi2_test = np.abs(np.ones_like(chi2_test) - chi2_test)
#         return self.cls_model.predict(chi2_test)
    
#     def get_neighbors(self, X, n_neighbors: int = 5):
#         chi2_test = chi2_kernel(X, self.memory)
#         chi2_test = np.abs(np.ones_like(chi2_test) - chi2_test)
#         dists, neighbors = self.nn_model.kneighbors(chi2_test, n_neighbors, return_distance=True)
#         return dists, neighbors
    
### Heterogeneous Euclidean-overlap 
from implemented_distances import heom_distance_matrix
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
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        heom_test = heom_distance_matrix(X, self.memory)
        dists, neighbors = self.nn_model.kneighbors(heom_test, n_neighbors, return_distance=True)
        return dists, neighbors
        
### Generalised Euclidean distance
from implemented_distances import ichino_yaguchi_distance_matrix
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
    
    def get_neighbors(self, X, n_neighbors: int = 5):
        gem_test = ichino_yaguchi_distance_matrix(X, self.memory)
        dists, neighbors = self.nn_model.kneighbors(gem_test, n_neighbors, return_distance=True)
        return dists, neighbors
    
### Heterogeneous value difference metric
from implemented_distances import hvdm_distance_matrix, compute_vdm_tables
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
    