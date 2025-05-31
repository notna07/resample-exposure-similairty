# Description: Script to generate synthetic datasets for experiments
# Author: Hafiz Saud Arshad
# Date: 30-05-2025

import warnings
warnings.filterwarnings('ignore')

import numpy as np

from numpy import ndarray
from typing import Tuple

from scipy.stats import ortho_group

from sklearn.utils import shuffle
from sklearn.datasets import make_moons

###################################
### For Large - Balanced : 7, 7 ###
###################################

def make_21d_moons(n_samples: int = 1000, 
                   noise: float = 0.15, 
                   rotation: bool = True, 
                   random_state: int = None) -> Tuple[ndarray, ndarray]:
    """ Generate intertwined 21-dimensional moon-shaped clusters.
    
    Parameters:
    n_samples : int, default=1000
        Total number of points (evenly split between moons).
    noise : float, default=0.15
        Standard deviation of Gaussian noise added to each dimension.
    rotation : bool, default=True
        Apply random rotation to distribute structure across dimensions.
    random_state : int, default=None
        Seed for reproducibility.
        
    Returns:
    X : ndarray of shape (n_samples, 21)
        The generated 21D moon data.
    y : ndarray of shape (n_samples,)
        Labels (0 or 1) for each moon.
    """
    np.random.seed(random_state)
    
    # Generate base 2D moons
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Initialize 21D array
    X_21d = np.zeros((n_samples, 21))
    
    # Compute radial and angular features
    r = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)
    theta = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    # Apply 21 distinct non-linear transformations
    X_21d[:, 0] = 2.8 * X_2d[:, 0]
    X_21d[:, 1] = 1.7 * X_2d[:, 1] + 0.9 * np.sin(X_2d[:, 0] * np.pi)
    X_21d[:, 2] = 1.5 * r * np.sin(2 * theta) - 0.8 * np.cos(X_2d[:, 0] * np.pi)
    X_21d[:, 3] = 0.7 * (X_2d[:, 0] * X_2d[:, 1]) + 1.2 * np.sin(r)
    X_21d[:, 4] = np.exp(-0.3 * r) * np.cos(3 * theta)
    X_21d[:, 5] = np.log(1 + np.abs(X_2d[:, 0])) * np.sign(X_2d[:, 1])
    X_21d[:, 6] = 0.5 * X_2d[:, 0]**2 - 0.8 * X_2d[:, 1]**2
    X_21d[:, 7] = np.sin(X_2d[:, 0]) * np.cos(X_2d[:, 1]) * r
    X_21d[:, 8] = np.tanh(0.4 * X_2d[:, 0]) + 0.6 * np.arctan(X_2d[:, 1])
    X_21d[:, 9] = np.sqrt(np.abs(X_2d[:, 0] + X_2d[:, 1])) * np.sin(2 * np.pi * r)
    X_21d[:, 10] = (X_2d[:, 0] * np.cos(theta) - X_2d[:, 1] * np.sin(theta)) * 1.5
    X_21d[:, 11] = 0.8 * np.sin(4 * theta) * r**0.7
    X_21d[:, 12] = np.arctan2(X_2d[:, 1] * 0.8, X_2d[:, 0] * 1.2) * np.exp(-r/3)
    X_21d[:, 13] = 1.2 * (X_2d[:, 0] - X_2d[:, 1])**3 / (1 + r)
    X_21d[:, 14] = np.cos(2 * X_2d[:, 0]) * np.sinh(0.5 * X_2d[:, 1])
    X_21d[:, 15] = 0.6 * np.log(1 + np.abs(X_2d[:, 0] * X_2d[:, 1] + 1))
    X_21d[:, 16] = np.where(X_2d[:, 1] > 0, np.sin(r), np.cos(r)) * 1.4
    X_21d[:, 17] = (X_2d[:, 0]**3 - 3*X_2d[:, 0]*X_2d[:, 1]**2) * 0.25
    X_21d[:, 18] = np.power(np.abs(X_2d[:, 0] - X_2d[:, 1]), 1.5) * np.sign(X_2d[:, 0])
    X_21d[:, 19] = 1.1 * np.sin(X_2d[:, 0] + theta) * np.cos(X_2d[:, 1] - theta)
    X_21d[:, 20] = 0.9 * (np.sin(2*np.pi*X_2d[:, 0]) + np.cos(2*np.pi*X_2d[:, 1]))
    
    # Add structured noise
    noise_vec = np.random.normal(0, noise, (n_samples, 21))
    X_21d += noise_vec
    
    # Apply random rotation for dimensionality entanglement
    if rotation:
        rotation_matrix = ortho_group.rvs(dim=21, random_state=random_state)
        X_21d = X_21d @ rotation_matrix
    
    return X_21d, y

def data_generator21(n_blobs: int = 3, 
                     n_moons: int = 2, 
                     dim: int = 21, 
                     n_samples_per_cluster: int = 150) -> Tuple[ndarray, ndarray]:
    """ Generate a dataset with Gaussian blobs and moon pairs in 21D space.
    
    Parameters:
    n_blobs : int, default=3
        Number of Gaussian clusters
    n_moons : int, default=2
        Number of moon clusters (must be even, each pair counts as 2 clusters)
    dim : int, default=21
        Dimensionality of the space (must be 21)
    n_samples_per_cluster : int, default=300
        Number of samples per cluster
        
    Returns:
    X : ndarray of shape (total_samples, 21)
        The generated dataset
    y : ndarray of shape (total_samples,)
        Cluster labels
    """
    if dim != 21:
        raise ValueError("This generator only supports dim=21")
    
    if n_moons % 2 != 0:
        raise ValueError("n_moons must be even (each moon pair counts as 2 clusters)")
    
    np.random.seed(42)
    n_moon_pairs = n_moons // 2
    
    # ==================================================================
    # Generate Gaussian blobs in 21D with minimal overlap
    # ==================================================================
    # Predefined centers with directional separation
    centers = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # Center 1
        [6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # Center 2
        [-4, -4, -4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # Center 3
        [0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # Center 4 (if needed)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, -6, -6, 0, 0, 0, 0, 0, 0, 0]         # Center 5 (if needed)
    ]
    
    # Directional variances to minimize overlap
    cluster_stds = [
        [1.4, 1.0, 1.7, 2.3, 1.2, 1.1, 1.3, 1.5, 0.9, 1.6, 1.8, 1.0, 1.2, 1.1, 1.4, 1.3, 1.5, 1.0, 1.2, 1.6, 1.1],
        [1.1, 1.5, 0.9, 1.3, 2.0, 2.5, 1.2, 1.0, 1.6, 1.4, 0.8, 1.3, 1.1, 1.7, 1.0, 1.5, 1.2, 1.8, 1.1, 1.3, 0.9],
        [1.7, 1.2, 2.3, 1.0, 1.4, 0.9, 1.5, 2.0, 2.5, 1.1, 1.3, 1.6, 1.8, 1.2, 1.0, 1.4, 1.1, 1.3, 1.5, 0.8, 1.2],
        [1.3, 1.1, 1.5, 0.8, 1.2, 1.6, 1.0, 1.4, 1.7, 2.1, 1.2, 1.5, 0.9, 1.3, 1.8, 1.1, 1.4, 1.0, 1.2, 1.5, 1.3],
        [1.0, 1.4, 1.2, 1.5, 1.1, 1.3, 1.7, 1.2, 0.8, 1.5, 1.3, 1.6, 2.0, 1.4, 1.1, 1.3, 0.9, 1.5, 1.2, 1.0, 1.4]
    ]
    
    # Ensure we have enough centers and stds for requested blobs
    if n_blobs > len(centers):
        warnings.warn(f"Only {len(centers)} Gaussian clusters available, using first {len(centers)}")
        n_blobs = len(centers)
    
    X_gaussian = np.empty((0, dim))
    y_gaussian = np.empty(0, dtype=int)
    
    for i in range(n_blobs):
        cov = np.diag(np.square(cluster_stds[i][:dim]))
        cluster = np.random.multivariate_normal(
            centers[i][:dim], 
            cov, 
            size=n_samples_per_cluster
        )
        X_gaussian = np.vstack([X_gaussian, cluster])
        y_gaussian = np.concatenate([y_gaussian, np.full(n_samples_per_cluster, i)])
    
    # ==================================================================
    # Generate 21D moon pairs
    # ==================================================================
    # Predefined offsets to separate moon pairs
    moon_offsets = [
        [10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # Pair 1
        [0, 0, 0, 0, 0, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # Pair 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0],         # Pair 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, -15, -15, -15, -15, 0]      # Pair 4
    ]
    
    # Ensure we have enough offsets for requested moon pairs
    if n_moon_pairs > len(moon_offsets):
        warnings.warn(f"Only {len(moon_offsets)} moon pairs available, using first {len(moon_offsets)}")
        n_moon_pairs = len(moon_offsets)
    
    X_moons = np.empty((0, dim))
    y_moons = np.empty(0, dtype=int)
    
    for i in range(n_moon_pairs):
        # Generate moon pair
        X_pair, y_pair = make_21d_moons(
            n_samples=2 * n_samples_per_cluster,
            noise=0.15,
            rotation=True,
            random_state=42 + i
        )
        
        # Apply offset
        X_pair += np.array(moon_offsets[i][:dim])
        
        # Relabel moons
        y_pair += n_blobs + 2 * i  # Start labeling after Gaussian clusters
        
        X_moons = np.vstack([X_moons, X_pair])
        y_moons = np.concatenate([y_moons, y_pair])
    
    # ==================================================================
    # Combine datasets and shuffle
    # ==================================================================
    X = np.vstack([X_gaussian, X_moons])
    y = np.concatenate([y_gaussian, y_moons])
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

#################################
### For Large - Reals : 11, 3 ###
#################################

def make_17d_moons(n_samples: int = 1000, 
                   noise: float = 0.15, 
                   rotation: bool = True, 
                   random_state: int = None) -> Tuple[ndarray, ndarray]:
    """ Generate intertwined 17-dimensional moon-shaped clusters."""

    np.random.seed(random_state)
    
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_17d = np.zeros((n_samples, 17))
    
    r = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)
    theta = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    X_17d[:, 0] = 2.8 * X_2d[:, 0]
    X_17d[:, 1] = 1.7 * X_2d[:, 1] + 0.9 * np.sin(X_2d[:, 0] * np.pi)
    X_17d[:, 2] = 1.5 * r * np.sin(2 * theta) - 0.8 * np.cos(X_2d[:, 0] * np.pi)
    X_17d[:, 3] = 0.7 * (X_2d[:, 0] * X_2d[:, 1]) + 1.2 * np.sin(r)
    X_17d[:, 4] = np.exp(-0.3 * r) * np.cos(3 * theta)
    X_17d[:, 5] = np.log(1 + np.abs(X_2d[:, 0])) * np.sign(X_2d[:, 1])
    X_17d[:, 6] = 0.5 * X_2d[:, 0]**2 - 0.8 * X_2d[:, 1]**2
    X_17d[:, 7] = np.sin(X_2d[:, 0]) * np.cos(X_2d[:, 1]) * r
    X_17d[:, 8] = np.tanh(0.4 * X_2d[:, 0]) + 0.6 * np.arctan(X_2d[:, 1])
    X_17d[:, 9] = np.sqrt(np.abs(X_2d[:, 0] + X_2d[:, 1])) * np.sin(2 * np.pi * r)
    X_17d[:, 10] = (X_2d[:, 0] * np.cos(theta) - X_2d[:, 1] * np.sin(theta)) * 1.5
    X_17d[:, 11] = 0.8 * np.sin(4 * theta) * r**0.7
    X_17d[:, 12] = np.arctan2(X_2d[:, 1] * 0.8, X_2d[:, 0] * 1.2) * np.exp(-r/3)
    X_17d[:, 13] = 1.2 * (X_2d[:, 0] - X_2d[:, 1])**3 / (1 + r)
    X_17d[:, 14] = np.cos(2 * X_2d[:, 0]) * np.sinh(0.5 * X_2d[:, 1])
    X_17d[:, 15] = 0.6 * np.log(1 + np.abs(X_2d[:, 0] * X_2d[:, 1] + 1))
    X_17d[:, 16] = np.where(X_2d[:, 1] > 0, np.sin(r), np.cos(r)) * 1.4
    
    noise_vec = np.random.normal(0, noise, (n_samples, 17))
    X_17d += noise_vec
    
    if rotation:
        rotation_matrix = ortho_group.rvs(dim=17, random_state=random_state)
        X_17d = X_17d @ rotation_matrix
    
    return X_17d, y

def data_generator17(n_blobs: int = 3, 
                     n_moons: int = 2, 
                     dim: int = 17, 
                     n_samples_per_cluster: int = 300) -> Tuple[ndarray, ndarray]:
    """ Generate a dataset with Gaussian blobs and moon pairs in 17D space. """
    np.random.seed(42)
    dim = 17
    
    centers = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # Center 1
        [6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # Center 2
        [-4, -4, -4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]         # Center 3
    ]
    
    cluster_stds = [
        [1.4, 1.0, 1.7, 2.3, 1.2, 1.1, 1.3, 1.5, 0.9, 1.6, 1.8, 1.0, 1.2, 1.1, 1.4, 1.3, 1.5],
        [1.1, 1.5, 0.9, 1.3, 2.0, 2.5, 1.2, 1.0, 1.6, 1.4, 0.8, 1.3, 1.1, 1.7, 1.0, 1.5, 1.2],
        [1.7, 1.2, 2.3, 1.0, 1.4, 0.9, 1.5, 2.0, 2.5, 1.1, 1.3, 1.6, 1.8, 1.2, 1.0, 1.4, 1.1]
    ]
    
    X_gaussian = np.empty((0, dim))
    y_gaussian = np.empty(0, dtype=int)
    
    for i, (center, stds) in enumerate(zip(centers, cluster_stds)):
        cov = np.diag(np.square(stds))
        cluster = np.random.multivariate_normal(
            center, 
            cov, 
            size=n_samples_per_cluster
        )
        X_gaussian = np.vstack([X_gaussian, cluster])
        y_gaussian = np.concatenate([y_gaussian, np.full(n_samples_per_cluster, i)])
    
    X_moons, y_moons = make_17d_moons(
        n_samples=2 * n_samples_per_cluster,
        noise=0.15,
        rotation=True,
        random_state=42
    )
    
    offset = np.array([10, 10, 10, 10, 10, -5, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    X_moons += offset
    
    # Relabel moons to start after Gaussian clusters (cluster 3 and 4)
    y_moons += 3
    
    X = np.vstack([X_gaussian, X_moons])
    y = np.concatenate([y_gaussian, y_moons])
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

################################
### For Large - Cats : 3, 11 ###
################################

def make_25d_moons(n_samples: int = 1000, 
                   noise: float = 0.15, 
                   rotation: bool = True, 
                   random_state: int = None) -> Tuple[ndarray, ndarray]:
    """ Generate intertwined 25-dimensional moon-shaped clusters. """
    np.random.seed(random_state)
    
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_25d = np.zeros((n_samples, 25))
    
    r = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)
    theta = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    X_25d[:, 0] = 2.8 * X_2d[:, 0]
    X_25d[:, 1] = 1.7 * X_2d[:, 1] + 0.9 * np.sin(X_2d[:, 0] * np.pi)
    X_25d[:, 2] = 1.5 * r * np.sin(2 * theta) - 0.8 * np.cos(X_2d[:, 0] * np.pi)
    X_25d[:, 3] = 0.7 * (X_2d[:, 0] * X_2d[:, 1]) + 1.2 * np.sin(r)
    X_25d[:, 4] = np.exp(-0.3 * r) * np.cos(3 * theta)
    X_25d[:, 5] = np.log(1 + np.abs(X_2d[:, 0])) * np.sign(X_2d[:, 1])
    X_25d[:, 6] = 0.5 * X_2d[:, 0]**2 - 0.8 * X_2d[:, 1]**2
    X_25d[:, 7] = np.sin(X_2d[:, 0]) * np.cos(X_2d[:, 1]) * r
    X_25d[:, 8] = np.tanh(0.4 * X_2d[:, 0]) + 0.6 * np.arctan(X_2d[:, 1])
    X_25d[:, 9] = np.sqrt(np.abs(X_2d[:, 0] + X_2d[:, 1])) * np.sin(2 * np.pi * r)
    X_25d[:, 10] = (X_2d[:, 0] * np.cos(theta) - X_2d[:, 1] * np.sin(theta)) * 1.5
    X_25d[:, 11] = 0.8 * np.sin(4 * theta) * r**0.7
    X_25d[:, 12] = np.arctan2(X_2d[:, 1] * 0.8, X_2d[:, 0] * 1.2) * np.exp(-r/3)
    X_25d[:, 13] = 1.2 * (X_2d[:, 0] - X_2d[:, 1])**3 / (1 + r)
    X_25d[:, 14] = np.cos(2 * X_2d[:, 0]) * np.sinh(0.5 * X_2d[:, 1])
    
    X_25d[:, 15] = 0.6 * np.log(1 + np.abs(X_2d[:, 0] * X_2d[:, 1] + 1))
    X_25d[:, 16] = np.where(X_2d[:, 1] > 0, np.sin(r), np.cos(r)) * 1.4
    X_25d[:, 17] = (X_2d[:, 0]**3 - 3*X_2d[:, 0]*X_2d[:, 1]**2) * 0.25
    X_25d[:, 18] = np.power(np.abs(X_2d[:, 0] - X_2d[:, 1]), 1.5) * np.sign(X_2d[:, 0])
    X_25d[:, 19] = 1.1 * np.sin(X_2d[:, 0] + theta) * np.cos(X_2d[:, 1] - theta)
    X_25d[:, 20] = 0.9 * (np.sin(2*np.pi*X_2d[:, 0]) + np.cos(2*np.pi*X_2d[:, 1]))
    X_25d[:, 21] = (X_2d[:, 0]**2 * X_2d[:, 1] - X_2d[:, 1]**3/3) * 0.3
    X_25d[:, 22] = np.arctan(X_2d[:, 0] * X_2d[:, 1]) * r**0.5
    X_25d[:, 23] = np.sin(0.5 * np.pi * (X_2d[:, 0] + X_2d[:, 1])) * np.exp(-r/4)
    X_25d[:, 24] = 0.7 * (X_2d[:, 0] * np.cos(2*theta) - X_2d[:, 1] * np.sin(2*theta))
    
    # Add structured noise
    noise_vec = np.random.normal(0, noise, (n_samples, 25))
    X_25d += noise_vec
    
    # Apply random rotation for dimensionality entanglement
    if rotation:
        rotation_matrix = ortho_group.rvs(dim=25, random_state=random_state)
        X_25d = X_25d @ rotation_matrix
    
    return X_25d, y

def data_generator25(n_blobs: int = 3, 
                     n_moons: int = 2, 
                     dim: int = 25, 
                     n_samples_per_cluster: int = 300) -> Tuple[ndarray, ndarray]:
    """ Generate a dataset with Gaussian blobs and moon pairs in 25D space."""
    if dim != 25:
        raise ValueError("This generator only supports dim=25")
    
    if n_moons % 2 != 0:
        raise ValueError("n_moons must be even (each moon pair counts as 2 clusters)")
    
    np.random.seed(42)
    n_moon_pairs = n_moons // 2
    centers = [
        [0]*25,                                                # Center 1
        [6, 6, 6, 6, 6] + [0]*20,                              # Center 2
        [-4, -4, -4] + [0]*4 + [4, 4] + [0]*18,                # Center 3
        [0]*7 + [8, 8, 8, 8] + [0]*16,                         # Center 4
        [0]*11 + [-6, -6, -6] + [0]*14,                        # Center 5
        [0]*15 + [10, 10, 10, 10, 10] + [0]*5,                 # Center 6
        [0]*20 + [12, 12, 12, 12, 12]                          # Center 7
    ]
    
    cluster_stds = [
        [1.4, 1.0, 1.7, 2.3, 1.2, 1.1, 1.3, 1.5, 0.9, 1.6, 1.8, 1.0, 1.2, 1.1, 1.4, 
         1.3, 1.5, 1.0, 1.2, 1.6, 1.1, 1.3, 1.4, 1.0, 1.2],
        [1.1, 1.5, 0.9, 1.3, 2.0, 2.5, 1.2, 1.0, 1.6, 1.4, 0.8, 1.3, 1.1, 1.7, 1.0, 
         1.5, 1.2, 1.8, 1.1, 1.3, 0.9, 1.4, 1.2, 1.5, 1.0],
        [1.7, 1.2, 2.3, 1.0, 1.4, 0.9, 1.5, 2.0, 2.5, 1.1, 1.3, 1.6, 1.8, 1.2, 1.0, 
         1.4, 1.1, 1.3, 1.5, 0.8, 1.2, 1.0, 1.3, 1.6, 1.1],
        [1.3, 1.1, 1.5, 0.8, 1.2, 1.6, 1.0, 1.4, 1.7, 2.1, 1.2, 1.5, 0.9, 1.3, 1.8, 
         1.1, 1.4, 1.0, 1.2, 1.5, 1.3, 1.6, 1.0, 1.4, 1.1],
        [1.0, 1.4, 1.2, 1.5, 1.1, 1.3, 1.7, 1.2, 0.8, 1.5, 1.3, 1.6, 2.0, 1.4, 1.1, 
         1.3, 0.9, 1.5, 1.2, 1.0, 1.4, 1.7, 1.1, 1.3, 0.8],
        [1.5, 1.8, 1.3, 1.1, 1.6, 1.2, 1.4, 1.0, 1.7, 1.3, 1.5, 0.9, 1.2, 1.6, 1.1, 
         1.4, 1.8, 1.2, 1.0, 1.5, 1.3, 1.1, 1.4, 1.7, 1.0],
        [1.2, 1.0, 1.4, 1.7, 1.1, 1.5, 1.3, 1.6, 1.0, 1.8, 1.2, 1.5, 1.3, 0.9, 1.1, 
         1.6, 1.4, 1.7, 1.0, 1.3, 1.5, 1.2, 0.8, 1.1, 1.4]
    ]
    
    if n_blobs > len(centers):
        warnings.warn(f"Only {len(centers)} Gaussian clusters available, using first {len(centers)}")
        n_blobs = len(centers)
    
    X_gaussian = np.empty((0, dim))
    y_gaussian = np.empty(0, dtype=int)
    
    for i in range(n_blobs):
        cov = np.diag(np.square(cluster_stds[i][:dim]))
        cluster = np.random.multivariate_normal(
            centers[i][:dim], 
            cov, 
            size=n_samples_per_cluster
        )
        X_gaussian = np.vstack([X_gaussian, cluster])
        y_gaussian = np.concatenate([y_gaussian, np.full(n_samples_per_cluster, i)])
    
    moon_offsets = [
        [10]*5 + [0]*20,                                       # Pair 1
        [0]*5 + [-10]*5 + [0]*15,                              # Pair 2
        [0]*10 + [15]*7 + [0]*8,                               # Pair 3
        [0]*17 + [-15]*8,                                      # Pair 4
        [12]*12 + [0]*13,                                      # Pair 5
        [0]*12 + [-12]*13,                                     # Pair 6
        [8]*8 + [0]*8 + [14]*9                                 # Pair 7
    ]
    
    if n_moon_pairs > len(moon_offsets):
        warnings.warn(f"Only {len(moon_offsets)} moon pairs available, using first {len(moon_offsets)}")
        n_moon_pairs = len(moon_offsets)
    
    X_moons = np.empty((0, dim))
    y_moons = np.empty(0, dtype=int)
    
    for i in range(n_moon_pairs):
    
        X_pair, y_pair = make_25d_moons(
            n_samples=2 * n_samples_per_cluster,
            noise=0.15,
            rotation=True,
            random_state=42 + i
        )
        
        # Apply offset
        X_pair += np.array(moon_offsets[i][:dim])
        
        # Relabel moons
        y_pair += n_blobs + 2 * i  # Start labeling after Gaussian clusters
        
        X_moons = np.vstack([X_moons, X_pair])
        y_moons = np.concatenate([y_moons, y_pair])
    
    X = np.vstack([X_gaussian, X_moons])
    y = np.concatenate([y_gaussian, y_moons])
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

####################################
### For Medium - Balanced : 4, 4 ###
####################################

def make_12d_moons(n_samples: int = 1000, 
                   noise: float = 0.15, 
                   rotation: bool = True, 
                   random_state: int = None) -> Tuple[ndarray, ndarray]:
    """ Generate intertwined 12-dimensional moon-shaped clusters."""
    np.random.seed(random_state)
    
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_12d = np.zeros((n_samples, 12))
    
    r = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)
    theta = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    X_12d[:, 0] = 2.8 * X_2d[:, 0]
    X_12d[:, 1] = 1.7 * X_2d[:, 1] + 0.9 * np.sin(X_2d[:, 0] * np.pi)
    X_12d[:, 2] = 1.5 * r * np.sin(2 * theta) - 0.8 * np.cos(X_2d[:, 0] * np.pi)
    X_12d[:, 3] = 0.7 * (X_2d[:, 0] * X_2d[:, 1]) + 1.2 * np.sin(r)
    X_12d[:, 4] = np.exp(-0.3 * r) * np.cos(3 * theta)
    X_12d[:, 5] = np.log(1 + np.abs(X_2d[:, 0])) * np.sign(X_2d[:, 1])
    X_12d[:, 6] = 0.5 * X_2d[:, 0]**2 - 0.8 * X_2d[:, 1]**2
    X_12d[:, 7] = np.sin(X_2d[:, 0]) * np.cos(X_2d[:, 1]) * r
    X_12d[:, 8] = np.tanh(0.4 * X_2d[:, 0]) + 0.6 * np.arctan(X_2d[:, 1])
    X_12d[:, 9] = np.sqrt(np.abs(X_2d[:, 0] + X_2d[:, 1])) * np.sin(2 * np.pi * r)
    X_12d[:, 10] = (X_2d[:, 0] * np.cos(theta) - X_2d[:, 1] * np.sin(theta)) * 1.5
    X_12d[:, 11] = 0.8 * np.sin(4 * theta) * r**0.7
    
    noise_vec = np.random.normal(0, noise, (n_samples, 12))
    X_12d += noise_vec
    
    if rotation:
        rotation_matrix = ortho_group.rvs(dim=12, random_state=random_state)
        X_12d = X_12d @ rotation_matrix
    
    return X_12d, y

def data_generator12(n_blobs: int = 3, 
                     n_moons: int = 2, 
                     dim: int = 12, 
                     n_samples_per_cluster: int = 300) -> Tuple[ndarray, ndarray]:
    """ Generate a dataset with Gaussian blobs and moon pairs in 12D space."""
    if dim != 12:
        raise ValueError("This generator only supports dim=12")
    
    if n_moons % 2 != 0:
        raise ValueError("n_moons must be even (each moon pair counts as 2 clusters)")
    
    np.random.seed(42)
    n_moon_pairs = n_moons // 2
    
    centers = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                    # Center 1
        [6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0],                     # Center 2
        [-4, -4, -4, 0, 0, 4, 4, 0, 0, 0, 0, 0],                  # Center 3
        [0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0],                     # Center 4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6]                     # Center 5
    ]
    
    cluster_stds = [
        [1.4, 1.0, 1.7, 2.3, 1.2, 1.1, 1.3, 1.5, 0.9, 1.6, 1.8, 1.0],
        [1.1, 1.5, 0.9, 1.3, 2.0, 2.5, 1.2, 1.0, 1.6, 1.4, 0.8, 1.3],
        [1.7, 1.2, 2.3, 1.0, 1.4, 0.9, 1.5, 2.0, 2.5, 1.1, 1.3, 1.6],
        [1.3, 1.1, 1.5, 0.8, 1.2, 1.6, 1.0, 1.4, 1.7, 2.1, 1.2, 1.5],
        [1.0, 1.4, 1.2, 1.5, 1.1, 1.3, 1.7, 1.2, 0.8, 1.5, 1.3, 1.6]
    ]
    
    if n_blobs > len(centers):
        warnings.warn(f"Only {len(centers)} Gaussian clusters available, using first {len(centers)}")
        n_blobs = len(centers)
    
    X_gaussian = np.empty((0, dim))
    y_gaussian = np.empty(0, dtype=int)
    
    for i in range(n_blobs):
        cov = np.diag(np.square(cluster_stds[i][:dim]))
        cluster = np.random.multivariate_normal(
            centers[i][:dim], 
            cov, 
            size=n_samples_per_cluster
        )
        X_gaussian = np.vstack([X_gaussian, cluster])
        y_gaussian = np.concatenate([y_gaussian, np.full(n_samples_per_cluster, i)])
    
    moon_offsets = [
        [10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0],                # Pair 1
        [0, 0, 0, 0, 0, -10, -10, -10, -10, -10, 0, 0],            # Pair 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15],                   # Pair 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15]                    # Pair 4
    ]
    
    if n_moon_pairs > len(moon_offsets):
        warnings.warn(f"Only {len(moon_offsets)} moon pairs available, using first {len(moon_offsets)}")
        n_moon_pairs = len(moon_offsets)
    
    X_moons = np.empty((0, dim))
    y_moons = np.empty(0, dtype=int)
    
    for i in range(n_moon_pairs):
        X_pair, y_pair = make_12d_moons(
            n_samples=2 * n_samples_per_cluster,
            noise=0.15,
            rotation=True,
            random_state=42 + i
        )
        
        X_pair += np.array(moon_offsets[i][:dim])
        
        y_pair += n_blobs + 2 * i  # Start labeling after Gaussian clusters
        
        X_moons = np.vstack([X_moons, X_pair])
        y_moons = np.concatenate([y_moons, y_pair])
    
    X = np.vstack([X_gaussian, X_moons])
    y = np.concatenate([y_gaussian, y_moons])
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

#################################
### For Medium - Reals : 6, 2 ###
#################################

def make_10d_moons(n_samples: int = 1000, 
                   noise: int = 0.15, 
                   rotation: int = True, 
                   random_state: int = None) -> Tuple[ndarray, ndarray]:
    """ Generate intertwined 10-dimensional moon-shaped clusters. """
    np.random.seed(random_state)
    
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_10d = np.zeros((n_samples, 10))
    
    r = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)
    theta = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    X_10d[:, 0] = 2.8 * X_2d[:, 0]
    X_10d[:, 1] = 1.7 * X_2d[:, 1] + 0.9 * np.sin(X_2d[:, 0] * np.pi)
    X_10d[:, 2] = 1.5 * r * np.sin(2 * theta) - 0.8 * np.cos(X_2d[:, 0] * np.pi)
    X_10d[:, 3] = 0.7 * (X_2d[:, 0] * X_2d[:, 1]) + 1.2 * np.sin(r)
    X_10d[:, 4] = np.exp(-0.3 * r) * np.cos(3 * theta)
    X_10d[:, 5] = np.log(1 + np.abs(X_2d[:, 0])) * np.sign(X_2d[:, 1])
    X_10d[:, 6] = 0.5 * X_2d[:, 0]**2 - 0.8 * X_2d[:, 1]**2
    X_10d[:, 7] = np.sin(X_2d[:, 0]) * np.cos(X_2d[:, 1]) * r
    X_10d[:, 8] = np.tanh(0.4 * X_2d[:, 0]) + 0.6 * np.arctan(X_2d[:, 1])
    X_10d[:, 9] = np.sqrt(np.abs(X_2d[:, 0] + X_2d[:, 1])) * np.sin(2 * np.pi * r)
    
    noise_vec = np.random.normal(0, noise, (n_samples, 10))
    X_10d += noise_vec
    
    if rotation:
        rotation_matrix = ortho_group.rvs(dim=10, random_state=random_state)
        X_10d = X_10d @ rotation_matrix
    
    return X_10d, y

def data_generator10(n_blobs: int = 3, 
                     n_moons: int = 2, 
                     dim: int = 10, 
                     n_samples_per_cluster: int = 300) -> Tuple[ndarray, ndarray]:
    """ Generate a dataset with Gaussian blobs and moon pairs in 10D space."""
    if dim != 10:
        raise ValueError("This generator only supports dim=10")
    
    if n_moons % 2 != 0:
        raise ValueError("n_moons must be even (each moon pair counts as 2 clusters)")
    
    np.random.seed(42)
    n_moon_pairs = n_moons // 2
    
    centers = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                    # Center 1
        [6, 6, 6, 6, 6, 0, 0, 0, 0, 0],                     # Center 2
        [-4, -4, -4, 0, 0, 4, 4, 0, 0, 0],                  # Center 3
        [0, 0, 0, 0, 0, 8, 8, 8, 8, 0],                     # Center 4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -6]                     # Center 5
    ]
    
    cluster_stds = [
        [1.4, 1.0, 1.7, 2.3, 1.2, 1.1, 1.3, 1.5, 0.9, 1.6],
        [1.1, 1.5, 0.9, 1.3, 2.0, 2.5, 1.2, 1.0, 1.6, 1.4],
        [1.7, 1.2, 2.3, 1.0, 1.4, 0.9, 1.5, 2.0, 2.5, 1.1],
        [1.3, 1.1, 1.5, 0.8, 1.2, 1.6, 1.0, 1.4, 1.7, 2.1],
        [1.0, 1.4, 1.2, 1.5, 1.1, 1.3, 1.7, 1.2, 0.8, 1.5]
    ]
    
    if n_blobs > len(centers):
        warnings.warn(f"Only {len(centers)} Gaussian clusters available, using first {len(centers)}")
        n_blobs = len(centers)
    
    X_gaussian = np.empty((0, dim))
    y_gaussian = np.empty(0, dtype=int)
    
    for i in range(n_blobs):
        cov = np.diag(np.square(cluster_stds[i][:dim]))
        cluster = np.random.multivariate_normal(
            centers[i][:dim], 
            cov, 
            size=n_samples_per_cluster
        )
        X_gaussian = np.vstack([X_gaussian, cluster])
        y_gaussian = np.concatenate([y_gaussian, np.full(n_samples_per_cluster, i)])
    
    moon_offsets = [
        [10, 10, 10, 10, 10, 0, 0, 0, 0, 0],               # Pair 1
        [0, 0, 0, 0, 0, -10, -10, -10, -10, -10],          # Pair 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 15],                   # Pair 3 (last dimension)
        [-10, 0, 0, 0, 0, 10, 0, 0, 0, 0]                  # Pair 4
    ]
    
    if n_moon_pairs > len(moon_offsets):
        warnings.warn(f"Only {len(moon_offsets)} moon pairs available, using first {len(moon_offsets)}")
        n_moon_pairs = len(moon_offsets)
    
    X_moons = np.empty((0, dim))
    y_moons = np.empty(0, dtype=int)
    
    for i in range(n_moon_pairs):
        # Generate moon pair
        X_pair, y_pair = make_10d_moons(
            n_samples=2 * n_samples_per_cluster,
            noise=0.05,
            rotation=True,
            random_state=42 + i
        )
        
        X_pair += np.array(moon_offsets[i][:dim])

        y_pair += n_blobs + 2 * i  # Start labeling after Gaussian clusters
        
        X_moons = np.vstack([X_moons, X_pair])
        y_moons = np.concatenate([y_moons, y_pair])
    
    X = np.vstack([X_gaussian, X_moons])
    y = np.concatenate([y_gaussian, y_moons])
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

############################
### Medium - Cats : 2, 6 ###
############################

def make_14d_moons(n_samples: int = 1000, 
                   noise: float = 0.15, 
                   rotation: bool = True, 
                   random_state: int = None) -> Tuple[ndarray, ndarray]:
    """ Generate intertwined 14-dimensional moon-shaped clusters."""
    np.random.seed(random_state)
    
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_14d = np.zeros((n_samples, 14))
    
    r = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)
    theta = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    X_14d[:, 0] = 2.8 * X_2d[:, 0]
    X_14d[:, 1] = 1.7 * X_2d[:, 1] + 0.9 * np.sin(X_2d[:, 0] * np.pi)
    X_14d[:, 2] = 1.5 * r * np.sin(2 * theta) - 0.8 * np.cos(X_2d[:, 0] * np.pi)
    X_14d[:, 3] = 0.7 * (X_2d[:, 0] * X_2d[:, 1]) + 1.2 * np.sin(r)
    X_14d[:, 4] = np.exp(-0.3 * r) * np.cos(3 * theta)
    X_14d[:, 5] = np.log(1 + np.abs(X_2d[:, 0])) * np.sign(X_2d[:, 1])
    X_14d[:, 6] = 0.5 * X_2d[:, 0]**2 - 0.8 * X_2d[:, 1]**2
    X_14d[:, 7] = np.sin(X_2d[:, 0]) * np.cos(X_2d[:, 1]) * r
    X_14d[:, 8] = np.tanh(0.4 * X_2d[:, 0]) + 0.6 * np.arctan(X_2d[:, 1])
    X_14d[:, 9] = np.sqrt(np.abs(X_2d[:, 0] + X_2d[:, 1])) * np.sin(2 * np.pi * r)
    X_14d[:, 10] = (X_2d[:, 0] * np.cos(theta) - X_2d[:, 1] * np.sin(theta)) * 1.5
    X_14d[:, 11] = 0.8 * np.sin(4 * theta) * r**0.7
    X_14d[:, 12] = np.arctan2(X_2d[:, 1] * 0.8, X_2d[:, 0] * 1.2) * np.exp(-r/3)
    X_14d[:, 13] = 1.2 * (X_2d[:, 0] - X_2d[:, 1])**3 / (1 + r)
    
    noise_vec = np.random.normal(0, noise, (n_samples, 14))
    X_14d += noise_vec
    
    if rotation:
        rotation_matrix = ortho_group.rvs(dim=14, random_state=random_state)
        X_14d = X_14d @ rotation_matrix
    
    return X_14d, y

def data_generator14(n_blobs: int = 3, 
                     n_moons: int = 2, 
                     dim: int = 14, 
                     n_samples_per_cluster: int = 300) -> Tuple[ndarray, ndarray]:
    """ Generate a dataset with Gaussian blobs and moon pairs in 14D space."""
    if dim != 14:
        raise ValueError("This generator only supports dim=14")
    
    if n_moons % 2 != 0:
        raise ValueError("n_moons must be even (each moon pair counts as 2 clusters)")
    
    np.random.seed(42)
    n_moon_pairs = n_moons // 2
    
    centers = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # Center 1
        [6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # Center 2
        [-4, -4, -4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0],        # Center 3
        [0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0],           # Center 4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, -6, -6]         # Center 5
    ]
    
    cluster_stds = [
        [1.4, 1.0, 1.7, 2.3, 1.2, 1.1, 1.3, 1.5, 0.9, 1.6, 1.8, 1.0, 1.2, 1.1],
        [1.1, 1.5, 0.9, 1.3, 2.0, 2.5, 1.2, 1.0, 1.6, 1.4, 0.8, 1.3, 1.1, 1.7],
        [1.7, 1.2, 2.3, 1.0, 1.4, 0.9, 1.5, 2.0, 2.5, 1.1, 1.3, 1.6, 1.8, 1.2],
        [1.3, 1.1, 1.5, 0.8, 1.2, 1.6, 1.0, 1.4, 1.7, 2.1, 1.2, 1.5, 0.9, 1.3],
        [1.0, 1.4, 1.2, 1.5, 1.1, 1.3, 1.7, 1.2, 0.8, 1.5, 1.3, 1.6, 2.0, 1.4]
    ]
    
    if n_blobs > len(centers):
        warnings.warn(f"Only {len(centers)} Gaussian clusters available, using first {len(centers)}")
        n_blobs = len(centers)
    
    X_gaussian = np.empty((0, dim))
    y_gaussian = np.empty(0, dtype=int)
    
    for i in range(n_blobs):
        cov = np.diag(np.square(cluster_stds[i][:dim]))
        cluster = np.random.multivariate_normal(
            centers[i][:dim], 
            cov, 
            size=n_samples_per_cluster
        )
        X_gaussian = np.vstack([X_gaussian, cluster])
        y_gaussian = np.concatenate([y_gaussian, np.full(n_samples_per_cluster, i)])
    
    moon_offsets = [
        [10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # Pair 1
        [0, 0, 0, 0, 0, -10, -10, -10, -10, -10, 0, 0, 0, 0],    # Pair 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 15, 15],          # Pair 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15]             # Pair 4
    ]
    
    if n_moon_pairs > len(moon_offsets):
        warnings.warn(f"Only {len(moon_offsets)} moon pairs available, using first {len(moon_offsets)}")
        n_moon_pairs = len(moon_offsets)
    
    X_moons = np.empty((0, dim))
    y_moons = np.empty(0, dtype=int)
    
    for i in range(n_moon_pairs):
        X_pair, y_pair = make_14d_moons(
            n_samples=2 * n_samples_per_cluster,
            noise=0.15,
            rotation=True,
            random_state=42 + i
        )
        
        # Apply offset
        X_pair += np.array(moon_offsets[i][:dim])
        
        # Relabel moons
        y_pair += n_blobs + 2 * i  # Start labeling after Gaussian clusters
        
        X_moons = np.vstack([X_moons, X_pair])
        y_moons = np.concatenate([y_moons, y_pair])
    
    X = np.vstack([X_gaussian, X_moons])
    y = np.concatenate([y_gaussian, y_moons])
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

##############################
### Small - Balanced : 2,2 ###
##############################

def make_6d_moons(n_samples: int = 1000, 
                  noise: int = 0.15, 
                  rotation: bool = True, 
                  random_state: int = None) -> Tuple[ndarray, ndarray]:
    """ Generate intertwined 6-dimensional moon-shaped clusters."""  
    np.random.seed(random_state)
    
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_6d = np.zeros((n_samples, 6))
    
    r = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)
    theta = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    X_6d[:, 0] = 2.8 * X_2d[:, 0]  # Scaled x-coordinate
    X_6d[:, 1] = 1.7 * X_2d[:, 1] + 0.9 * np.sin(X_2d[:, 0] * np.pi)  # Scaled y with sinusoidal modulation
    X_6d[:, 2] = 1.5 * r * np.sin(2 * theta) - 0.8 * np.cos(X_2d[:, 0] * np.pi)  # Radial-angular combination
    X_6d[:, 3] = 0.7 * (X_2d[:, 0] * X_2d[:, 1]) + 1.2 * np.sin(r)  # Product with radial sine
    X_6d[:, 4] = np.exp(-0.3 * r) * np.cos(3 * theta)  # Radial basis function
    X_6d[:, 5] = 0.5 * X_2d[:, 0]**2 - 0.8 * X_2d[:, 1]**2  # Quadratic polynomial
    
    noise_vec = np.random.normal(0, noise, (n_samples, 6))
    X_6d += noise_vec
    
    if rotation:
        rotation_matrix = ortho_group.rvs(dim=6, random_state=random_state)
        X_6d = X_6d @ rotation_matrix
    
    return X_6d, y

def data_generator6(n_blobs: int = 3, 
                    n_moons: int = 2, 
                    dim: int = 6, 
                    n_samples_per_cluster: int = 300) -> Tuple[ndarray, ndarray]:
    """ Generate a dataset with Gaussian blobs and moon pairs in 6D space."""
    if dim != 6:
        raise ValueError("This generator only supports dim=6")
    
    if n_moons % 2 != 0:
        raise ValueError("n_moons must be even (each moon pair counts as 2 clusters)")
    
    np.random.seed(42)
    n_moon_pairs = n_moons // 2
    
    centers = [
        [0, 0, 0, 0, 0, 0],                    # Center 1
        [6, 6, 6, 0, 0, 0],                     # Center 2
        [-4, -4, 0, 4, 4, 0],                   # Center 3
        [0, 0, 8, 0, 0, -6]                     # Center 4
    ]
    
    cluster_stds = [
        [1.4, 1.0, 1.7, 1.2, 1.1, 1.3],
        [1.1, 1.5, 0.9, 2.0, 1.2, 1.0],
        [1.7, 1.2, 2.3, 1.0, 1.4, 0.9],
        [1.3, 1.1, 1.5, 0.8, 1.2, 1.6]
    ]
    
    if n_blobs > len(centers):
        warnings.warn(f"Only {len(centers)} Gaussian clusters available, using first {len(centers)}")
        n_blobs = len(centers)
    
    X_gaussian = np.empty((0, dim))
    y_gaussian = np.empty(0, dtype=int)
    
    for i in range(n_blobs):
        cov = np.diag(np.square(cluster_stds[i][:dim]))
        cluster = np.random.multivariate_normal(
            centers[i][:dim], 
            cov, 
            size=n_samples_per_cluster
        )
        X_gaussian = np.vstack([X_gaussian, cluster])
        y_gaussian = np.concatenate([y_gaussian, np.full(n_samples_per_cluster, i)])
    
    moon_offsets = [
        [10, 10, 0, 0, 0, 0],                 # Pair 1
        [0, 0, -10, 0, 0, 0],                 # Pair 2
        [0, 0, 0, 15, 15, 0],                 # Pair 3
        [0, 0, 0, 0, 0, -15]                  # Pair 4
    ]
    
    if n_moon_pairs > len(moon_offsets):
        warnings.warn(f"Only {len(moon_offsets)} moon pairs available, using first {len(moon_offsets)}")
        n_moon_pairs = len(moon_offsets)
    
    X_moons = np.empty((0, dim))
    y_moons = np.empty(0, dtype=int)
    
    for i in range(n_moon_pairs):
    
        X_pair, y_pair = make_6d_moons(
            n_samples=2 * n_samples_per_cluster,
            noise=0.05,
            rotation=True,
            random_state=42 + i
        )
        
        # Apply offset
        X_pair += np.array(moon_offsets[i][:dim])
        
        # Relabel moons
        y_pair += n_blobs + 2 * i  # Start labeling after Gaussian clusters
        
        X_moons = np.vstack([X_moons, X_pair])
        y_moons = np.concatenate([y_moons, y_pair])
    
    X = np.vstack([X_gaussian, X_moons])
    y = np.concatenate([y_gaussian, y_moons])
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

############################
### Small - Reals : 3, 1 ###
############################

def make_5d_moons(n_samples: int = 1000, 
                  noise: int = 0.15, 
                  rotation: bool = True, 
                  random_state: int = None) -> Tuple[ndarray, ndarray]:
    """ Generate intertwined 5-dimensional moon-shaped clusters."""
    np.random.seed(random_state)
    
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_5d = np.zeros((n_samples, 5))
    
    r = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)
    theta = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    # Apply 5 distinct non-linear transformations
    X_5d[:, 0] = 2.8 * X_2d[:, 0]  # Scaled x-coordinate
    X_5d[:, 1] = 1.7 * X_2d[:, 1] + 0.9 * np.sin(X_2d[:, 0] * np.pi)  # Scaled y with sinusoidal modulation
    X_5d[:, 2] = 1.5 * r * np.sin(2 * theta) - 0.8 * np.cos(X_2d[:, 0] * np.pi)  # Radial-angular combination
    X_5d[:, 3] = 0.7 * (X_2d[:, 0] * X_2d[:, 1]) + 1.2 * np.sin(r)  # Product with radial sine
    X_5d[:, 4] = np.exp(-0.3 * r) * np.cos(3 * theta)  # Radial basis function
    
    noise_vec = np.random.normal(0, noise, (n_samples, 5))
    X_5d += noise_vec
    
    if rotation:
        rotation_matrix = ortho_group.rvs(dim=5, random_state=random_state)
        X_5d = X_5d @ rotation_matrix
    
    return X_5d, y

def data_generator5(n_blobs: int = 3, 
                    n_moons: int = 2, 
                    dim: int = 5, 
                    n_samples_per_cluster: int = 300) -> Tuple[ndarray, ndarray]:
    """ Generate a dataset with Gaussian blobs and moon pairs in 5D space."""
    if dim != 5:
        raise ValueError("This generator only supports dim=5")
    
    if n_moons % 2 != 0:
        raise ValueError("n_moons must be even (each moon pair counts as 2 clusters)")
    
    np.random.seed(42)
    n_moon_pairs = n_moons // 2
    
    centers = [
        [0, 0, 0, 0, 0],                    # Center 1
        [6, 6, 0, 0, 0],                     # Center 2
        [-4, 0, 4, 0, 0],                    # Center 3
        [0, 0, 0, 8, 0]                      # Center 4
    ]
    
    cluster_stds = [
        [1.4, 1.0, 1.7, 1.2, 1.1],
        [1.1, 1.5, 2.0, 1.2, 1.0],
        [1.7, 1.2, 1.0, 1.4, 0.9],
        [1.3, 1.1, 1.5, 0.8, 1.2]
    ]
    
    if n_blobs > len(centers):
        warnings.warn(f"Only {len(centers)} Gaussian clusters available, using first {len(centers)}")
        n_blobs = len(centers)
    
    X_gaussian = np.empty((0, dim))
    y_gaussian = np.empty(0, dtype=int)
    
    for i in range(n_blobs):
        cov = np.diag(np.square(cluster_stds[i][:dim]))
        cluster = np.random.multivariate_normal(
            centers[i][:dim], 
            cov, 
            size=n_samples_per_cluster
        )
        X_gaussian = np.vstack([X_gaussian, cluster])
        y_gaussian = np.concatenate([y_gaussian, np.full(n_samples_per_cluster, i)])
    
    moon_offsets = [
        [10, 10, 0, 0, 0],                 # Pair 1
        [0, 0, -10, 0, 0],                 # Pair 2
        [0, 0, 0, 15, 0],                  # Pair 3
        [0, 0, 0, 0, -15]                  # Pair 4
    ]
    
    if n_moon_pairs > len(moon_offsets):
        warnings.warn(f"Only {len(moon_offsets)} moon pairs available, using first {len(moon_offsets)}")
        n_moon_pairs = len(moon_offsets)
    
    X_moons = np.empty((0, dim))
    y_moons = np.empty(0, dtype=int)
    
    for i in range(n_moon_pairs):
        X_pair, y_pair = make_5d_moons(
            n_samples=2 * n_samples_per_cluster,
            noise=0.15,
            rotation=True,
            random_state=42 + i
        )
        
        X_pair += np.array(moon_offsets[i][:dim])
        
        y_pair += n_blobs + 2 * i  # Start labeling after Gaussian clusters
        
        X_moons = np.vstack([X_moons, X_pair])
        y_moons = np.concatenate([y_moons, y_pair])
    
    X = np.vstack([X_gaussian, X_moons])
    y = np.concatenate([y_gaussian, y_moons])
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

###########################
### Small - Cats : 1, 3 ###
###########################

def make_7d_moons(n_samples: int = 1000, 
                  noise: float = 0.15, 
                  rotation: bool = True, 
                  random_state: int = None) -> Tuple[ndarray, ndarray]:
    """ Generate intertwined 7-dimensional moon-shaped clusters."""
    np.random.seed(random_state)
    
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_7d = np.zeros((n_samples, 7))
    
    r = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)
    theta = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    X_7d[:, 0] = 2.8 * X_2d[:, 0]  # Scaled x-coordinate
    X_7d[:, 1] = 1.7 * X_2d[:, 1] + 0.9 * np.sin(X_2d[:, 0] * np.pi)  # Scaled y with sinusoidal modulation
    X_7d[:, 2] = 1.5 * r * np.sin(2 * theta) - 0.8 * np.cos(X_2d[:, 0] * np.pi)  # Radial-angular combination
    X_7d[:, 3] = 0.7 * (X_2d[:, 0] * X_2d[:, 1]) + 1.2 * np.sin(r)  # Product with radial sine
    X_7d[:, 4] = np.exp(-0.3 * r) * np.cos(3 * theta)  # Radial basis function
    X_7d[:, 5] = 0.5 * X_2d[:, 0]**2 - 0.8 * X_2d[:, 1]**2  # Quadratic polynomial
    X_7d[:, 6] = np.tanh(0.4 * X_2d[:, 0]) + 0.6 * np.arctan(X_2d[:, 1])  # Hyperbolic + inverse trig
    
    noise_vec = np.random.normal(0, noise, (n_samples, 7))
    X_7d += noise_vec
    
    if rotation:
        rotation_matrix = ortho_group.rvs(dim=7, random_state=random_state)
        X_7d = X_7d @ rotation_matrix
    
    return X_7d, y

def data_generator7(n_blobs: int = 3, 
                    n_moons: int = 2, 
                    dim: int = 7, 
                    n_samples_per_cluster: int = 300) -> Tuple[ndarray, ndarray]:
    """ Generate a dataset with Gaussian blobs and moon pairs in 7D space."""
    if dim != 7:
        raise ValueError("This generator only supports dim=7")
    
    if n_moons % 2 != 0:
        raise ValueError("n_moons must be even (each moon pair counts as 2 clusters)")
    
    np.random.seed(42)
    n_moon_pairs = n_moons // 2
    
    centers = [
        [0, 0, 0, 0, 0, 0, 0],                    # Center 1
        [6, 6, 6, 0, 0, 0, 0],                     # Center 2
        [-4, -4, 0, 4, 4, 0, 0],                   # Center 3
        [0, 0, 0, 0, 0, 8, 0],                     # Center 4
        [0, 0, 0, 0, 0, 0, -6]                     # Center 5
    ]
    
    cluster_stds = [
        [1.4, 1.0, 1.7, 1.2, 1.1, 1.3, 1.5],
        [1.1, 1.5, 0.9, 2.0, 1.2, 1.0, 1.6],
        [1.7, 1.2, 2.3, 1.0, 1.4, 0.9, 1.1],
        [1.3, 1.1, 1.5, 0.8, 1.2, 1.6, 1.4],
        [1.0, 1.4, 1.2, 1.5, 1.1, 1.3, 0.8]
    ]
    
    if n_blobs > len(centers):
        warnings.warn(f"Only {len(centers)} Gaussian clusters available, using first {len(centers)}")
        n_blobs = len(centers)
    
    X_gaussian = np.empty((0, dim))
    y_gaussian = np.empty(0, dtype=int)
    
    for i in range(n_blobs):
        cov = np.diag(np.square(cluster_stds[i][:dim]))
        cluster = np.random.multivariate_normal(
            centers[i][:dim], 
            cov, 
            size=n_samples_per_cluster
        )
        X_gaussian = np.vstack([X_gaussian, cluster])
        y_gaussian = np.concatenate([y_gaussian, np.full(n_samples_per_cluster, i)])
    
    moon_offsets = [
        [10, 10, 0, 0, 0, 0, 0],                 # Pair 1
        [0, 0, -10, 0, 0, 0, 0],                 # Pair 2
        [0, 0, 0, 15, 0, 0, 0],                  # Pair 3
        [0, 0, 0, 0, -15, 0, 0]                  # Pair 4
    ]
    
    if n_moon_pairs > len(moon_offsets):
        warnings.warn(f"Only {len(moon_offsets)} moon pairs available, using first {len(moon_offsets)}")
        n_moon_pairs = len(moon_offsets)
    
    X_moons = np.empty((0, dim))
    y_moons = np.empty(0, dtype=int)
    
    for i in range(n_moon_pairs):
        X_pair, y_pair = make_7d_moons(
            n_samples=2 * n_samples_per_cluster,
            noise=0.15,
            rotation=True,
            random_state=42 + i
        )
        
        # Apply offset
        X_pair += np.array(moon_offsets[i][:dim])
        
        # Relabel moons
        y_pair += n_blobs + 2 * i  # Start labeling after Gaussian clusters
        
        X_moons = np.vstack([X_moons, X_pair])
        y_moons = np.concatenate([y_moons, y_pair])
    
    X = np.vstack([X_gaussian, X_moons])
    y = np.concatenate([y_gaussian, y_moons])
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

