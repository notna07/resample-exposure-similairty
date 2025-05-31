# Resample Exposure Similarity

Resample Exposure similarity is a novel approach to measuring similarity in heterogeneous data based on the frequency of categorical variables and the density distribution of numericals. This library provides a rough implementation of the measure as a class that can compute a similarity matrix. Additional code is included in the repository, for implementing and experimenting with resample exposure similarity and competitors, including functions for calculating similarity matrices, performing nearest neighbour classification, and clustering using partitioning around medoids.

## Installation

 To install the library, run the following command:

```bash
pip install 
```

## Tutorial

```	python
import seaborn as sns
from resample_exposure import ResampleExposure

df = sns.load_dataset('penguins')
df_train, df_test = df.iloc[:300], df.iloc[300:]

# Create an instance of the ResampleExposure class
# categorical features need not be specified, and will be automatically inferred
rex = ResampleExposure(target_distribution=df_train,
                       categorical_features=['species', 'island'], 
                       unique_threshold=5, 
                       feature_weights=None,
                       )

# single point comparison
similarity = rex.resample_exposure_sim(query_point=df_test.iloc[0], 
                                        target_point=df_train.iloc[0], 
                                        normalised=True
                                        )

# similarity matrix
similarity_matrix = rex.resample_exposure_matrix(query_df=df_test,
                                                  normalised = False, 
                                                  reverse_direction = False,
                                                  overwrite_memory = False,
                                                  n_jobs: int = -1
                                                  )
```
For the calculation of the similarity matrix, if no arguments are given it will return the similarity matrix of the target distribution with itself. Reverse direction is to use the query distribution as the target distribution, and the target distribution as the query distribution. Overwrite memory is an experimental feature to overwrite the memory of the marginal distributions with those of the queries in case the distribution of the target points are assumed to be unknown. The `n_jobs` parameter allows for parallel computation, where `-1` uses all available cores. 

## Experiment Codebooks
This library is accompanied by a set of codebooks that demonstrate how to use the library and replicate the results shown in the paper "Similarity Based on Resample Exposure". The codebooks are designed to be used in Jupyter notebooks and provide step-by-step instructions for running experiments and generating synthetic data.
 
Below is codebooks that can be used to replicate the results shown in the paper.
| Link | Description | Fig. refs. |
| --- | --- | --- |
| [Codebook 1](01_dataset_char.ipynb) | Figures and experiments exploring behaviour of resample exposure | Fig.3 |
| [Codebook 2](02_nearest_neighbours.ipynb) | Experiments and results for nearest neighbours classification | Fig.4 |
| [Codebook 3](03_clustering_medoids.ipynb) | Experiments and results for the partitioning around medoids | Fig.5 |


## Requirements

