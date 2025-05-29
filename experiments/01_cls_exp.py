# Description: Script holding the experimenter module for the classification tack experiments.
# Author: Anton D. Lautrup
# Date: 20-05-2025

import os
import sys
sys.path.append(".")

import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

from sklearn.model_selection import StratifiedKFold, KFold

from experiments.prepare_data import preprocess_data, uci_dataset_id_import
from experiments.plots import plot_cross_validation_results
from experiments.KNN_adapters import KNNAdapter

from joblib import Parallel, delayed

def run_experiment(df_tuple: Tuple[DataFrame, str], KNN_method: KNNAdapter, best_k: int = None, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
    """ Run a classification experiment using the specified KNN method.

    Arguments:
    - df_tuple (Tuple[DataFrame, str]): A tuple containing the DataFrame and its name.
    - KNN_method (KNNAdapter): The KNN method to use for classification.
    - best_k (int): The number of neighbors to use for KNN. If None, the best k will be determined using cross-validation.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Random state for reproducibility.

    Returns:
    - Dictionary with accuracy, precision, recall, and F1 score.
    """
    # Unpack the materials
    df, df_name = df_tuple
    exp_name = KNN_method.name()

    X, y = df.drop(columns=['class']), df['class']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
 
    best_k = (np.sqrt(len(X))/2).astype(int)

    group_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    knn_model = KNN_method(n_neighbors=best_k)

    accuracy_scores, precision_scores, recall_scores, f1_scores, error_rates, roc_auc_scores = [], [], [], [], [], []
    for train_index, test_index in group_kfold.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        knn_model.fit_cls(X_train_fold, y_train_fold)
        y_pred = knn_model.predict(X_test_fold)
        y_score = knn_model.predict_proba(X_test_fold)

        accuracy = balanced_accuracy_score(y_test_fold, y_pred)
        precision = precision_score(y_test_fold, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_fold, y_pred, average='weighted')
        f1 = f1_score(y_test_fold, y_pred, average='weighted')
        err = 1 - np.mean(y_pred != y_test_fold)

        # update y_test to have the same format as y_score for roc_auc_score
        if len(y_test_fold.shape) == 1:
            y_test_fold_dummies = pd.get_dummies(y_test_fold)
            y_test_fold = y_test_fold_dummies.values

        roc_auc = roc_auc_score(y_test_fold, y_score, multi_class='ovr', average='weighted')
        
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        error_rates.append(err)
        roc_auc_scores.append(roc_auc)

    #dataset size category
    if len(X) < 100: data_size = 'small'
    elif len(X) < 1000: data_size = 'medium'
    else: data_size = 'large'

    # cls size
    if len(np.unique(y)) == 2: cls_task = 'binary'
    else: cls_task = 'multi'

    return {
        'method': exp_name,
        'test_size': test_size,
        'df_name': df_name,
        'best_k': best_k,
        'accuracy': np.mean(accuracy_scores),
        'accuracy_std': np.std(accuracy_scores, ddof=1),
        'precision': np.mean(precision_scores),
        'precision_std': np.std(precision_scores, ddof=1),
        'recall': np.mean(recall_scores),
        'recall_std': np.std(recall_scores, ddof=1),
        'f1_score': np.mean(f1_scores),
        'f1_score_std': np.std(f1_scores, ddof=1),
        'error_rate': np.mean(error_rates),
        'error_rate_std': np.std(error_rates, ddof=1),
        'roc_auc': np.mean(roc_auc_scores),
        'roc_auc_std': np.std(roc_auc_scores, ddof=1),
        'data_size': data_size,
        'cls_task': cls_task,
        'seed': random_state,
    }

if __name__ == "__main__":
    import time
    import seaborn as sns
    import uci_dataset as dataset
    from experiments.KNN_adapters import (
        EuclideanKNN, GowerKNN, REX_KNN,
        EuclideanKNN_OneHot, HeomKNN, 
        GeneralisedEuclideanKNN, HvdmKNN
    )

    OVERWRITE = False
    results_file = '01_knn_cls_results.csv'

    datasets = {
        'autism' : (dataset.load_autism_screening(), "Class/ASD"),
        'balance_scale' : (uci_dataset_id_import(12), "class"),
        'breast_cancer': (dataset.load_breast_cancer(), "Class"),
        'cervical_cancer': (dataset.load_cervical_cancer(), "Biopsy"),
        'cirrhosis': (uci_dataset_id_import(878), "class"),
        'credit_approval': (dataset.load_credit_approval(), "A16"),
        'cylinder_bands': (dataset.load_cylinder_bands(), "band type"),
        'dermatology': (dataset.load_dermatology(), "class"),
        'diabetic_retino': (dataset.load_diabetic(), "Class"),
        'early_diabetes': (dataset.load_early_stage_diabetes_risk(), "class"),
        'fertility': (dataset.load_fertility(), "Diagnosis"),
        'glass' : (uci_dataset_id_import(42), "class"),
        'german_credit' : (uci_dataset_id_import(144), "class"),
        'haberman': (dataset.load_haberman(), "survival"),
        'hayes_roth': (dataset.load_hayes_roth(), "class"),
        'hcv_values': (dataset.load_hcv(), "Category"),
        'heart': (uci_dataset_id_import(145), 'class'),
        'heart_disease': (dataset.load_heart_disease(), "target"),
        'hepatitis': (dataset.load_hepatitis(), "Class"),
        'indian_liver': (dataset.load_indian_liver(), "Selector"),
        'iris': (sns.load_dataset('iris'), "species"),
        'liver_disorder': (dataset.load_liver_disorders(), "selector"),
        'kidney_disease': (uci_dataset_id_import(336), "class"),
        'lymphography': (dataset.load_lymphography(), "class"),
        'mammographic': (uci_dataset_id_import(161), "class"),
        'maternal': (uci_dataset_id_import(863), "class"),
        'mushroom': (uci_dataset_id_import(73), "class"),
        'obesity_levels': (uci_dataset_id_import(544), "class"),
        'parkinsons': (dataset.load_parkinson(), "status"),
        'penguins': (sns.load_dataset('penguins'), "species"),
        'raisin': (uci_dataset_id_import(850), "class"),
        'soy_bean': (uci_dataset_id_import(90), "class"),
        'student_performance': (uci_dataset_id_import(856), "class"),
        'thoracic_surgery': (dataset.load_thoracic_surgery(), "Risk1Yr"),
        'voting' : (uci_dataset_id_import(105), "class"),
        'wisconsin_bc' : (dataset.load_breast_cancer_wis_diag(), "diagnosis"),
    }

    experiments = {
        'L2': EuclideanKNN,
        'L2_OHE': EuclideanKNN_OneHot,
        'Gower': GowerKNN,
        'HEOM': HeomKNN,
        'HVDM': HvdmKNN,
        'GEM' : GeneralisedEuclideanKNN,
        'REX': REX_KNN,
    }

    for dataset_name, (df, label) in datasets.items():
        df, cat_cols, type = preprocess_data(df, label)
        df_tuple = (df, dataset_name)

        for exp_name, method in experiments.items():
            time_start = time.time()
            results = run_experiment(df_tuple, method, random_state=42)
            time_end = time.time()
            print(f"Time taken: {time_end - time_start:.2f} seconds")
            # print(results)
            results['type'] = type

            # Save results to CSV
            results_df = pd.DataFrame([results])
            if OVERWRITE or not os.path.exists(results_file):
                results_df.to_csv(results_file, mode='w', index=False)
                OVERWRITE = False
            else:
                results_df.to_csv(results_file, mode='a', header=False, index=False)
