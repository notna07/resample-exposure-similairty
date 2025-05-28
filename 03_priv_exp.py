# Description: Script holding the experimenter module for the privacy experiments.
# Author: Anton D. Lautrup
# Date: 26-05-2025

import os
import copy

import numpy as np
import pandas as pd

import uci_dataset as dataset
from synthpop_adapter import rSynthpop

from privacy_metrics import distance_to_closest_record, epsilon_identifiability_risk
from prepare_data import preprocess_data, uci_dataset_id_import, _clean_up_data

NUM_REPEATS = 5

def run_experiment(df_tuple, KNN_method, random_state=42):
    """ Run a privacy experiment using the specified KNN method.

    Arguments:
    - df_tuple (Tuple[DataFrame, str]): A tuple containing the DataFrame and its name.
    - KNN_method (KNNAdapter): The KNN method to use for classification.
    - random_state (int): Random state for reproducibility.

    Returns:
    - Dictionary with dataset name, average distance and epsilon risk
    """
    df_tuple_lab, df_name = df_tuple
    df, label = df_tuple_lab

    exp_name = KNN_method.name()

    if exp_name == 'L2':
        df, cat_cols, type = preprocess_data(df.copy(), label)
    else:
        df = _clean_up_data(df.copy(), label)
    
    if exp_name == 'GOW':
        # convert integers to float to prevent issues with Gower distance
        df = df.copy().astype({col: 'float64' for col in df.select_dtypes(include=['int']).columns})

    # print(df.head())
    dist_list, dcrs_list, eps_list = [], [], []
    for i in range(NUM_REPEATS):
        df_syn = rSynthpop(df, seed=random_state*i)

        if exp_name == 'GOW' or exp_name == 'GOW':
            # convert integers to float to prevent issues with Gower distance
            df_syn = df_syn.copy().astype({col: 'float64' for col in df_syn.select_dtypes(include=['int']).columns})

        self_dists = distance_to_closest_record(df, df, KNN_method)
        syn_dists = distance_to_closest_record(df_syn, df, KNN_method)

        dcr = syn_dists['mean']/self_dists['mean']

        eps = epsilon_identifiability_risk(df_syn, df, KNN_method)

        dist_list.append(syn_dists['mean'])
        dcrs_list.append(dcr)
        eps_list.append(eps)

    #dataset size category
    if len(df) < 100: data_size = 'small'
    elif len(df) < 1000: data_size = 'medium'
    else: data_size = 'large'

    return {
        'method': exp_name,
        'df_name': df_name,
        'avg_distance': np.mean(dist_list),
        'std_distance': np.std(dist_list, ddof=1),
        'avg_dcr': np.mean(dcrs_list),
        'std_dcr': np.std(dcrs_list, ddof=1),
        'avg_epsilon': np.mean(eps_list),
        'std_epsilon': np.std(eps_list, ddof=1),
        'data_size': data_size,
        'seed': random_state,
    }


if __name__ == "__main__":
    import time
    import seaborn as sns
    import uci_dataset as dataset
    from KNN_adapters import (
        EuclideanKNN, GowerKNN, REX_KNN,
    )

    OVERWRITE = False
    results_file = '03_privacy_results.csv'

    datasets = {
        'autism' : (dataset.load_autism_screening(), "Class/ASD"),
        # 'balance_scale' : (uci_dataset_id_import(12), "class"),
        # 'breast_cancer': (dataset.load_breast_cancer(), "Class"),
        # 'cervical_cancer': (dataset.load_cervical_cancer(), "Biopsy"),
        'cirrhosis': (uci_dataset_id_import(878), "class"),
        'credit_approval': (dataset.load_credit_approval(), "A16"),
        'cylinder_bands': (dataset.load_cylinder_bands(), "band type"),
        'dermatology': (dataset.load_dermatology(), "class"),
        'diabetic_retino': (dataset.load_diabetic(), "Class"),
        # 'early_diabetes': (dataset.load_early_stage_diabetes_risk(), "class"),
        # 'fertility': (dataset.load_fertility(), "Diagnosis"),
        'glass' : (uci_dataset_id_import(42), "class"),
        'german_credit' : (uci_dataset_id_import(144), "class"),
        # 'haberman': (dataset.load_haberman(), "survival"),
        # 'hayes_roth': (dataset.load_hayes_roth(), "class"),
        # 'hcv_values': (dataset.load_hcv(), "Category"),
        # 'heart': (uci_dataset_id_import(145), 'class'),
        # 'heart_disease': (dataset.load_heart_disease(), "target"),
        # 'hepatitis': (dataset.load_hepatitis(), "Class"),
        # 'indian_liver': (dataset.load_indian_liver(), "Selector"),
        # 'iris': (sns.load_dataset('iris'), "species"),
        # 'liver_disorder': (dataset.load_liver_disorders(), "selector"),
        'kidney_disease': (uci_dataset_id_import(336), "class"),
        # 'lymphography': (dataset.load_lymphography(), "class"),
        # 'mammographic': (uci_dataset_id_import(161), "class"),
        'maternal': (uci_dataset_id_import(863), "class"),
        # 'mushroom': (uci_dataset_id_import(73), "class"),
        'obesity_levels': (uci_dataset_id_import(544), "class"),
        # 'parkinsons': (dataset.load_parkinson(), "status"),
        # 'penguins': (sns.load_dataset('penguins'), "species"),
        'raisin': (uci_dataset_id_import(850), "class"),
        # 'soy_bean': (uci_dataset_id_import(90), "class"),
        # 'student_performance': (uci_dataset_id_import(856), "class"),
        'thoracic_surgery': (dataset.load_thoracic_surgery(), "Risk1Yr"),
        # 'voting' : (uci_dataset_id_import(105), "class"),
        'wisconsin_bc' : (dataset.load_breast_cancer_wis_diag(), "diagnosis"),
    }

    experiments = {
        # 'L2': EuclideanKNN,
        # 'Gower': GowerKNN,
        'REX': REX_KNN,
    }

    for dataset_name, df_tuple in datasets.items():
        # df, cat_cols, type = preprocess_data(df, label)
        df_tuple_nm = (df_tuple, dataset_name)

        for exp_name, method in experiments.items():
            try:
                time_start = time.time()
                results = run_experiment(df_tuple_nm, method, random_state=42)
                time_end = time.time()
                print(f"Time taken: {time_end - time_start:.2f} seconds")

                # Save results to CSV
                results_df = pd.DataFrame([results])
                if OVERWRITE or not os.path.exists(results_file):
                    results_df.to_csv(results_file, mode='w', index=False)
                    OVERWRITE = False
                else:
                    results_df.to_csv(results_file, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Error processing {dataset_name} with {exp_name}: {e}")
                continue