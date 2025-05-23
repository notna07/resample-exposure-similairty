# Description: Script for accessing the SynthPop R package for generating synthetic data.
# Author: Anton D. Lautrup
# Date: 23-05-2025

import os
import subprocess
import pandas as pd

from typing import List
from pandas import DataFrame

def _load_data(file_name: str) -> DataFrame:
    df_train = pd.read_csv(file_name + '.csv').dropna()
    return df_train

def _write_data(df: DataFrame, file_name: str) -> None:
    df.to_csv(file_name, index=False)

def _cleanup_files(file_names: List[str]) -> None:
    for file_name in file_names:
        if os.path.exists(file_name):
            os.remove(file_name)
    pass

def rSynthpop(train_data: str | DataFrame, num_to_generate: int = None, seed: int = None, id = 0,  **kwargs) -> DataFrame:
    """ Generate synthetic data using SynthPop in R using subprocess.
    Be sure to check that R is installed and Rscript is a valid command in the terminal.

    Reference:
        Nowok, B., Raab, G. M., & Dibben, C. (2016). synthpop: Bespoke Creation of Synthetic Data in R. 
        Journal of Statistical Software, 74(11), 1--26. https://doi.org/10.18637/jss.v074.i11

    Arguments:
        - train_data (str, DataFrame): The name of the training data file or the DataFrame.
        - num_to_generate (int): The number of synthetic data points to generate.
        - seed (int): The random seed for reproducibility.
        - id (int): An identifier for the temporary file name.

    Returns:
        DataFrame: The generated synthetic data.

    Example:
        >>> df_syn = rSynthpop('tests/dummy_train')
        >>> isinstance(df_syn, pd.DataFrame)
        True
    """
    if isinstance(train_data, str):
        train_data_name = train_data
    else:
        train_data_name = f'synthpop_temp_{id}'
        _write_data(train_data, train_data_name + '.csv')

    df_train = _load_data(train_data_name)

    info_dir = 'synthesis_info_' + train_data_name.split('/')[0]
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)

    command = [
                "Rscript",
                "synthpop_subprocess.R",
                train_data_name +".csv",
                train_data_name + "_synthpop",
                str(num_to_generate) if num_to_generate is not None else str(len(df_train)),
                str(seed) if seed is not None else "",
            ]
    subprocess.run(command, check=True)

    df_syn = pd.read_csv(train_data_name + '_synthpop.csv')
    df_syn.columns = [col for col in df_train.columns]

    _cleanup_files(['synthesis_info_' + train_data_name + '_synthpop.txt', 
                    train_data_name + '_synthpop.csv', 
                    f'synthpop_temp_{id}.csv'])

    os.removedirs(info_dir)
    return df_syn