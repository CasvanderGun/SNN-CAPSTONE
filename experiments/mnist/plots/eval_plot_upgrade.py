import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
import os

# Specify the path to the directory
data_directory = Path(r'C:\Users\hanna\Downloads\SNN-CAPSTONE-1\results\train_multiple_runsv2')
num_runs = 5

def load_files(dic_path):
    """ Loading .npz files from directory and storing as useable data"""
    loaded_data = []
    for filename in os.listdir(dic_path):
        file_path = os.path.join(dic_path, filename)

        # Check if the path is a file and has a .npz extension
        if os.path.isfile(file_path) and filename.endswith('.npz'):
            print(f"Loading data from file: {filename}")

            data = np.load(file_path)
            loaded_data.append({
                "filename": filename,
                "epochs": data['epochs'],
                "values": data["values"]
            })
    return loaded_data

def extract_metric_data(loaded_data: Dict[str, np.ndarray], metric_name: str) -> List[Dict[str, np.ndarray]]:
    """Extract metric data from loaded_data and return a list of dictionaries."""
    metric_data = []
    for file_data in loaded_data.values():
        if metric_name in file_data['filename']:
            action = {f'Epoch_{epoch}': value for epoch, value in zip(file_data['epochs'], file_data['values'])}
            metric_data.append({"Run": file_data["filename"].split('_')[0], **action})
    return metric_data

def create_dataframes(metric_data: List[Dict[str, np.ndarray]], metric_name: str) -> pd.DataFrame:
    """Create a DataFrame from the metric_data and return it."""
    df = pd.DataFrame(metric_data)
    df['Epoch'] = pd.to_numeric(df['Epoch'].str.replace(f'Epoch_{metric_name}_', ''), errors='coerce')
    return df

def extract_stats_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract statistics from the DataFrame and return a new DataFrame."""
    melted_df = pd.melt(df, id_vars=['Run'], var_name='Epoch', value_name='Value')
    melted_df['Epoch'] = pd.to_numeric(melted_df['Epoch'].str.replace('Epoch_', ''), errors='coerce')
    result_df = melted_df.groupby('Epoch')['Value'].agg(['mean', 'std']).reset_index()
    result_df.columns = ['Epoch', 'mean', 'sd']
    return result_df

data_dict = {}
for run in range(num_runs):
    run_path = data_directory / f'Run_{run + 1}/output_metrics'
    print(f' """     LOADING RUN {str(run + 1)}      """')
    data_dict[f'Run_{run + 1}'] = load_files(run_path)

metric_names = ['accuracy_train', 'loss_train', 'silent_neurons', 'accuracy_count_test', 'accuracy_ttfs_test',
                'loss_count_test', 'loss_ttfs_test', 'weight_norm_Hidden', 'weight_norm_Output']

metric_data_dict = {metric_name: extract_metric_data(data_dict, metric_name) for metric_name in metric_names}

dataframes = {metric_name: create_dataframes(metric_data, metric_name) for metric_name, metric_data in metric_data_dict.items()}

stats_dataframes = {metric_name: extract_stats_from_df(df) for metric_name, df in dataframes.items()}