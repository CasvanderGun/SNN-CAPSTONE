import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
import os

# Specify the path to the directory
directory_count_ttfs = Path(r'C:\Users\hanna\Downloads\SNN-CAPSTONE-1\results\train_count_eval_ttfs\train_multiple_runsv2')
directory_ttfs_count = Path(r"C:\Users\hanna\Downloads\SNN-CAPSTONE-1\results\train_ttfs_eval_count\multiple_runs")
# data_directory_decay = Path(r"")
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
    for run_key, run_data in loaded_data.items():
        for file_data in run_data:
            if metric_name in file_data['filename']:
                action = {f'Epoch_{epoch}': value for epoch, value in zip(file_data['epochs'], file_data['values'])}
                metric_data.append({"Run":run_key, **action})
    return metric_data

def create_dataframes(metric_data: List[Dict[str, np.ndarray]], metric_name: str) -> pd.DataFrame:
    """Create a DataFrame from the metric_data and return it."""
    df = pd.DataFrame(metric_data)
    # df['Epoch'] = pd.to_numeric(df['epochs'].str.replace(f'Epoch_{metric_name}_', ''), errors='coerce')
    return df

def extract_stats(df):
    # Melt the DataFrame to have 'Epoch' as a separate column
    melted_df = pd.melt(df, id_vars=['Run'], var_name='Epoch', value_name='Value')

    # Convert 'Epoch' column to numeric for proper sorting
    melted_df['Epoch'] = pd.to_numeric(melted_df['Epoch'].str.replace('Epoch_', ''), errors='coerce')

    # Group by 'Epoch' and calculate mean and standard deviation
    result_df = melted_df.groupby('Epoch')['Value'].agg(['mean', 'std']).reset_index()

    # Rename columns for clarity
    result_df.columns = ['Epoch', 'mean', 'sd']
    return result_df

def make_data_dict(path, num_runs=5):
    data_dict = {}
    for run in range(num_runs):
        run_path = path / f'Run_{run + 1}/output_metrics'
        print(f' """     LOADING RUN {str(run + 1)}      """')
        data_dict[f'Run_{run + 1}'] = load_files(run_path)
    return data_dict

data_dict_count_ttfs = make_data_dict(directory_count_ttfs)
data_dict_ttfs_count = make_data_dict(directory_ttfs_count)

metric_names_count_ttfs = ['accuracy_train', 'loss_train', 'silent_neurons', 'accuracy_count_test', 'accuracy_ttfs_test',
                'loss_count_test', 'loss_ttfs_test', 'weight_norm_Hidden', 'weight_norm_Output']

metric_data_dict = {metric_name: extract_metric_data(data_dict, metric_name) for metric_name in metric_names_count_ttfs}

dataframes = {metric_name: create_dataframes(metric_data, metric_name) for metric_name, metric_data in metric_data_dict.items()}

stats_dataframes = {metric_name: extract_stats(df) for metric_name, df in dataframes.items()}