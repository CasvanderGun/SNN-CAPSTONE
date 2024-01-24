import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Sequence
import os

# Specify the path to the directory
directory_count_ttfs = Path(r'C:\Users\hanna\Downloads\SNN-CAPSTONE-1\results\train_count_eval_ttfs\train_multiple_runsv2')
directory_ttfs_count = Path(r"C:\Users\hanna\Downloads\SNN-CAPSTONE-1\results\train_ttfs_eval_count\multiple_runs")
directory_decay = Path(r"C:\Users\hanna\Downloads\SNN-CAPSTONE-1\results\train_decay_rate\multiple_runs")
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
    melted_df = pd.melt(df, id_vars=['Run'], var_name='Epoch', value_name='Value')
    melted_df['Epoch'] = pd.to_numeric(melted_df['Epoch'].str.replace('Epoch_', ''), errors='coerce')
    result_df = melted_df.groupby('Epoch')['Value'].agg(['mean', 'std']).reset_index()
    result_df.columns = ['Epoch', 'mean', 'sd']
    return result_df


def make_data_dict(path, num_runs=5):
    data_dict = {}
    for run in range(num_runs):
        run_path = path / f'Run_{run + 1}/output_metrics'
        print(f' """     LOADING RUN {str(run + 1)}      """')
        data_dict[f'Run_{run + 1}'] = load_files(run_path)
    return data_dict

def create_line_plot(df, x_name: str='Epoch', y_name: str='mean', r='sd', title: str = "", 
                     label: list[str] = None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=x_name, y=y_name, data=df, errorbar="sd", label=label)
    plt.fill_between(df[x_name], df[y_name] - df[r], df[y_name] + df[r], 
                     alpha=0.4, label='Confidence Interval')
    plt.title(title + 'Confidence Intervals over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Value')
    plt.legend()
    plt.show()

def create_line_plot_multiple(df_list: list[pd.DataFrame], x_name: str, y_name: str, r: str='sd', title: str | None = None, 
                              ylabel: str | None = None, labels: list[str] = None, set_limit:  bool=False, 
                              blimit: float=0.0, tlimit: float=100, loc="lower right", rlimit: str | None=None,
                              legend_outside_grid: bool=False, style: str="whitegrid", path: str="") -> None:
    """ Function to plot multiple lines in a graph. The input must be a list containing the pd.DataFrames you want to make a graph of.
    Use the x_name and y_name to specify the columns in the dataframe to get the data you want to use in the plot."""
    sns.set(style=style)
    if legend_outside_grid:
        plt.figure(figsize=(16.4, 6))
    else:
        plt.figure(figsize=(12, 6))

    for df, label in zip(df_list, labels):
        sns.lineplot(x=x_name, y=y_name, data=df, errorbar=r, label=label)
        plt.fill_between(df[x_name], df[y_name] - df[r], df[y_name] + df[r], 
                         alpha=0.4)
        
    if legend_outside_grid:
        plt.subplots_adjust(right=0.7)
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
    else:
        plt.legend(loc=loc)
    plt.title(title + ' with confidence intervals over runs')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel=ylabel)
    if set_limit:
        plt.ylim(bottom=blimit, top=tlimit)
        plt.xlim(right=rlimit)
    if path != "":
        plt.savefig(path + "/" + title)
    plt.show()

def get_best_acc(df: pd.DataFrame, col_name1: str='mean', col_name2: str='sd') -> tuple[float, float]:
    """ Function to get the best accuracy in the given pandas dataframe with the standard deviation at the location"""
    loc = df.loc[df[col_name1].idxmax()]
    return loc['Epoch'], loc[col_name1], loc[col_name2]

def get_dfs_to_list(dfs: list[dict[str, pd.DataFrame]], metric_name: str, include_cross_eval: bool=True, not_include: Sequence[str]="") -> list[pd.DataFrame]:
    """ Extracts the pandas Dataframes from the list of dictionaries. 
    - dfs: this is the list of dictionaries you are going to search in. If you only want to search in one dictionary
    make sure to put it in a list!
    - metric_name: specify to filter for a specific metric, e.g. 'accuracy' or 'loss' etc.
    - include_cross_eval: True or False. Do you want to include the cross validation data in the result.
    - not_include: to specify when you do not want to include cross validation data. Put the loss function (count, ttfs, etc.)
    in the order you do not want to include. """
    result = []
    num = 0
    for d in dfs:
        for key in d:
            if include_cross_eval and metric_name in key:
                result.append(d[key])
                if print_load:
                    print(f"{key} data added to list")
            elif metric_name in key and not_include[num] not in key:
                result.append(d[key])
                if print_load:
                    print(f"{ key} data added to list")
        num += 1
    return result

data_dict_count_e_ttfs = make_data_dict(directory_count_ttfs)
data_dict_ttfs_e_count = make_data_dict(directory_ttfs_count)
data_dict_decay = make_data_dict(directory_decay)

metric_names_count = ['accuracy_train_count', 'loss_train_count', 'silent_neurons', 'accuracy_count_test', 'accuracy_ttfs_test',
                      'loss_count_test', 'loss_ttfs_test', 'weight_norm_Hidden', 'weight_norm_Output']
metric_names_ttfs = ['accuracy_train_ttfs', 'loss_train_ttfs', 'silent_neurons', 'accuracy_count_test', 'accuracy_ttfs_test',
                     'loss_count_test', 'loss_ttfs_test', 'weight_norm_Hidden', 'weight_norm_Output']
metric_names_decay = ['accuracy_test', 'accuracy_train', 'loss_test', 'loss_train', 'weight_norm_Hidden', 'weight_norm_Output']


# Load and store data train count eval TTFS
metric_data_dict_count_e_ttfs = {metric_name: extract_metric_data(data_dict_count_e_ttfs, metric_name) 
                                 for metric_name in metric_names_count}
dataframes_count_e_ttfs = {metric_name: create_dataframes(metric_data, metric_name) 
                           for metric_name, metric_data in metric_data_dict_count_e_ttfs.items()}
stats_dataframes_count_e_ttfs = {metric_name: extract_stats(df) for metric_name, df in dataframes_count_e_ttfs.items()}

loc, best_test_acc_count, sd_test_acc_count = get_best_acc(stats_dataframes_count_e_ttfs['accuracy_count_test'])
print(f"The best accuracy of count trained on count loss is: {best_test_acc_count:.2f}% With \u00B1{sd_test_acc_count:.3f}% at epoch {loc}.")

# Load and store data train TTFS eval count
metric_data_dict_ttfs_e_count = {metric_name: extract_metric_data(data_dict_ttfs_e_count, metric_name) 
                                 for metric_name in metric_names_ttfs}
dataframes_ttfs_e_count = {metric_name: create_dataframes(metric_data, metric_name) 
                           for metric_name, metric_data in metric_data_dict_ttfs_e_count.items()}
stats_dataframes_ttfs_e_count = {metric_name: extract_stats(df) for metric_name, df in dataframes_ttfs_e_count.items()}

loc, best_test_acc_ttfs, sd_test_acc_ttfs = get_best_acc(stats_dataframes_ttfs_e_count['accuracy_ttfs_test'])
print(f"The best accuracy of TTFS trained on TTFS loss is: {best_test_acc_ttfs:.2f}% With \u00B1{sd_test_acc_ttfs:.3f}% at epoch {loc}.")

# Load and store data decay loss
metric_data_dict_decay = {metric_name: extract_metric_data(data_dict_decay, metric_name) 
                                 for metric_name in metric_names_decay}
dataframes_decay = {metric_name: create_dataframes(metric_data, metric_name) 
                           for metric_name, metric_data in metric_data_dict_decay.items()}
stats_dataframes_decay = {metric_name: extract_stats(df) for metric_name, df in dataframes_decay.items()}

loc, best_test_acc_decay, sd_test_acc_decay = get_best_acc(stats_dataframes_decay[''])
print(f"The best accuracy of TTFS trained on TTFS loss is: {best_test_acc_decay:.2f}% With \u00B1{sd_test_acc_decay:.3f}% at epoch {loc}.")


#######################################################################################################################################
####################                                           PLOTTING                                            ####################
#######################################################################################################################################

# save path
save_path = "/Users/hanna/Downloads/plots"

##########################################################################
#########                     ACCURACY PLOTS                     #########
##########################################################################
execute_acc = False  # Want to show accurary plot
if execute_acc:
    accuracy_dfs_all = get_dfs_to_list([stats_dataframes_count_e_ttfs, stats_dataframes_ttfs_e_count], "accuracy")
    accuracy_df_decay = get_dfs_to_list([stats_dataframes_decay], 'accuracy')
    accuracy_dfs_zoom = get_dfs_to_list([stats_dataframes_count_e_ttfs, stats_dataframes_ttfs_e_count, stats_dataframes_decay], 
                                "accuracy", include_cross_eval=False, not_include=('ttfs', 'count', 'ipsum'))
    accuracy_labels_all = ['train count', 'train count test count', "train count test TTFS", 
                           'train TTFS', 'train TTFS test count', "train TTFS test TTFS"]
    accuracy_labels_decay = ['test decay', 'train decay']
    accuracy_labels_zoom = ['train count', 'test count', 'train ttfs', 'test ttfs', 'test decay', 'train decay']

    create_line_plot_multiple(accuracy_dfs_all, 'Epoch', 'mean', title="All Accuracy", ylabel="accuracy (%)", 
                              labels=accuracy_labels_all, path=save_path, legend_outside_grid=True)
    create_line_plot_multiple(accuracy_df_decay, 'Epoch', 'mean', title="Accuracy of decay loss function", ylabel="accuracy (%)", 
                              labels=accuracy_labels_decay, path=save_path)
    create_line_plot_multiple(accuracy_dfs_zoom, 'Epoch', 'mean', title="Accuracy good performance", ylabel="accuracy (%)", 
                              labels=accuracy_labels_zoom, set_limit=True, blimit=95, rlimit=30, path=save_path)
    
######################################################################
#########                     lOSS PLOTS                     #########
######################################################################
execute_loss = True  # Want to show loss plot
if execute_loss:
    loss_dfs_all = get_dfs_to_list([stats_dataframes_count_e_ttfs, stats_dataframes_ttfs_e_count, stats_dataframes_decay], 
                                    "loss", include_cross_eval=True)
    loss_dfs_zoom_count = get_dfs_to_list([stats_dataframes_count_e_ttfs], "loss", include_cross_eval=False, not_include=['ttfs'])
    loss_dfs_zoom_ttfs = get_dfs_to_list([stats_dataframes_ttfs_e_count], "loss", include_cross_eval=False, not_include=['count'])
    loss_dfs_decay = get_dfs_to_list([stats_dataframes_decay], "loss")
    loss_labels_all = ['train count', 'train count test count', "train count test TTFS", 
                        'train TTFS', 'train TTFS test count', "train TTFS test TTFS"]
    loss_labels_zoom_count = ['train', 'test']
    loss_labels_zoom_decay = ['test', 'train']

    create_line_plot_multiple(loss_dfs_all, 'Epoch', 'mean', title="Losses of count and TTFS", ylabel="loss", tlimit=1000,
                              labels=loss_labels_all, set_limit=True, loc='upper right', path=save_path)
    create_line_plot_multiple(loss_dfs_zoom_count, 'Epoch', 'mean', title="Losses of count", ylabel="loss", 
                              labels=loss_labels_zoom_count, loc='upper right', path=save_path, set_limit=True, tlimit=30)
    create_line_plot_multiple(loss_dfs_zoom_ttfs, 'Epoch', 'mean', title="Losses of TTFS", ylabel="loss", 
                              labels=loss_labels_zoom_count, loc='upper right', path=save_path, set_limit=True, tlimit=1)
    create_line_plot_multiple(loss_dfs_decay, 'Epoch', 'mean', title="Loss development of own implemented loss function", 
                              ylabel="loss", labels=loss_labels_zoom_count, loc='upper right', path=save_path)

