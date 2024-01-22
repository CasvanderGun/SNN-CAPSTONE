import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import pandas as pd
import os

# Specify the path to the directory
directory_path = r'C:\Users\hanna\Downloads\SNN-CAPSTONE-1\results\train_multiple_runsv2\Run_'
num_runs = 5

def loading_files(dic_path):
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

list_data = {}
for run in range(num_runs):
    run_path = directory_path + str(run + 1) + "\output_metrics"
    print(f' """     LOADING RUN {str(run + 1)}      """')
    list_data["Run_" + str(run + 1)] = loading_files(run_path)

# Initialize empty lists for accuracy, loss, weight norm, and silent_neurons data
train_accuracy_data = []
train_loss_data = []
test_accuracy_count_data = []
test_accuracy_ttfs_data = []
test_loss_count_data = []
test_loss_ttfs_data = []
weight_norm_hid_data = []
weight_norm_out_data = []
silent_neurons_data = []      

# Iterate through each run's loaded data
for run_key, run_data in list_data.items():

    # Create dictionaries to store data for each metric
    train_accuracy_dict = {'Run': run_key}
    train_loss_dict = {'Run': run_key}
    silent_neurons_dict = {'Run': run_key}
    test_accuracy_count_dict = {'Run': run_key}
    test_accuracy_ttfs_dict = {'Run': run_key}
    test_loss_count_dict = {'Run': run_key}
    test_loss_ttfs_dict = {'Run': run_key}
    weight_norm_hid_dict = {'Run': run_key}
    weight_norm_out_dict = {'Run': run_key}

    # Populate dictionaries with data for each metric and add to list
    for file_data in run_data:
        action = {f'Epoch_{epoch}': value for epoch, value in zip(file_data['epochs'], file_data['values'])}
        if "accuracy_train" in file_data['filename']:
            train_accuracy_dict.update(action)
            train_accuracy_data.append(train_accuracy_dict)
        elif "loss_train" in file_data['filename']:
            train_loss_dict.update(action)
            train_loss_data.append(train_loss_dict)
        elif "silent_neurons" in file_data['filename']:
            silent_neurons_dict.update(action)
            silent_neurons_data.append(silent_neurons_dict)
        elif "accuracy_count_test" in file_data['filename']:
            test_accuracy_count_dict.update(action)
            test_accuracy_count_data.append(test_accuracy_count_dict)
        elif "accuracy_ttfs_test" in file_data['filename']:
            test_accuracy_ttfs_dict.update(action)
            test_accuracy_ttfs_data.append(test_accuracy_ttfs_dict)
        elif "loss_count_test" in file_data['filename']:
            test_loss_count_dict.update(action)
            test_loss_count_data.append(test_loss_count_dict)
        elif "loss_ttfs_test" in file_data['filename']:
            test_loss_ttfs_dict.update(action)
            test_loss_ttfs_data.append(test_loss_ttfs_dict)
        elif "weight_norm_Hidden" in file_data['filename']:
            weight_norm_hid_dict.update(action)
            weight_norm_hid_data.append(weight_norm_hid_dict)
        elif "weight_norm_Output" in file_data['filename']:
            weight_norm_out_dict.update(action)
            weight_norm_out_data.append(weight_norm_out_dict)

# Create dataframes for accuracy, loss, weight norm, and silent_neurons
train_accuracy_df = pd.DataFrame(train_accuracy_data)
train_loss_df = pd.DataFrame(train_loss_data)
test_accuracy_count_df = pd.DataFrame(test_accuracy_count_data)
test_accuracy_ttfs_df = pd.DataFrame(test_accuracy_ttfs_data)
test_loss_count_df = pd.DataFrame(test_loss_count_data)
test_loss_ttfs_df = pd.DataFrame(test_loss_ttfs_data)
weight_norm_hid_df = pd.DataFrame(weight_norm_hid_data)
weight_norm_out_df = pd.DataFrame(weight_norm_out_data)
silent_neurons_df = pd.DataFrame(silent_neurons_data)

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

train_accuracy_stats_df = extract_stats(train_accuracy_df)
train_loss_stats_df = extract_stats(train_loss_df)
test_accuracy_count_stats_df = extract_stats(test_accuracy_count_df)
test_accuracy_ttfs_stats_df = extract_stats(test_accuracy_ttfs_df)
test_loss_count_stats_df = extract_stats(test_loss_count_df)
test_loss_ttfs_stats_df = extract_stats(test_loss_ttfs_df)
weight_norm_hid_stats_df = extract_stats(weight_norm_hid_df)
weight_norm_out_stats_df = extract_stats(weight_norm_out_df)
silent_neurons_stats_df = extract_stats(silent_neurons_df)

def create_line_plot(df, x_name, y_name, r, title, label):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=x_name, y=y_name, data=df, ci="sd", label=label)
    plt.fill_between(df[x_name], df[y_name] - df[r], df[y_name] + df[r], 
                     alpha=0.4, label='Confidence Interval')
    plt.title(title + 'Confidence Intervals over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Value')
    plt.legend()
    plt.show()

def create_line_plot_multiple(df_list: list[pd.DataFrame], x_name: str, y_name: str, r='sd', title: str | None = None, 
                              ylabel: str | None = None, labels: list[str] = None, set_limit:  bool = False, 
                              blimit: float = 0.0, tlimit: float = 100, style: str = "whitegrid", loc="lower right",
                              eval_df: pd.DataFrame | None = None, path: str = "") -> None:
    """ Function to plot multiple lines in a graph. The input must be a list containing the pd.DataFrames you want to make a graph of.
    Use the x_name and y_name to specify the columns in the dataframe to get the data you want to use in the plot."""
    sns.set(style=style)
    plt.figure(figsize=(12, 6))

    for df, label in zip(df_list, labels):
        sns.lineplot(x=x_name, y=y_name, data=df, errorbar=r, label=label)
        plt.fill_between(df[x_name], df[y_name] - df[r], df[y_name] + df[r], 
                         alpha=0.4, label=f'{label} confidence interval')
    if eval_df is not None:
        max_epoch = eval_df.loc[eval_df['mean'].idxmax()]
        max_accuracy = max_epoch['mean']
        std = max_epoch['sd']
        full_title = title + ' with confidence intervals over runs\n' + f"Best test accuracy: {max_accuracy:.2f}% with \u00B1{std:.3f}% at epoch {max_epoch['Epoch']:.0f}"
    else:
        full_title = title + ' with confidence intervals over runs'
    plt.title(full_title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel=ylabel)
    if set_limit:
        plt.ylim(bottom=blimit, top=tlimit)
    if path != "":
        plt.savefig(path + "/" + title)
    plt.legend(loc=loc)
    plt.show()

# create_line_plot(train_accuracy_stats_df, 'Epoch', 'mean', 'sd', "train accuracy count loss with", "train")
print(f"The best accuracy of count trained on count loss is: {max(test_accuracy_count_stats_df['mean']):.2f}\n"
      f"With \u00B1{test_accuracy_count_stats_df.loc[test_accuracy_count_stats_df['mean'].idxmax()]['sd']:.3f}%")

# save path
save_path = "/Users/hanna/Downloads/train_count_eval_ttfs_plots"
# Accuracies plotted together
accuracy_dfs = [train_accuracy_stats_df, test_accuracy_count_stats_df]
accuracy_labels = ['train', 'test count', 'test ttfs']
execute_acc = False  # Want to show accurary plot
if execute_acc:
    create_line_plot_multiple(accuracy_dfs, 'Epoch', 'mean', title="Accuracy trained on count loss zoomed", ylabel="accuracy (%)", 
                              labels=accuracy_labels, set_limit=True, blimit=95, path=save_path)

# Loss plotted together
loss_dfs = [train_loss_stats_df, test_loss_count_stats_df]
loss_labels = ['train', 'test count', 'test ttfs']
execute_loss = True  # Want to show loss plot
if execute_loss:
    create_line_plot_multiple(loss_dfs, 'Epoch', 'mean', title="Loss trained on count loss zoomed", ylabel="loss", 
                              labels=loss_labels, set_limit=True, tlimit=20, path=save_path, loc='upper right')

# Weight norm plotted together
weight_dfs = [weight_norm_hid_stats_df, weight_norm_out_stats_df]
weight_labels = ["hidden layer", 'output layer']
execute_weight = False  # Want to show weight plot
if execute_weight:
    create_line_plot_multiple(weight_dfs, 'Epoch', 'mean', title="Weight norm trained on count", ylabel="weight norm",
                              labels=weight_labels, path=save_path, loc='upper left')

