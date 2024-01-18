import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import pandas as pd
import os

# Specify the path to the directory
directory_path = r'\Users\hanna\Downloads\SNN-CAPSTONE-1\results\own_loss\output_metrics_50-epoch'

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

data = loading_files(directory_path)

# Find the common set of epochs between test and training data
common_epochs = set(data[0]['epochs']).intersection(set(data[1]['epochs']))

# Create dataframes for accuracy, loss, and weight norm
accuracy_df = pd.DataFrame({
    'Epochs': list(common_epochs),
    'Test Accuracy': [data[0]['values'][data[0]['epochs'].tolist().index(epoch)] for epoch in common_epochs],
    'Train Accuracy': [data[1]['values'][data[1]['epochs'].tolist().index(epoch)] for epoch in common_epochs]
})

loss_df = pd.DataFrame({
    'Epochs': list(common_epochs),
    'Test Loss': [data[2]['values'][data[2]['epochs'].tolist().index(epoch)] for epoch in common_epochs],
    'Train Loss': [data[3]['values'][data[3]['epochs'].tolist().index(epoch)] for epoch in common_epochs]
})

weight_norm_df = pd.DataFrame({
    'Epochs': list(common_epochs),
    'Hidden Layer Weight Norm': [data[4]['values'][data[4]['epochs'].tolist().index(epoch)] for epoch in common_epochs],
    'Output Layer Weight Norm': [data[5]['values'][data[5]['epochs'].tolist().index(epoch)] for epoch in common_epochs]
})

# Set up Seaborn style
sns.set(style='whitegrid')

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot Accuracy
sns.lineplot(x='Epochs', y='Test Accuracy', data=accuracy_df, color='r', label='Test Accuracy', ax=axes[0])
sns.lineplot(x='Epochs', y='Train Accuracy', data=accuracy_df, color='b', label='Train Accuracy', ax=axes[0])
axes[0].set_title('Accuracy')
axes[0].set_ylabel('accuracy (%)')

# Plot Loss
sns.lineplot(x='Epochs', y='Test Loss', data=loss_df, color='r', label='Test Loss', ax=axes[1])
sns.lineplot(x='Epochs', y='Train Loss', data=loss_df, color='b', label='Train Loss', ax=axes[1])
axes[1].set_title('Loss')
axes[1].set_ylabel('loss')

# Plot Weight Norm
sns.lineplot(x='Epochs', y='Hidden Layer Weight Norm', data=weight_norm_df, color='r', label='Hidden Layer', ax=axes[2])
sns.lineplot(x='Epochs', y='Output Layer Weight Norm', data=weight_norm_df, color='b', label='Output Layer', ax=axes[2])
axes[2].set_title('Weight Norm')
axes[2].set_ylabel('weight norm')

