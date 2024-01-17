import matplotlib.pyplot as plt 
import numpy as np

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1]})

# plot the accuries together
ax1.plot(data[0]["epochs"], data[0]["values"], color='r', label="test")
ax1.plot(data[1]["epochs"], data[1]["values"], color='b', label="train")
ax1.set_xlabel('Epochs')
ax1.set_ylabel("accuracy (%)")
ax1.set_title("Accuracy")
ax1.legend()

# plot the loss together
ax2.plot(data[2]["epochs"], data[2]["values"], color='r', label="test")
ax2.plot(data[3]["epochs"], data[3]["values"], color='b', label="train")
ax2.set_xlabel('Epochs')
ax2.set_ylabel("loss")
ax2.set_title("Loss")
ax2.legend()

# plot the weight norm together
ax3.plot(data[4]["epochs"], data[4]["values"], color='r', label="Hidden layer")
ax3.plot(data[5]["epochs"], data[5]["values"], color='b', label="Output layer")
ax3.set_xlabel('Epochs')
ax3.set_ylabel("weight norm")
ax3.set_title("Weight norm")
ax3.legend()

# Show the plot
plt.show()

