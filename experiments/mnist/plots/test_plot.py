import matplotlib.pyplot as plt 
import numpy as np

file_path = "/Users/hanna/Downloads/SNN-CAPSTONE-1/results/train_ttfs_eval_count/output_metrics/accuracy_train_ttfs.npz"
data = np.load(file_path)

epochs = data["epochs"]
values = data["values"]

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(values, label="loss, TTFS")

# Add labels, title, legend, etc.
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Training accuracy, TTFS')
plt.legend()

# Show the plot or save it to a file
plt.show()
