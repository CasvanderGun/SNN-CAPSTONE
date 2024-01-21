import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_spike_count_map(avg_spike_counts: dict, amount_of_neurons_shown: int, vmax: int, title: str, path: str, experiment_name: str) -> None:
        """Creates a visual representation of the average spike counts per neuron in the hidden layer for each label and stores a png in the results folder
        args:
        - avg_spike_counts: input data, stored as a dict
        - amount_of_neurons_shown: specifeis how many hidden neurons you want to be plotted
        - vmax: specifies the maximum spike count of the colorbar
        - title: set to the title of the plot
        - expirement_name: set to the name of the experiment, example: train_count_eval_count"""
        data_to_plot = np.array([avg_spike_counts[digit] for digit in sorted(avg_spike_counts.keys())])
        plt.figure(figsize=(8, 8))
        custom_map = LinearSegmentedColormap.from_list('custom', ['white', 'lavender', 'yellow', 'red', 'black'])
        plt.imshow(data_to_plot, cmap=custom_map, interpolation='nearest', origin='lower', aspect=amount_of_neurons_shown/10, vmin=0, vmax=vmax)
        plt.xlim(0,amount_of_neurons_shown)
        plt.colorbar()
        plt.xlabel('Hiddden Neuron Index')
        plt.ylabel('Label')
        plt.title(title)
        # save_path = f'/content/SNN-CAPSTONE/results/{experiment_name}/neuron_plots/{title}.png'
        save_path = path + experiment_name + title
        plt.savefig(save_path)
        print(f"Neuron Plot saved to {save_path}")

      
    
  


