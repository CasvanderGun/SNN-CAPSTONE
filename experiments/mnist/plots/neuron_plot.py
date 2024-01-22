import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_spike_count_map(avg_spike_counts: dict, amount_of_neurons_shown: int, vmax: int, title: str, path: str, root_path: str) -> None:
        """Creates a visual representation of the average spike counts per neuron in the hidden layer for each label and stores a png in the results folder
        args:
        - avg_spike_counts: input data, stored as a dict
        - amount_of_neurons_shown: specifeis how many hidden neurons you want to be plotted
        - vmax: specifies the maximum spike count of the colorbar
        - title: set to the title of the plot
        - expirement_name: set to the name of the experiment, example: train_count_eval_count
        - path: the path consists of the path, which is the directory of the experiment you want to save the results. The 
        inclusion of '/neuron_plots/' ensures that the .png are stored in a separated folder. !Make sure you make the folder in the
        directory! The title ensures that every .png has a different name and does not overwrite the older ones. Makes sure that the
        titles are different for every epoch!
        - root_path: the root path, different for kaggle and google colab"""
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
        save_path = root_path + path + "/neuron_plots/" + title
        plt.savefig(save_path)
        print(f"Neuron Plot saved to {save_path}")

      
    
  


