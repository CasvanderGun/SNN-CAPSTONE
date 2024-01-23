from pathlib import Path
import cupy as cp
import numpy as np
import math 
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, "../../../")  # Add repository root to python path

from experiments.mnist.Dataset import Dataset
from bats.Layers import InputLayer, LIFLayer
from bats.Network import Network

# These were the settings of a train_decy_rate 
def load_model(model_path):
    N_INPUTS = 28 * 28

    # Hidden layer
    N_NEURONS_1 = 800
    TAU_S_1 = 0.130
    THRESHOLD_HAT_1 = 0.2
    DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
    SPIKE_BUFFER_SIZE_1 = 30

    # Output_layer
    N_OUTPUTS = 10
    TAU_S_OUTPUT = 0.130
    THRESHOLD_HAT_OUTPUT = 1.3
    DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
    SPIKE_BUFFER_SIZE_OUTPUT = 30

    max_int = np.iinfo(np.int32).max
    np_seed = np.random.randint(low=0, high=max_int)
    cp_seed = np.random.randint(low=0, high=max_int)
    np.random.seed(np_seed)
    cp.random.seed(cp_seed)
    print(f"Numpy seed: {np_seed}, Cupy seed: {cp_seed}")

    # load network weights
    hidden_layer_1_params = np.load(model_path + '/Hidden layer 1_weights.npy')
    output_layer_params = np.load(model_path + '/Output layer_weights.npy')

    # create network
    print("Creating network...")
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)

    hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                            theta=THRESHOLD_HAT_1,
                            delta_theta=DELTA_THRESHOLD_1,
                            max_n_spike=SPIKE_BUFFER_SIZE_1,
                            name="Hidden layer 1")
    hidden_layer.weights = hidden_layer_1_params
    network.add_layer(hidden_layer)

    output_layer = LIFLayer(previous_layer=hidden_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    output_layer.weights = output_layer_params
    network.add_layer(output_layer)

    return network

def get_image(index):
    # Dataset
    DATASET_PATH = Path("../../../datasets/mnist.npz")

    # Dataset
    dataset = Dataset(path=DATASET_PATH)

    # get the first image
    im = dataset.get_test_image_at_index(index)

    return im.astype(np.float64)

def show_image(index, path, title = ''):
    im = get_image(index)

    # Normalize the pixel values to the range [0, 1]
    im = im / 255.0

    # Plot the normalized image as a grayscale image
    plt.imshow(im, cmap='gray')
    plt.title(title)
    plt.savefig(path)

def get_spike_train(image, model, sim_time):
    # Dataset
    DATASET_PATH = Path("../../../datasets/mnist.npz")

    # Dataset
    dataset = Dataset(path=DATASET_PATH)

    # input spike trains
    # spikes, n_spikes, labels = dataset.get_train_batch(batch_index=0, batch_size=1)
    spikes, n_spikes = dataset.to_spike_train(image)
    
    # Inference
    model.reset()
    model.forward(spikes, n_spikes, max_simulation=sim_time, training=True)
    out_spikes, n_out_spikes = model.output_spike_trains

    return out_spikes, n_out_spikes
    
def plot_spike_train(image, model, sim_time, title, save_path):
    out_spikes, n_out_spikes = get_spike_train(image, model, sim_time)
    for label, spike_train in enumerate(out_spikes[0]):
      for spike in spike_train:
        spike = spike.get()
        if not math.isinf(spike):
            plt.scatter(spike, label, c='k')
    plt.title(title)
    plt.grid(alpha=0.2, color='k', linewidth=1)
    plt.xticks(np.linspace(0, sim_time, 5))
    plt.yticks(np.arange(10))
    plt.savefig(str(save_path) + f'/{title}')

def plot_all_spike_trains(model, sim_time, title, save_folder):
    indices = np.arange(20)
    
    for i in indices:
        image = get_image(i)
        plot_spike_train(image, model, sim_time, title + f'_{i}', save_folder)