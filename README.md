# SNN CAPSTONE PROJECT

Welcome to the repository of our capstone project regarding Spiking Neural Networks (SNNs). During this project we built upon the [research](https://arxiv.org/abs/2212.09500) and [repository](https://github.com/Florian-BACHO/bats) of [Florian Bacho](https://github.com/Florian-BACHO), from whom this repository is forked. 

First, in section 1, the original explanation of the algorithm can be found. This is the explanation as can be found in the repository by Florian Bacho. Second, in section 2, the required dependencies and libraries are described. Lastly, in section 3, all experiments along with a short explanation of the experiment can be found.


## 1. Algorithm

The explanation of the algorithm by Florian Bacho in the README of his repository:

> "Error Backpropagation Through Spikes (BATS) [1] is a GPU-compatible algorithm that extends Fast & Deep [2], 
a method to performs exact gradient descent in Deep Spiking Neural Networks (SNNs). 
In contrast with Fast & Deep, BATS allows error backpropagation with multiple spikes per neuron, leading to increased 
performances. The proposed algorithm backpropagates the errors through post-synaptic spikes with linear time complexity 
O(N) making the error backpropagation process fast for multi-spike SNNs.<br>
This repository contains the full Cuda implementations of our efficient event-based SNN simulator and the BATS algorithm.
All the experiments on the convergence of single and multi-spike models, on the MNIST dataset, its extended version 
EMNIST and Fashion MNIST are also provided to reproduce our results."

## 2. Dependencies and Libraries

The recommended Python version is 3.8 or higher.

Libraries:
- Cuda (Version 10.1 is the version used to develop BATS. Other versions should also work.)
  
Python packages:
- CuPy (Needs to correspond to the installed version of Cuda)
- matplotlib (To generate plots with the monitors)
- requests (To download the dataset)


## 3. Experiments

There are X experiments available in our repository. These are the experiments we ran to obtain the results used for our project. 

### Downloading the dataset

For our experiments we only considered the MNIST dataset. It is possible to download MNIST by navigating to the <em>datasets</em> directory and running the <em>get_mnist.py</em> script. This will result in the download of the MNIST dataset into the <em>datasets</em> directory.


```console
$ cd datasets
$ ls
download_file.py  get_mnist.py
$ python3 get_mnist.py
Downloading MNIST...
[██████████████████████████████████████████████████]
Done.
$ ls
download_file.py get_mnist.py  mnist.npz
```

### Saving the results

During the experiments, plots and data are saved in the <em>output_metrics</em> directory, while weights of the best model are saved in the <em>best_model</em> directory. For each experiment there is a separate directory in which the <em>output_metrics</em> and <em>best_model</em> directories can be found. The 

Prior to running an experiment the results directory for the respective experiment must exist. This can be done by navigating to the <em>experiments/mnist</em> directory and creating a new directory with the respective experiment name. Note that the results directory is hardcoded into the experiments. 

```console
$ cd results
$ mkdir <experiment_name_placeholder>
...
```

For example, to run the <em>train_count_eval_count.py</em> experiment script, a directory with the name <em>train_count_eval_count</em> must be created in the <em>experiments/mnist</em>.

```console
$ cd results
$ ls

$ mkdir train_count_eval_count
$ ls
train_count_eval_count
...
```


### Running the experiments

#### Experiment 1
Tree models are available to train with the MNIST dataset: 
- a single-spike model (<em>train_ttfs.py</em>)
```console
$ cd experiments/mnist
$ python3 train_ttfs.py
...
```
- a multi-spike model (<em>train_spike_count.py</em>)
```console
$ cd experiments/mnist
$ python3 train_spike_count.py
...
```





## References

[1] Bacho, F., & Chu, D.. (2022). Exact Error Backpropagation Through Spikes for Precise Training of Spiking Neural Networks. https://arxiv.org/abs/2212.09500 <br>
[2] J. Göltz, L. Kriener, A. Baumbach, S. Billaudelle, O. Breitwieser, B. Cramer, D. Dold, A. F. Kungl, W. Senn, J. Schemmel, K. Meier, & M. A. Petrovici (2021). Fast and energy-efficient neuromorphic deep learning with first-spike times. <em>Nature Machine Intelligence, 3(9), 823–835.</em> <br>

