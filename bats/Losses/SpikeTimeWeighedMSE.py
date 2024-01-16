from typing import Tuple

from ..AbstractLoss import AbstractLoss
import cupy as cp

'''
This class will weighs all elements of the spike timings with a decaying exponential to weigh 
the first spikes more than the last spikes in the spike train

This loss function will take the Mean Squared Error between the set targets and the actual spikes
The targets will be modified by the decay rate 
'''

class SpikeTimeWeighedMSE(AbstractLoss):
  # Finished, compied from: SpikeCountLoss.py, SpikeCountLossClass.py, SpikeTimeWeightedSoftmaxCrossEntropy.py
    def __init__(self, target_true: float, target_false: float, decay_rate: float):

        # This kernel will calculate the loss 
        self.__loss_kernel = cp.ReductionKernel("float32 out_count, float32 out_target",
                                        "float32 loss",
                                        "(out_target - out_count) * (out_target - out_count)",
                                        "a + b",
                                        "loss = a / 2",
                                        "0",
                                        "loss_kernel")

        # this kernel will elementwise calculate the weighted value of the spike times
        # the values that are inf. (no spikes in this alloted space) will become zero
        self.__weight_kernel = cp.ElementwiseKernel("float32 spike_time, float32 decay_rate",
                                              "float32 weight",
                                              "weight = exp(-decay_rate * spike_time)",
                                              "weight_kernel")

        self.__decay_rate: cp.float32 = cp.float32(decay_rate)

        self.__target_true: cp.float32 = cp.float32(target_true)
        self.__target_false: cp.float32 = cp.float32(target_false)

    # finished, copied from SpikeCountLoss.py
    def predict(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray) -> cp.ndarray:
        weights = self.__weight_kernel(spikes_per_neuron, self.__decay_rate)
        summed_weights = cp.sum(weights, axis=2) 
        prediction = cp.argmax(summed_weights, axis=1)
        return prediction

    # Finished, copied from SpikeCountClassLoss.py
    def __compute_targets(self, n_spike_per_neuron: cp.ndarray, labels: cp.ndarray):
        targets = cp.full(n_spike_per_neuron.shape, self.__target_false) # TODO modify this with the decay rate
        targets[cp.arange(labels.size), labels] = self.__target_true # TODO modify this with the decay rate
        return targets
  	
    # Finished, copied from SpikeCountLoss.py, SpikeTimeWeightedSoftmaxCrossEntropy.py
    def compute_loss(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                     labels: cp.ndarray) -> cp.ndarray:
        targets = self.__compute_targets(n_spike_per_neuron, labels)

        weighed_spike_times = self.__weight_kernel(spikes_per_neuron, self.__decay_rate) # results in 2D array [N_samples, N_outputs, N_spikes]
        summed_weighed_spike_times = cp.sum(weighed_spike_times, axis=2) # results in 2D array [N_samples, N_outputs]

        loss = self.__loss_kernel(summed_weighed_spike_times, targets, axis=1)

        return loss

    # Finished, merged first of compute loss with compute errors of SpikeCount.py 
    def compute_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                       labels: cp.ndarray) -> cp.ndarray:
        targets = self.__compute_targets(n_spike_per_neuron, labels)

        weighed_spike_times = self.__weight_kernel(spikes_per_neuron, self.__decay_rate) # results in 2D array [N_samples, N_outputs, N_spikes]
        summed_weighed_spike_times = cp.sum(weighed_spike_times, axis=2) # results in 2D array [N_samples, N_outputs]

        max_n_spike = spikes_per_neuron.shape[2]

        neurons_errors = targets - summed_weighed_spike_times.astype(cp.float32)
        return cp.repeat(neurons_errors[:, :, cp.newaxis], repeats=max_n_spike, axis=2)

    # Finished, just reused compute_loss() and compute_errors()
    def compute_loss_and_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                labels: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        targets = self.__compute_targets(n_spike_per_neuron, labels)

        weighed_spike_times = self.__weight_kernel(spikes_per_neuron, self.__decay_rate) # results in 2D array [N_samples, N_outputs, N_spikes]
        summed_weighed_spike_times = cp.sum(weighed_spike_times, axis=2) # results in 2D array [N_samples, N_outputs]

        loss = self.__loss_kernel(summed_weighed_spike_times, targets, axis=1)
        
        max_n_spike = spikes_per_neuron.shape[2]
        neurons_errors = targets - summed_weighed_spike_times.astype(cp.float32)

        errors = cp.repeat(neurons_errors[:, :, cp.newaxis], repeats=max_n_spike, axis=2)
        return loss, errors
