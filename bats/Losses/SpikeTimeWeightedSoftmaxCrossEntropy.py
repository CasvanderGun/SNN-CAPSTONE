from typing import Tuple

from ..AbstractLoss import AbstractLoss
import cupy as cp


class SpikeTimeWeightedSoftmaxCrossEntropy(AbstractLoss):
    def __init__(self, decay_rate: float):
        
        self.__decay_rate: cp.float32 = cp.float32(decay_rate)

        self.__weight_kernel = cp.ElementwiseKernel("float32 spike_time, float32 decay_rate",
                                                     "float32 weight",
                                                     "weight = exp(-decay_rate * spike_time)",
                                                     "weight_kernel")

        self.__cross_entropy_kernel = cp.ElementwiseKernel("float32 labels_summed_weights, float32 sums",
                                                           "float32 out",
                                                           "out = - __logf(labels_summed_weights / sums)",
                                                           "cross_entropy_kernel")

    def predict(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray) -> cp.ndarray:
        weights = self.__weight_kernel(spikes_per_neuron, self.__decay_rate)
        summed_weights = cp.sum(weights, axis=2) 
        prediction = cp.argmax(summed_weights, axis=1)
        return prediction

    def compute_loss(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                     labels: cp.ndarray) -> cp.ndarray:
        weights = self.__weight_kernel(spikes_per_neuron, self.__decay_rate) # results in 2D array [N_samples, N_outputs, N_spikes]
        summed_weights = cp.sum(weights, axis=2) # results in 2D array [N_samples, N_outputs]
        sums = cp.sum(summed_weights, axis=1) # results in a 1D array of [N_outputs]
        labels_summed_weights = summed_weights[cp.arange(labels.size), labels]
        loss = self.__cross_entropy_kernel(labels_summed_weights, sums)
        return loss

    def compute_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                       labels: cp.ndarray) -> cp.ndarray:
        
        weights = self.__weight_kernel(spikes_per_neuron, self.__decay_rate) # results in 2D array [N_samples, N_outputs, N_spikes]
        summed_weights = cp.sum(weights, axis=2) # results in 2D array [N_samples, N_outputs]
        sums = cp.sum(summed_weights, axis=1)
        
       
        # Compute negative softmax (error gradient)
        neg_softmax = -summed_weights / sums[:, cp.newaxis]
        neg_softmax[cp.arange(labels.size), labels] += 1
        neg_softmax /= self.__decay_rate

        # Expand errors to match the original spike dimensions
        errors = cp.zeros(spikes_per_neuron.shape, dtype=cp.float32)
        errors[..., 0] = cp.nan_to_num(neg_softmax)

        return errors

    def compute_loss_and_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                labels: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        weights = self.__weight_kernel(spikes_per_neuron, self.__decay_rate) # results in 2D array [N_samples, N_outputs, N_spikes]
        summed_weights = cp.sum(weights, axis=2) # results in 2D array [N_samples, N_outputs]
        sums = cp.sum(summed_weights, axis=1)
        
        labels_summed_weights = summed_weights[cp.arange(labels.size), labels]
        loss = self.__cross_entropy_kernel(labels_summed_weights, sums)
       
        # Compute negative softmax (error gradient)
        neg_softmax = -summed_weights / sums[:, cp.newaxis]
        neg_softmax[cp.arange(labels.size), labels] += 1
        neg_softmax /= self.__decay_rate

        # Expand errors to match the original spike dimensions
        errors = cp.zeros(spikes_per_neuron.shape, dtype=cp.float32)
        errors[..., 0] = cp.nan_to_num(neg_softmax)

        return loss, errors
