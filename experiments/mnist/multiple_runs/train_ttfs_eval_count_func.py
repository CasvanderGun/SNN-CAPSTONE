from pathlib import Path
import cupy as cp
import numpy as np

import os
import sys

sys.path.insert(0, "../../../")  # Add repository root to python path

from experiments.mnist.load_model import plot_spike_train, get_image
from experiments.mnist.plots.neuron_plot import create_spike_count_map
from experiments.mnist.Dataset import Dataset
from bats.Monitors import *
from bats.Layers import InputLayer, LIFLayer
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *

def train_ttfs_eval_count(epochs, export_path):
    # Dataset
    DATASET_PATH = Path("../../../datasets/mnist.npz")

    N_INPUTS = 28 * 28
    SIMULATION_TIME = 1.0

    # Hidden layer
    N_NEURONS_1 = 800
    TAU_S_1 = 0.130
    THRESHOLD_HAT_1 = 0.7
    DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
    SPIKE_BUFFER_SIZE_1 = 30

    # Output_layer
    N_OUTPUTS = 10
    TAU_S_OUTPUT = 0.130
    THRESHOLD_HAT_OUTPUT = 2.0
    DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
    SPIKE_BUFFER_SIZE_OUTPUT = 30

    # Training parameters
    N_TRAINING_EPOCHS = 10
    N_TRAIN_SAMPLES = 60000
    N_TEST_SAMPLES = 10000
    TRAIN_BATCH_SIZE = 50
    TEST_BATCH_SIZE = 100
    N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
    N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
    TRAIN_PRINT_PERIOD = 0.1
    TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
    TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
    TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
    LEARNING_RATE = 0.002
    LR_DECAY_EPOCH = 10  # Perform decay very n epochs
    LR_DECAY_FACTOR = 1.0
    MIN_LEARNING_RATE = 0
    TAU_LOSS = 0.005
    TARGET_FALSE = 3
    TARGET_TRUE = 15

    # Plot parameters
    EXPORT_METRICS = True
    EXPORT_DIR = Path("/content/SNN-CAPSTONE/results/train_ttfs_eval_count/output_metrics")
    SAVE_DIR = Path("/content/SNN-CAPSTONE/results/train_ttfs_eval_count/best_model")


    def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
        return cp.random.uniform(0.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


    def weight_initializer_out(n_post: int, n_pre: int) -> cp.ndarray:
        return cp.random.uniform(0.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)

    max_int = np.iinfo(np.int32).max
    np_seed = np.random.randint(low=0, high=max_int)
    cp_seed = np.random.randint(low=0, high=max_int)
    np.random.seed(np_seed)
    cp.random.seed(cp_seed)
    print(f"Numpy seed: {np_seed}, Cupy seed: {cp_seed}")

    if EXPORT_METRICS and not EXPORT_DIR.exists():
        EXPORT_DIR.mkdir()

    # Dataset
    print("Loading datasets...")
    dataset = Dataset(path=DATASET_PATH)

    print("Creating network...")
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)

    hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                            theta=THRESHOLD_HAT_1,
                            delta_theta=DELTA_THRESHOLD_1,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_1,
                            name="Hidden layer 1")
    network.add_layer(hidden_layer)

    output_layer = LIFLayer(previous_layer=hidden_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            weight_initializer=weight_initializer_out,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct_ttfs = TTFSSoftmaxCrossEntropy(tau=TAU_LOSS)
    loss_fct_count = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)

    # Metrics
    training_steps = 0
    train_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_train_ttfs", decimal=4)
    train_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_train_ttfs")
    train_silent_label_monitor = SilentLabelsMonitor()
    train_time_monitor = TimeMonitor()
    train_monitors_manager = MonitorsManager([train_loss_monitor,
                                            train_accuracy_monitor,
                                            train_silent_label_monitor,
                                            train_time_monitor],
                                            print_prefix="Train | ")

    test_loss_ttfs_monitor = LossMonitor(export_path=EXPORT_DIR / "train_ttfs_loss_ttfs_test", decimal=4)
    test_loss_count_monitor = LossMonitor(export_path=EXPORT_DIR / "train_ttfs_loss_count_test", decimal=4)
    test_accuracy_ttfs_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "train_ttfs_accuracy_ttfs_test")
    test_accuracy_count_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "train_ttfs_accuracy_count_test")
    test_learning_rate_monitor = ValueMonitor(name="Learning rate", decimal=5)
    # Only monitor LIF layers
    test_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
    test_silent_monitors = {l: SilentNeuronsMonitor(l.name, export_path=EXPORT_DIR / ("silent_neurons_" + l.name))
                        for l in network.layers if isinstance(l, LIFLayer)}
    test_norm_monitors = {l: WeightsNormMonitor(l.name, export_path=EXPORT_DIR / ("weight_norm_" + l.name))
                        for l in network.layers if isinstance(l, LIFLayer)}
    test_time_monitor = TimeMonitor()
    all_test_monitors = [test_loss_ttfs_monitor, test_loss_count_monitor, 
                        test_accuracy_ttfs_monitor, test_accuracy_count_monitor,
                        test_learning_rate_monitor]
    all_test_monitors.extend(test_spike_counts_monitors.values())
    all_test_monitors.extend(test_silent_monitors.values())
    all_test_monitors.extend(test_norm_monitors.values())
    all_test_monitors.append(test_time_monitor)
    test_monitors_manager = MonitorsManager(all_test_monitors,
                                            print_prefix="Test | ")
    
    # Initialize a dictionary to hold spike count data
    spike_counts = {i: [] for i in range(10)}  # 10 digits in MNIST

    best_acc = 0.0

    # initial no training accuracy
    epoch_metrics = 0.0
    test_time_monitor.start()
    for batch_idx in range(N_TEST_BATCH):
        spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
        network.reset()
        network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME)
        out_spikes, n_out_spikes = network.output_spike_trains
        
        # ttfs evaluation
        pred_ttfs = loss_fct_ttfs.predict(out_spikes, n_out_spikes)
        loss_ttfs = loss_fct_ttfs.compute_loss(out_spikes, n_out_spikes, labels)

        pred_ttfs_cpu = pred_ttfs.get()
        cor_pred_ttfs = pred_ttfs_cpu[::30]
        loss_ttfs_cpu = loss_ttfs.get()
        test_loss_ttfs_monitor.add(loss_ttfs_cpu)
        test_accuracy_ttfs_monitor.add(cor_pred_ttfs, labels)

        # count evaluation
        pred_count = loss_fct_count.predict(out_spikes, n_out_spikes)
        loss_count = loss_fct_count.compute_loss(out_spikes, n_out_spikes, labels)

        pred_count_cpu = pred_count.get()
        loss_count_cpu = loss_count.get()
        test_loss_count_monitor.add(loss_count_cpu)
        test_accuracy_count_monitor.add(pred_count_cpu, labels)

        for l, mon in test_spike_counts_monitors.items():
            mon.add(l.spike_trains[1])

        for l, mon in test_silent_monitors.items():
            mon.add(l.spike_trains[1])

    for l, mon in test_norm_monitors.items():
        mon.add(l.weights)

    test_learning_rate_monitor.add(optimizer.learning_rate)

    records = test_monitors_manager.record(epoch_metrics)
    test_monitors_manager.print(epoch_metrics)
    test_monitors_manager.export()

    acc = records[test_accuracy_count_monitor]
    if acc > best_acc:
        best_acc = acc
        network.store(SAVE_DIR)
        print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save to: {SAVE_DIR}")

    print("Training...")
    for epoch in range(N_TRAINING_EPOCHS):
        train_time_monitor.start()
        dataset.shuffle()

        # Learning rate decay
        if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
            optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE)

        for batch_idx in range(N_TRAIN_BATCH):
            # Get next batch
            spikes, n_spikes, labels = dataset.get_train_batch(batch_idx, TRAIN_BATCH_SIZE)

            # Inference
            network.reset()
            network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME, training=True)
            out_spikes, n_out_spikes = network.output_spike_trains

            # Get the spike count from the hidden layer
            hidden_layer_spike_count = hidden_layer.spike_trains[1].get()

            # Store the spike count data
            for i, label in enumerate(labels):
                spike_counts[label].append(hidden_layer_spike_count[i])

            # Predictions, loss and errors
            pred_ttfs = loss_fct_ttfs.predict(out_spikes, n_out_spikes)
            loss_ttfs, errors_ttfs = loss_fct_ttfs.compute_loss_and_errors(out_spikes, n_out_spikes, labels)

            pred_ttfs_cpu = pred_ttfs.get()[::SPIKE_BUFFER_SIZE_OUTPUT]
            loss_ttfs_cpu = loss_ttfs.get()[::SPIKE_BUFFER_SIZE_OUTPUT]
            n_out_spikes_cpu = n_out_spikes.get()

            # Update monitors
            train_loss_monitor.add(loss_ttfs_cpu)
            train_accuracy_monitor.add(pred_ttfs_cpu, labels)
            train_silent_label_monitor.add(n_out_spikes_cpu, labels)

            # Compute gradient
            gradient = network.backward(errors_ttfs)
            avg_gradient = [None if g is None else cp.mean(g, axis=0) for g in gradient]
            del gradient

            # Apply step
            deltas = optimizer.step(avg_gradient)
            del avg_gradient

            network.apply_deltas(deltas)
            del deltas

            training_steps += 1
            epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES

            # Training metrics
            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                # Compute metrics

                train_monitors_manager.record(epoch_metrics)
                train_monitors_manager.print(epoch_metrics)
                train_monitors_manager.export()
            
            # Test evaluation with initial no training accuracy
            if training_steps % TEST_PERIOD_STEP == 0:
                test_time_monitor.start()
                for batch_idx in range(N_TEST_BATCH):
                    spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
                    network.reset()
                    network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME)
                    out_spikes, n_out_spikes = network.output_spike_trains
                    
                    # ttfs evaluation
                    pred_ttfs = loss_fct_ttfs.predict(out_spikes, n_out_spikes)
                    loss_ttfs = loss_fct_ttfs.compute_loss(out_spikes, n_out_spikes, labels)

                    pred_ttfs_cpu = pred_ttfs.get()[::SPIKE_BUFFER_SIZE_OUTPUT]
                    loss_ttfs_cpu = loss_ttfs.get()[::SPIKE_BUFFER_SIZE_OUTPUT]
                    test_loss_ttfs_monitor.add(loss_ttfs_cpu)
                    test_accuracy_ttfs_monitor.add(pred_ttfs_cpu, labels)

                    # count evaluation
                    pred_count = loss_fct_count.predict(out_spikes, n_out_spikes)
                    loss_count = loss_fct_count.compute_loss(out_spikes, n_out_spikes, labels)

                    pred_count_cpu = pred_count.get()[::SPIKE_BUFFER_SIZE_OUTPUT]
                    loss_count_cpu = loss_count.get()[::SPIKE_BUFFER_SIZE_OUTPUT]
                    test_loss_count_monitor.add(loss_count_cpu)
                    test_accuracy_count_monitor.add(pred_count_cpu, labels)

                    for l, mon in test_spike_counts_monitors.items():
                        mon.add(l.spike_trains[1])

                    for l, mon in test_silent_monitors.items():
                        mon.add(l.spike_trains[1])

                for l, mon in test_norm_monitors.items():
                    mon.add(l.weights)

                test_learning_rate_monitor.add(optimizer.learning_rate)

                records = test_monitors_manager.record(epoch_metrics)
                test_monitors_manager.print(epoch_metrics)
                test_monitors_manager.export()

                acc = records[test_accuracy_ttfs_monitor]
                if acc > best_acc:
                    best_acc = acc
                    network.store(SAVE_DIR)
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save to: {SAVE_DIR}")

                    image = get_image(0)
                    plot_spike_train(image, network, SIMULATION_TIME, 'spike train', SAVE_DIR)

        # Calculate average spike counts
        avg_spike_counts = {digit: np.mean(spike_counts[digit], axis=0) for digit in spike_counts}
        
        # Create a figure to visualize network activity and sparsity
        os.makedirs(export_path + '/neuron_plots', exist_ok=True)
        create_spike_count_map(avg_spike_counts, 800, 15, f'SpikeCountMap_800Neurons_TTFS_eval_Count_Epoch{epoch + 1}', export_path)
        create_spike_count_map(avg_spike_counts, 100, 15, f'SpikeCountMap_100Neurons_TTFS_eval_Count_Epoch{epoch + 1}', export_path)