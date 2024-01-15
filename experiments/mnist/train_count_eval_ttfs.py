from pathlib import Path
import cupy as cp
import numpy as np

import sys

sys.path.insert(0, "../../")  # Add repository root to python path

from Dataset import Dataset
from bats.Monitors import *
from bats.Layers import InputLayer, LIFLayer
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *

# Dataset
DATASET_PATH = Path("../../datasets/mnist.npz")

N_INPUTS = 28 * 28
SIMULATION_TIME = 0.2

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

# Training parameters
N_TRAINING_EPOCHS = 100
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
LEARNING_RATE = 0.003
LR_DECAY_EPOCH = 10  # Perform decay very n epochs
LR_DECAY_FACTOR = 1.0
MIN_LEARNING_RATE = 0

# SPIKE COUNT
TARGET_FALSE = 3
TARGET_TRUE = 15

# TIME TO FIRST SPIKE
TAU_LOSS = 0.005

# Plot parameters
EXPORT_METRICS = True
EXPORT_DIR = Path("/content/SNN-CAPSTONE/results/train_count_eval_ttfs/output_metrics")
SAVE_DIR = Path("/content/SNN-CAPSTONE/results/train_count_eval_ttfs/best_model")


def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


if __name__ == "__main__":
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
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    eval_loss_fct = TTFSSoftmaxCrossEntropy(tau=TAU_LOSS) ##NIEUW
    
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)

    # EVALUATING ON TRAINING DATA WITH TRAINING LOSS FUNCTION
    training_steps = 0
    train_train_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "train_train_loss_loss")
    train_train_loss_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "train_train_loss_accuracy")
    train_train_loss_silent_label_monitor = SilentLabelsMonitor()
    train_train_loss_time_monitor = TimeMonitor()
    train_train_loss_monitors_manager = MonitorsManager([train_train_loss_monitor,
                                                         train_train_loss_accuracy_monitor,
                                                         train_train_loss_silent_label_monitor,
                                                         train_train_loss_time_monitor],
                                                         print_prefix="Train on train loss function | ")

    
    # EVALUATING ON TEST DATA WITH TRAINING LOSS FUNCTION
    test_train_loss_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "test_train_loss_loss")
    test_train_loss_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "test_train_loss_accuracy")
    test_train_loss_learning_rate_monitor = ValueMonitor(name="Learning rate", decimal=5)
    test_train_loss_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
    test_train_loss_silent_monitors = {l: SilentNeuronsMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
    test_train_loss_norm_monitors = {l: WeightsNormMonitor(l.name, export_path=EXPORT_DIR / ("weight_norm_" + l.name))
                          for l in network.layers if isinstance(l, LIFLayer)}
    test_train_loss_time_monitor = TimeMonitor()
    all_test_train_loss_monitors = [test_train_loss_loss_monitor, test_train_loss_accuracy_monitor, test_train_loss_learning_rate_monitor]
    all_test_train_loss_monitors.extend(test_train_loss_spike_counts_monitors.values())
    all_test_train_loss_monitors.extend(test_train_loss_silent_monitors.values())
    all_test_train_loss_monitors.extend(test_train_loss_norm_monitors.values())
    all_test_train_loss_monitors.append(test_train_loss_time_monitor)
    test_train_loss_monitors_manager = MonitorsManager(all_test_train_loss_monitors,
                                                  print_prefix="Test on train loss function| ")
    

    # EVALUATING ON TEST DATA WITH EVALUATION LOSS FUNCTION
    test_eval_loss_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "test_eval_loss_loss") 
    test_eval_loss_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "test_eval_loss_accuracy")
    test_eval_loss_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
    test_eval_loss_silent_monitors = {l: SilentNeuronsMonitor(l.name, export_path=EXPORT_DIR / ("silent_neurons_" + l.name))
                          for l in network.layers if isinstance(l, LIFLayer)}
    test_eval_loss_norm_monitors = {l: WeightsNormMonitor(l.name, export_path=EXPORT_DIR / ("test_eval_loss_weight_norm_" + l.name))
                          for l in network.layers if isinstance(l, LIFLayer)}
    test_eval_loss_time_monitor = TimeMonitor()
    all_test_eval_loss_monitors = [test_eval_loss_loss_monitor, test_eval_loss_accuracy_monitor]
    all_test_eval_loss_monitors.extend(test_eval_loss_spike_counts_monitors.values())
    all_test_eval_loss_monitors.extend(test_eval_loss_silent_monitors.values())
    all_test_eval_loss_monitors.append(test_eval_loss_time_monitor)
    test_eval_loss_monitors_manager = MonitorsManager(all_test_eval_loss_monitors,
                                            print_prefix="Test on eval loss function | ")

    best_acc = 0.0

    print("Training...")
    for epoch in range(N_TRAINING_EPOCHS):
        train_train_loss_time_monitor.start()
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

            # Predictions, loss and errors
            train_loss_pred = loss_fct.predict(out_spikes, n_out_spikes)
            train_loss, errors = loss_fct.compute_loss_and_errors(out_spikes, n_out_spikes, labels)

            train_pred_cpu = train_loss_pred.get()
            train_loss_cpu = train_loss.get()

            


            n_out_spikes_cpu = n_out_spikes.get()

            # Update monitors
            train_train_loss_monitor.add(train_loss_cpu)
            train_train_loss_accuracy_monitor.add(train_pred_cpu, labels)
            train_train_loss_silent_label_monitor.add(n_out_spikes_cpu, labels)

            # Compute gradient
            gradient = network.backward(errors)
            avg_gradient = [None if g is None else cp.mean(g, axis=0) for g, layer in zip(gradient, network.layers)]
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

                train_train_loss_monitors_manager.record(epoch_metrics)
                train_train_loss_monitors_manager.print(epoch_metrics)
                train_train_loss_monitors_manager.export()

            #  Evaluation
            if training_steps % TEST_PERIOD_STEP == 0:
                test_eval_loss_time_monitor.start()
                for batch_idx in range(N_TEST_BATCH):
                    spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
                    
                    network.reset()
                    network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME)
                    
                    out_spikes, n_out_spikes = network.output_spike_trains

                    # EVALUATING ON TRAIN LOSS FUNCTION

                    test_train_loss_pred = loss_fct.predict(out_spikes, n_out_spikes)
                    test_train_loss = loss_fct.compute_loss(out_spikes, n_out_spikes, labels)

                    test_train_loss_pred_cpu = test_train_loss_pred.get()
                    test_train_loss_cpu = test_train_loss.get()
                    
                    test_train_loss_loss_monitor.add(test_train_loss_cpu)
                    test_train_loss_accuracy_monitor.add(test_train_loss_pred_cpu, labels)

                    for l, mon in test_train_loss_spike_counts_monitors.items():
                        mon.add(l.spike_trains[1])

                    for l, mon in test_train_loss_silent_monitors.items():
                        mon.add(l.spike_trains[1])

                    for l, mon in test_train_loss_norm_monitors.items():
                        mon.add(l.weights)

                    test_train_loss_learning_rate_monitor.add(optimizer.learning_rate)

                    # EVALUATING ON EVAL LOSS FUNCTION

                    test_eval_loss_pred = eval_loss_fct.predict(out_spikes, n_out_spikes)
                    test_eval_loss = eval_loss_fct.compute_loss(out_spikes, n_out_spikes, labels)

                    test_eval_loss_pred_cpu = test_eval_loss_pred.get()
                    
                    print(test_eval_loss_pred.shape)
                    print(test_eval_loss_pred)

                    test_eval_loss_cpu = test_eval_loss.get()
                    
                    test_eval_loss_loss_monitor.add(test_eval_loss_cpu)
                    test_eval_loss_accuracy_monitor.add(test_eval_loss_pred_cpu, labels)

                    for l, mon in test_eval_loss_spike_counts_monitors.items():
                        mon.add(l.spike_trains[1])

                    for l, mon in test_eval_loss_silent_monitors.items():
                        mon.add(l.spike_trains[1])

                    for l, mon in test_eval_loss_norm_monitors.items():
                        mon.add(l.weights)


                test_records = test_train_loss_monitors_manager.record(epoch_metrics)
                test_train_loss_monitors_manager.print(epoch_metrics)
                test_train_loss_monitors_manager.export()

                records = test_eval_loss_monitors_manager.record(epoch_metrics)
                test_eval_loss_monitors_manager.print(epoch_metrics)
                test_eval_loss_monitors_manager.export()

                acc = test_records[test_train_loss_accuracy_monitor]
                
                if acc > best_acc:
                    best_acc = acc
                    network.store(SAVE_DIR)
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save to: {SAVE_DIR}")
