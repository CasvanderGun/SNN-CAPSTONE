import sys
import os

sys.path.insert(0, "../../../")  # Add repository root to python path

from experiments.mnist.Own_loss.train_decay_rate import train_decay_rate

epochs = 1
simulation_time = 0.2
decay_rate = 1
runs = 3

root_path = '/kaggle/working/SNN-CAPSTONE/'
path = root_path + "results/train_decay_rate/multiple_runs/Run_"

for run in range(runs):
    print('\nRun: ' + str(run + 1) + '\n')
    run_path = path + str(run + 1)
    os.makedirs(run_path , exist_ok=True)
    train_decay_rate(epochs, decay_rate, simulation_time, run_path, root_path)