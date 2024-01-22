import sys
import os

sys.path.insert(0, "../")  # Add repository root to python path

from multiple_runs.train_ttfs_eval_count_func import train_ttfs_eval_count

epochs = 30
num_runs = 2

path = '/kaggle/working/SNN-CAPSTONE/results/train_ttfs_eval_count/train_multiple_runs/Run_'

for run in range(num_runs):
    print('Run: ' + str(run + 1))
    os.makedirs(path + str(run + 1), exist_ok=True)
    train_ttfs_eval_count(epochs, path + str(run + 1))