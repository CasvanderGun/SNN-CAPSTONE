import sys
import os

sys.path.insert(0, "../")  # Add repository root to python path

from multiple_runs.train_count_eval_ttfs_func import train_count_eval_ttfs

epochs = 2
num_runs = 2

path = '/kaggle/working/SNN-CAPSTONE/results/train_multiple_runsv2/Run_'

for run in range(num_runs):
    print('Run: ' + str(run + 1))
    os.makedirs(path + str(run + 1), exist_ok=True)
    train_count_eval_ttfs(epochs, path + str(run + 1))