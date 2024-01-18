import sys

sys.path.insert(0, "../")  # Add repository root to python path

from multiple_runs.train_count_eval_ttfs_func import train_count_eval_ttfs

epochs = 30

path = '/kaggle/working/SNN-CAPSTONE/results/train_multiple_runs/Run_'

for run in range(5):
    print('Run: ' + str(run + 1))
    train_count_eval_ttfs(epochs, path + str(run + 1))