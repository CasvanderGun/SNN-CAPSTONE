import sys

sys.path.insert(0, "../../")  # Add repository root to python path

from experiments.mnist.multiple_runs.train_count_eval_ttfs_func import train_count_eval_ttfs_func

epochs = 30

path = '/content/SNN-CAPSTONE/results/train_multiple_runs/Run_'

for run in range(5):
    print('Run: ' + str(run + 1))
    train_count_eval_ttfs_func(epochs, path + str(run + 1))