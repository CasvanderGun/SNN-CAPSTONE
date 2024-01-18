import sys

sys.path.insert(0, "../../")  # Add repository root to python path

from experiments.mnist.train_decay_rate import train_decay_rate

epochs = 20
simulation_time = 0.2
decay_rates = [3, 2, 0.5]

paths = ["/content/SNN-CAPSTONE/results/train_decay_rate/decay_rate_3",
         "/content/SNN-CAPSTONE/results/train_decay_rate/decay_rate_2",
         "/content/SNN-CAPSTONE/results/train_decay_rate/decay_rate_0.5"]

for run, decay_rate in enumerate(decay_rates):
    print(f'Run {run+1}\nDecay rate {decay_rate}\n')
    train_decay_rate(epochs, decay_rate, simulation_time, paths[run])
