import sys

sys.path.insert(0, "../../../")  # Add repository root to python path

from experiments.mnist.Own_loss.train_decay_rate import train_decay_rate

epochs = 20
simulation_time = 0.2
decay_rates = [1]

path = "/content/SNN-CAPSTONE/results/train_decay_rate/Simulation_time_0.2/decay_rate_"

for run, decay_rate in enumerate(decay_rates):
    print(f'Run {run+1}\nDecay rate {decay_rate}\n')
    train_decay_rate(epochs, decay_rate, simulation_time, path + str(decay_rate))
