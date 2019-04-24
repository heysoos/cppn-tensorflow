import sampler
import numpy as np

seed = 12345678
np.random.seed(seed=seed)

# net_size = C_n * exp( mu * l / total_neurons) * (alpha + sin(-w*l)

total_neurons = 250
num_layers = 10
w = 1.25  # sinusoidal frequency (affects bottleneck shape and frequency)
alpha = 5  # constant factor that controls the amplitude of the sinusoid
mu = 0.5  # exp. decay rate

for
net_size = sampler.generate_architecture(total_neurons=total_neurons, num_layers=num_layers,
                                         w=w, alpha=alpha, mu=mu)

