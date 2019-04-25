import sampler
import numpy as np
import itertools
from os import path, makedirs
from datetime import datetime
import imageio
import matplotlib.pyplot as plt
from multiprocessing import Process

# seed = 12345678
# np.random.seed(seed=seed)
#
# # net_size = C_n * exp( mu * l / total_neurons) * (alpha + sin(-w*l)
#
#
# total_neurons = [50, 250, 500]
# num_layers = [3, 5, 10]
# omega = [0, 0.5, 1, 1.5]  # sinusoidal frequency (affects bottleneck shape and frequency)
# alpha = [2, 5, 10]  # constant factor that controls the amplitude of the sinusoid (>2)
# mu = [0, 0.2, 0.5, 1]  # exp. decay rate
#
# params = [total_neurons, num_layers, omega, alpha, mu]
# params = list(itertools.product(*params))
#
# for param_args in params:
#     net_size = sampler.generate_architecture(*param_args)
#     l = np.arange(1, param_args[1]+1)
#     plt.plot(l, np.log10(net_size), alpha=0.2)
#     # print(np.sum(net_size))
#
# plt.show()

def generate_params():
    total_neurons = [50, 250, 500]
    num_layers = [3, 5, 10]
    omega = [0, 0.5, 1, 1.5]  # sinusoidal frequency (affects bottleneck shape and frequency)
    alpha = [2, 5, 10]  # constant factor that controls the amplitude of the sinusoid (>2)
    mu = [0, 0.2, 0.5, 1]  # exp. decay rate

    params = [total_neurons, num_layers, omega, alpha, mu]
    params = list(itertools.product(*params))

    return params


def generate_images(total_neurons, num_layers, omega, alpha, mu):
    net_size = sampler.generate_architecture(total_neurons, num_layers, omega, alpha, mu)
    seed = 0
    # TODO: fix img_null nonesense...
    # GENERATE SAMPLER/ARCHITECTURE
    img_sampler = sampler.Sampler(z_dim=16, scale=8, net_size=net_size,
                  num_layers=num_layers, c_dim=3, seed=seed, img=None)

    # GENERATE IMAGES
    z2 = img_sampler.generate_z()[0]
    img = None

    z_scale = 1
    z_factor = np.random.normal(size=16) * z_scale

    sortidx = np.argsort(np.abs(z_factor))
    sortidy = np.argsort(np.abs(z2[0]))

    z_factor = z_factor[sortidx]
    z2 = z2[sortidy]
    zz = z2 * z_factor

    f_params = [15, 0.1, 0, 0, 0]

    img_data = img_sampler.generate(zz, x_dim=512, y_dim=512, scale=1,
                                f_params=f_params, img=img)

    time = datetime.now().strftime("%y-%m-%d-%H-%M-%S.%f")
    folder = 'save/img_architectures/'
    if not path.exists(folder):
        makedirs(folder)
    figname = folder + time + '.png'
    imageio.imwrite(figname, (img_data * 255).astype(np.uint8), format='png')
    img_sampler.cppn.close()  # necessary?!


if __name__ == "__main__":
    params = generate_params()

    procs = []
    # proc = Process(target=generate_image) # instantiating without any argument
    # procs.append(proc)
    # proc.start()

    # instantiating process with arguments
    for param in params:
        proc = Process(target=generate_images, args=param)
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
