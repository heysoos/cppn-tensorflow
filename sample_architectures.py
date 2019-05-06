import sampler
import numpy as np
import itertools
from os import path, makedirs
from datetime import datetime
import imageio
import matplotlib.pyplot as plt
import json
from multiprocessing import Pool


def generate_params(folder):
    # total_neurons = [50, 250, 500]
    # num_layers = [3, 5, 10]
    # omega = [0, 0.5, 1, 1.5]  # sinusoidal frequency (affects bottleneck shape and frequency)
    # alpha = [2, 5, 10]  # constant factor that controls the amplitude of the sinusoid (>2)
    # mu = [0, 0.2, 0.5, 1]  # exp. decay rate

    total_neurons = [100, 250, 500]
    num_layers = [3, 5, 10]
    omega = [-2, -1, -0.5, 0, 0.5, 1, 2]  # sinusoidal frequency (affects bottleneck shape and frequency)
    alpha = [2, 5]  # constant factor that controls the amplitude of the sinusoid (>2)
    mu = [0, 0.1, -0.1, 0.5, -0.5, 1, -1]  # exp. decay rate

    # total_neurons = [100]
    # num_layers = [3]
    # omega = [-2, -1, -0.5, 0, 0.5, 1, 2]  # sinusoidal frequency (affects bottleneck shape and frequency)
    # alpha = [2]  # constant factor that controls the amplitude of the sinusoid (>2)
    # mu = [0]  # exp. decay rate

    params = [total_neurons, num_layers, omega, alpha, mu]
    params = list(itertools.product(*params))

    for i, param in enumerate(params):
        params[i] = params[i] + (folder,)

    # params = [list(tup) for tup in list(itertools.product(*params))]

    return params

def plot_architecture_samples(params):
    alpha = 0.1
    for param in params:
        net_size = sampler.generate_architecture(*param[:-1])
        plt.plot(net_size, alpha=alpha)

    plt.show()



def generate_images(total_neurons, num_layers, omega, alpha, mu, folder):
    use_z = False
    net_size = sampler.generate_architecture(total_neurons, num_layers, omega, alpha, mu)

    seed = 0
    # TODO: fix img_null nonesense...
    # GENERATE SAMPLER/ARCHITECTURE
    img_sampler = sampler.Sampler(z_dim=16, scale=8, net_size=net_size,
                  num_layers=num_layers, c_dim=3, seed=seed, img=None)

    num_images_per_architecture = 20
    # folder = 'save/img_architectures/'
    for i in range(num_images_per_architecture):
        # GENERATE IMAGES
        if i is not 0:
            img_sampler.reinit()

        if use_z:
            # generates a new latent vector z for each initialization.

            z = img_sampler.generate_z()[0]

            z_scale = 1
            # z_scale = np.random.uniform(1, 3)
            z_factor = np.random.normal(size=16) * z_scale  # mostly used for music

            sortidx = np.argsort(np.abs(z_factor))
            sortidy = np.argsort(np.abs(z[0]))

            z_factor = z_factor[sortidx]
            z = z[sortidy]
            zz = z * z_factor

        else:
            # set the latent vector to 0
            z_scale = 0
            zz = img_sampler.generate_z()[0] * z_scale

        scale = 1
        # scale = np.random.uniform(0.5, 3)
        img = None
        f_params = [0, 0, 0, 0, 0]

        ########################### GENERATE IMAGE ###########################
        img_data = img_sampler.generate(zz, x_dim=512, y_dim=512, scale=scale,
                                    f_params=f_params, img=img)
        ######################################################################

        params_str = '/N{}_L{}_w{}_a{}_m{}_'.format(total_neurons, num_layers, omega, alpha, mu)

        folder_json = path.join(folder, 'json')
        if not path.exists(folder):
            makedirs(folder)

        if not path.exists(folder_json):
            makedirs(folder_json)

        data = {
            'net_size': net_size.tolist(),
            'total_neurons': total_neurons,
            'num_layers': num_layers,
            'omega': omega,
            'alpha': alpha,
            'mu': mu,
            'seed': seed,
            'f_params': f_params,
            'zz': zz.tolist(),
            'iteration': i
        }
        figname = folder + params_str + str(i).zfill(2) + '.png'
        figname_json = folder_json + params_str + str(i).zfill(2) + '.json'

        imageio.imwrite(figname, (img_data * 255).astype(np.uint8), format='png')
        sampler.saveJSON(data=data, filename=figname_json)

    img_sampler.cppn.close()  # necessary?!



# if __name__ == "__main__":
#     params = generate_params()
#
#     procs = []
#     # proc = Process(target=generate_image) # instantiating without any argument
#     # procs.append(proc)
#     # proc.start()
#
#     # instantiating process with arguments
#     for param in params:
#         proc = Process(target=generate_images, args=param)
#         procs.append(proc)
#         proc.start()
#         # proc.join()  # complete the processes
#
#     # # complete the processes
#     for proc in procs:
#         proc.join()


if __name__ == "__main__":

    # save folder
    time = datetime.now().strftime("%y-%m-%d-%H-%M-%S.%f")
    folder = path.join('save/img_architectures', time)

    params = generate_params(folder)

    # plot_architecture_samples(params)

    pool = Pool(processes=11)
    pool.starmap(generate_images, params)



