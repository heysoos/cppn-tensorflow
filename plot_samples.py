import matplotlib.pyplot as plt
import imageio
import sampler
import numpy as np
import os
from operator import itemgetter

def load_images_and_jsons(filters, img_folder=None):

    # if len(filters) < 4:
    #     raise Exception('Please include more parameters in filter.')

    # transform filter dict to match savefile convention
    filters_isNone = {}
    sort_keys = []
    for k, v in filters.items():
        if filters[k] is None:
            filters_isNone[k] = ''
            sort_keys.append(k)
        else:
            filters_isNone[k] = str(filters[k]) + '_'
    sort_keys.append('iteration')

    filter_list = [
        'N' + str(filters_isNone['total_neurons']),
        'L' + str(filters_isNone['num_layers']),
        'w' + str(filters_isNone['omega']),
        'a' + str(filters_isNone['alpha']),
        'm' + str(filters_isNone['mu'])
    ]

    dir = 'save/img_architectures'
    img_dir = os.path.join(dir, img_folder)
    img_json_dir = os.path.join(img_dir, 'json')
    files = os.listdir(img_dir)

    # filter through imgs and keep only desired params

    img_list = []
    for file in files:
        if all(filter in file for filter in filter_list):
            img_list.append(file)

    # img_list.sort()

    # load files
    imgs = []
    imgs_params = []
    for img_ind in img_list:
        # create full pathname for individual img and its json file
        img_img_dir = os.path.join(img_dir, img_ind)
        img_img_json_dir = os.path.join(img_json_dir, os.path.splitext(img_ind)[0] + '.json')

        # load img and its json file with its params
        imgs.append(imageio.imread(img_img_dir))

        data = sampler.loadJSON(img_img_json_dir)
        imgs_params.append(data)

    # sort images according to the parameters not held constant
    # get indices of the new sorted list
    sort_list = [tuple((img_params[k]) for k in sort_keys) for img_params in imgs_params]
    sort_indices = sorted(enumerate(sort_list), key=itemgetter(1))
    sort_indices = [i[0] for i in sort_indices]

    # uncomment to debug/check if sorting works as intended
    # img_list = [img_list[i] for i in sort_indices]

    # re-arrange img list according to sort indices
    imgs = [imgs[i] for i in sort_indices]
    imgs_params = [imgs_params[i] for i in sort_indices]

    return imgs, imgs_params

def plot_samples(filters, img_folder=None):

    imgs, imgs_params = load_images_and_jsons(filters=filters, img_folder=img_folder)

    plt.style.use('seaborn-dark')
    plt.rcParams['font.size'] = 5
    # plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    # plt.rcParams['axes.titlesize'] = 10
    # plt.rcParams['xtick.labelsize'] = 8
    # plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True

    latex_dict = {
            'total_neurons': 'N',
            'num_layers': 'L',
            'omega': '\\omega',
            'alpha': '\\alpha',
            'mu': '\\mu',
    }

    # img_params = {x.replace(f, '') for x in img_list for f in filters}

    numImgs = 5  # number of images per architecture initialization
    ncols = numImgs + 1  # added one extra column to also plot net_size
    nrows = int(len(imgs) / numImgs)


    figsize = [6, 8]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150)

    # suptitle
    suptitle = []
    for k, v, in filters.items():
        if filters[k] is not None:
            if not len(suptitle):
                suptitle.append(latex_dict[k] + '=' + str(filters[k]))
            else:
                suptitle.append(', ' + latex_dict[k] + '=' + str(filters[k]))
    suptitle.insert(0, '$')
    suptitle.append('$')
    # suptitle = r"$" + "".join(suptitle) + "$"
    fig.suptitle("".join(suptitle))

    img_counter = 0
    for i, axi in enumerate(ax.flat):

        if not i % ncols:

            net_size = np.array(imgs_params[img_counter + 1]['net_size'])
            x = np.arange(1, len(net_size) + 1)
            axi.bar(x, net_size)
            axi.set_yticks(np.unique(net_size))
            axi.set_aspect(len(x) / np.max(net_size))
            # axi.set_aspect('equal')

            if i > 0:
                net_size_old = np.array(imgs_params[img_counter]['net_size'])
                if len(net_size) is not len(net_size_old):
                    axi.set_xticks(x)
                else:
                    axi.set_xticks([])



            # axi title
            title = []
            for k, v in filters.items():
                if filters[k] is None:
                    if not len(title):
                        title.append(latex_dict[k] + '=' + str(imgs_params[img_counter + 1][k]))
                    else:
                        title.append(', ' + latex_dict[k] + '=' + str(imgs_params[img_counter + 1][k]))
            title.insert(0, '$')
            title.append('$')
            axi.set_title("".join(title))

            # plot network axis labels on the last row
            if (i / ncols) == (nrows - 1):
                axi.set_ylabel(' of Neurons')
                axi.set_xlabel('Net Size')

        else:
            axi.imshow(imgs[img_counter])
            axi.set_xticks([])
            axi.set_yticks([])
            img_counter += 1

        if i < ncols and i > 0:
            axi.set_title(str(i % ncols))

    plt.show()


if __name__ == '__main__':

    N = 100  # total_neurons
    L = 3  # num layers
    omega = -2  # omega
    alpha = 2  # alpha
    mu = None  # mu

    filters = {
        'total_neurons': N,
        'num_layers': L,
        'omega': omega,
        'alpha': alpha,
        'mu': mu,
    }

    img_folder = '19-04-30-19-20-18.898760'

    plot_samples(filters, img_folder)

