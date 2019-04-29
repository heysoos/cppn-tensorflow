import matplotlib.pyplot as plt
import numpy as np
import os

def plot_samples(filters, img_folder=None):
    dir = 'save/img_architectures'
    # img_folder = 'new_params'
    img_dir = os.path.join(dir, img_folder)

    files = os.listdir(img_dir)

    # filters = ['N250', 'L3', 'w-1', 'a2']

    if len(filters) < 4:
        raise Exception('Please include more parameters in filter.')

    imgs = []

    for file in files:
        if all(filter in file for filter in filters):
            imgs.append(file)
            imgs_params.append()

    imgs.sort()

    img_params = {x.replace(f, '') for x in imgs for f in filters}

    numImgs = 5  # number of images per architecture initialization
    ncols = numImgs + 1  # added one column to plot net_size
    nrows = len(imgs) / numImgs

    w = 10
    h = 10
    figsize = [6, 8]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    ax = []

    for i, axi in enumerate(ax.flat):

        img = []
        axi.imshow(img)

        if (i < numImgs) and (i > 0):
            ax[-1].set_title('img:' + str(i))

        if not i % ncols:
            ax[-1].set_ylabel('')


if __name__ == '__main__':
    filters = ['N250', 'L3', 'w-1', 'a2']
    img_folder = 'new_params'

    plot_samples(filters, img_folder)

