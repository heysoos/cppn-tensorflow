import numpy as np
import os
import sampler
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import colors
from orderedset import OrderedSet



def architecture_correlation_means(img_folder, filter_keys):
    # load jsons, use to sort corr_path list, return corr_paths

    dir = 'save/img_architectures'
    img_dir = os.path.join(dir, img_folder)

    corrs_folder = os.path.join(img_dir, 'corrs')
    jsons_folder = os.path.join(img_dir, 'json')

    # generate savefiles for correlation means
    means_folder = os.path.join(corrs_folder, 'means')
    mean_filename = os.path.join(means_folder, img_folder + '_corr_means.txt')

    # first check if means have already been calculated, if so, load.
    exists = os.path.isfile(mean_filename)
    if exists:
        with open(mean_filename, 'rb') as fp:
            data = pickle.load(fp)
        load_str = 'Loading correlation means: ' + mean_filename
        print(load_str)
        corrs_means = data['corrs_means']
        means_params = data['means_params']
    else:
        calc_str = 'Calculating correlation means...'
        print(calc_str)

        corrs_paths = [os.path.join(corrs_folder, f) for f in os.listdir(corrs_folder)
                       if f.endswith('.txt')]

        imgs_params = []
        for corr in corrs_paths:

            img_name = os.path.splitext(os.path.basename(corr))[0]
            json_path = os.path.join(jsons_folder, img_name + '.json')

            data = sampler.loadJSON(json_path)
            imgs_params.append(data)


        # sort so similar params are together, with iteration number as well
        sort_keys = filter_keys
        sort_keys.append('iteration')

        sort_list = [tuple((img_params[k]) for k in sort_keys) for img_params in imgs_params]
        sort_indices = sorted(enumerate(sort_list), key=itemgetter(1))
        sort_indices = [i[0] for i in sort_indices]

        max_iters = np.max([p[-1] for p in sort_list])

        corrs_paths = [corrs_paths[i] for i in sort_indices]
        imgs_params = [imgs_params[i] for i in sort_indices]

        corrs = []
        for corr_path in corrs_paths:
            with open(corr_path, 'rb') as fp:
                corrs.append(pickle.load(fp))

        # calculate correlation means
        corrs_means = [ np.mean(corrs[i:i + max_iters + 1], axis=0)
                        for i in range(0, len(corrs), max_iters + 1)]

        # gather img params after removing redundancy in iterations
        means_params = [ imgs_params[i]
                         for i in range(0, len(imgs_params), max_iters + 1)]


        data = {
            'corrs_means': corrs_means,
            'means_params': means_params
        }


        # save correlations
        if not os.path.exists(means_folder):
            os.makedirs(means_folder)

        with open(mean_filename, 'wb') as fp:
            pickle.dump(data, fp)

    return corrs_means, means_params

# def plot_correlations_grid(filters, img_folder):
#
#
#
#
#
#
def plot_correlations_all(corrs_means, means_params, highlight):

    values = list(np.sort(np.unique([v[highlight] for v in means_params])))

    plt.style.use('seaborn-dark')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['text.usetex'] = True

    latex_dict = {
            'total_neurons': 'N',
            'num_layers': 'L',
            'omega': '\\omega',
            'alpha': '\\alpha',
            'mu': '\\mu',
    }

    cmap = plt.get_cmap('gnuplot')
    norm = colors.Normalize(vmin=0, vmax=len(values))  # age/color mapping

    fig, ax = plt.subplots(dpi=150)
    ax.set_title('Correlation Function of CPPN Images')
    ax.set_xlabel('Distance')
    ax.set_ylabel('r')

    for i, corr in enumerate(corrs_means):
        corr_value = means_params[i][highlight]
        c = cmap(norm( values.index(corr_value) ))

        label = '$' + latex_dict[highlight] + ': ' + str(corr_value) + '$'


        ax.plot(np.arange(0, len(corr)), corr, alpha=0.1, linewidth=1, c=c, label=label)

    handles, labels = unique_labels(ax)

    leg = ax.legend(handles, labels)  # draw the legend with the filtered handles and labels lists
    for h in leg.legendHandles:
        h.set_alpha(1)

    # plt.yscale('log')
    # plt.xscale('log')

    plt.show()

def unique_labels(ax):
    handles, labels = ax.get_legend_handles_labels()  # get existing legend item handles and labels
    i = np.arange(len(labels))  # make an index for later
    filter = np.array([])  # set up a filter (empty for now)
    unique_labels = list(OrderedSet(labels))  # find unique labels

    for ul in unique_labels:  # loop through unique labels
        filter = np.append(filter, [
            i[np.array(labels) == ul][0]])  # find the first instance of this label and add its index to the filter
    handles = [handles[int(f)] for f in
               filter]  # filter out legend items to keep only the first instance of each repeated label
    labels = [labels[int(f)] for f in filter]

    return handles, labels


if __name__ == '__main__':
    filter_keys = [
        'total_neurons',
        'num_layers',
        'omega',
        'alpha',
        'mu'
    ]

    img_folder = 'big'

    corrs_means, means_params = architecture_correlation_means(img_folder, filter_keys)

    highlight = 'total_neurons'

    plot_correlations_all(corrs_means, means_params, highlight)


