import numpy as np
import os
import sampler
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import colors
from orderedset import OrderedSet
import scipy.stats



def architecture_correlation_means(img_folder, filter_keys, method, forceCalculate=False):
    # load jsons, use to sort corr_path list, return corr_paths

    dir = 'save/img_architectures'
    img_dir = os.path.join(dir, img_folder)

    corrs_folder = os.path.join(img_dir, 'corrs', method)
    jsons_folder = os.path.join(img_dir, 'json')

    # generate savefiles for correlation means
    means_folder = os.path.join(corrs_folder, 'means')
    mean_filename = os.path.join(means_folder, img_folder + '_corr_means.txt')

    # first check if means have already been calculated, if so, load.
    exists = os.path.isfile(mean_filename)
    if exists and not forceCalculate:
        with open(mean_filename, 'rb') as fp:
            data = pickle.load(fp)
        load_str = 'Loading correlation means: ' + mean_filename
        print(load_str)
        corrs_means = data['corrs_means']
        corrs_quantiles = data['corrs_quantiles']
        corrs_vars = data['corrs_vars']
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
        sort_keys = filter_keys[:]
        sort_keys.append('iteration')

        sort_list = [tuple((img_params[k]) for k in sort_keys) for img_params in imgs_params]
        sort_indices = sorted(enumerate(sort_list), key=itemgetter(1))
        sort_indices = [i[0] for i in sort_indices]

        # here p[-1] should correspond to the 'iteration' parameter which designates the
        # img number for the same seed. The idea is to sort, and then average, over imgs
        # generated from the same seed.
        max_iters = np.max([p[-1] for p in sort_list]) + 1

        corrs_paths = [corrs_paths[i] for i in sort_indices]
        imgs_params = [imgs_params[i] for i in sort_indices]

        if method == 'sample':
            # can load all files at once since files are relatively small...
            corrs = []
            for corr_path in corrs_paths:
                with open(corr_path, 'rb') as fp:
                    corrs.append(pickle.load(fp))

            corrs = np.nan_to_num(corrs)

            # calculate correlation means
            corrs_means = [ np.nanmean(corrs[i:i + max_iters], axis=0)
                            for i in range(0, len(corrs), max_iters)]

            corrs_quantiles = [ ( np.quantile(corrs[i:i + max_iters], 0.25, axis=0),
                                  np.quantile(corrs[i:i + max_iters], 0.75, axis=0)
                                  )
                                for i in range(0, len(corrs), max_iters)]

            corrs_vars = [ np.var(corrs[i:i + max_iters], axis=0)
                            for i in range(0, len(corrs), max_iters)]

            # gather img params after removing redundancy in iterations
            means_params = [ imgs_params[i]
                             for i in range(0, len(imgs_params), max_iters)]
        elif method == 'fft' or method == 'fft_windowed':
            #  should load files one by one since they are much larger

            corrs = []
            corrs_means = []
            corrs_quantiles = []
            corrs_vars = []
            means_params = []
            for i, corr_path in enumerate(corrs_paths):
                with open(corr_path, 'rb') as fp:
                    corrs.append(pickle.load(fp))


                if not (i + 1) % (max_iters + 1):
                    # calculate correlation means
                    corrs = np.nan_to_num(corrs)  # set nan correlations to 0

                    corrs_means.append(np.mean(corrs, axis=0))

                    corrs_quantiles.append(
                        (np.quantile(corrs, 0.25, axis=0),
                         np.quantile(corrs, 0.75, axis=0)
                         ))

                    corrs_vars.append(np.var(corrs, axis=0) / corrs_means[-1])

                    # gather img params after removing redundancy in iterations
                    means_params.append(imgs_params[i])
                    imgs_name = os.path.splitext(os.path.basename(corr_path))[0][:-3]
                    print('Finished {}...'.format(imgs_name))
                    corrs = []


        data = {
            'corrs_means': corrs_means,
            'corrs_quantiles': corrs_quantiles,
            'corrs_vars': corrs_vars,
            'means_params': means_params
        }


        # save correlations
        if not os.path.exists(means_folder):
            os.makedirs(means_folder)

        with open(mean_filename, 'wb') as fp:
            pickle.dump(data, fp)

    return corrs_means, corrs_quantiles, corrs_vars, means_params

# def plot_correlations_grid(filters, img_folder):
#
#
#
#
#
#
def plot_correlations_all(corrs_means, corrs_quantiles, corrs_vars, means_params, method,
                          highlight_keys, filters=None):

    if filters is not None:
        filtered_index = filter_corrs(means_params, filters)

        corrs_means = [corrs_means[i] for i in filtered_index]
        corrs_quantiles = [corrs_quantiles[i] for i in filtered_index]
        corrs_vars = [corrs_vars[i] for i in filtered_index]
        means_params = [means_params[i] for i in filtered_index]

    plt.style.use('seaborn-dark')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['text.usetex'] = True
    alpha = np.tanh(0.05 + 1 / len(corrs_means))  ## transparency related to number of lines

    latex_dict = {
            'total_neurons': 'N',
            'num_layers': 'L',
            'omega': '\\omega',
            'alpha': '\\alpha',
            'mu': '\\mu',
    }

    cmap = plt.get_cmap('gnuplot')
    # cmap = plt.get_cmap('summer')

    fig, ax = plt.subplots(nrows=2, ncols=len(highlight_keys), figsize=(15, 6), dpi=100,
                           sharex=True, sharey='row', constrained_layout=True)

    # if len(highlight_keys) == 1:
    #     ax = [ax]

    suptitle = []
    for k, v, in filters.items():
        if filters[k] is not None:
            if not len(suptitle):
                suptitle.append(latex_dict[k] + '=' + str(filters[k]))
            else:
                suptitle.append(', ' + latex_dict[k] + '=' + str(filters[k]))
    if suptitle:
        suptitle.insert(0, '$')
        suptitle.append('$')
    # suptitle = r"$" + "".join(suptitle) + "$"
    fig.suptitle("".join(suptitle))

    # plt.xlabel('Distance')
    fig.text(0.5, 0.04, 'Distance', ha='center', va='center')

    corrs_q25 = [c[0] for c in corrs_quantiles]
    corrs_q75 = [c[1] for c in corrs_quantiles]

    # TODO: make distances a function of img resolution...
    if method == 'fft' or method == 'fft_windowed':
        # bin the fft auto-corr results since they are very high resolution
        distances = dist_vec((1023, 1023))  ## assumes a 512x512 image.

        #######################################################
        # playing around with how the variances are calculated. comment this for new calculations!!
        # corrs_vars = [corrs_vars[i] * corrs_means[i]
        #               for i in range(len(corrs_vars))]
        #######################################################

        corrs_means, bin_edges, _ = scipy.stats.binned_statistic(
            distances, corrs_means, statistic='mean', bins=300)

        corr_q25, _, _ = scipy.stats.binned_statistic(
            distances, corrs_q25, statistic='mean', bins=300)

        corr_q75, _, _ = scipy.stats.binned_statistic(
            distances, corrs_q75, statistic='mean', bins=300)

        corr_var, _, _ = scipy.stats.binned_statistic(
            distances, corrs_vars, statistic='mean', bins=300)

        d = (bin_edges[:-1] + bin_edges[1:]) / 2
    else:
        d = np.arange(1, 300)


    for ii, highlight in enumerate(highlight_keys):

        # if ii == 1:
        #     cmap = plt.get_cmap('copper')

        values = list(np.sort(np.unique([v[highlight] for v in means_params])))
        norm = colors.Normalize(vmin=0, vmax=len(values))  # age/color mapping

        if ii == 0:
            ax[0, ii].set_ylabel('r')
            ax[1, ii].set_ylabel('$\\sigma^2(r)/\\mu$')
        # fig.set_xlabel('Distance')
        # fig.set_ylabel('r')

        for i, corr in enumerate(corrs_means):

            corr_value = means_params[i][highlight]

            c = cmap(norm( values.index(corr_value) ))

            label = '$' + latex_dict[highlight] + ': ' + str(corr_value) + '$'

            if method == 'sample':
                ax[0, ii].plot(np.arange(0, len(corr)),
                               corr, alpha=alpha, linewidth=1, c=c, label=label)
                ax[1, ii].plot(np.arange(0, len(corr)),
                               corr_var[i, :], alpha=alpha, linewidth=1, c=c, label=label)
            elif method == 'fft' or method == 'fft_windowed':
                # ax[0].plot(d, corr, alpha=alpha, linewidth=1, c=c, label=label)
                ax[0, ii].plot(d, corr, alpha=alpha, linewidth=1, c=c, label=label)
                # ax[1, ii].plot(d, corr_var[i, :], alpha=2*alpha, linewidth=1, c=c, label=label)
                ax[1, ii].scatter(d, corr_var[i, :], alpha=2*alpha, s=1,
                                  c=[c], label=label)
                # ax[0, ii].fill_between(d, corr_q75[i, :], corr_q25[i, :], alpha=0.03, color=c)

            else:
                raise Exception(r'Method must be either "fft", "fft_windowed" or "sample".')



        handles, labels = unique_labels(ax[0, ii])

        leg1 = ax[0, ii].legend(handles, labels)  # draw the legend with the filtered handles and labels lists
        leg2 = ax[0, ii].legend(handles, labels)
        for h in leg1.legendHandles:
            h.set_alpha(1)
        for h in leg2.legendHandles:
            h.set_alpha(1)

        for irow in range(0, 2):
            # xmax, xmin = ax[irow, ii].get_xlim()
            # ymax, ymin = ax[irow, ii].get_ylim()
            # ax[irow, ii].set_aspect(np.max(xmax-xmin)/(ymax-ymin))
            ax[irow, ii].set_axisbelow(True)
            ax[irow, ii].minorticks_on()
            ax[irow, ii].grid(which='major', linestyle='-', color='black', alpha=0.1)
            ax[irow, ii].grid(which='minor', linestyle=':', color='black', alpha=0.1)

        # ax[1, 0].set_yscale("log")
        # ax[1, 1].set_yscale("log")

        # plt.yscale('log')
        # plt.xscale('log')


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

def filter_corrs(means_params, filters):
    # returns index of correlations to plot as filtered according to the values given in 'filters'

    keys_to_remove = [k for k, v in filters.items() if any(vi is None for vi in v)]
    for k in keys_to_remove:
        del filters[k]

    filtered_index = [ i for i in range(len(means_params))
                    if all( any( np.isclose(vi, means_params[i][k]) for vi in v)
                            for k, v in filters.items()) ]

    return np.array(filtered_index)

def dist_vec(shape):
    n = shape[0]
    m = shape[1]
    iu = np.triu_indices(n=n, m=m, k=0)

    distu = np.sqrt((iu[0] - (n-1)/2)**2 + (iu[1] - (m-1)/2)**2)

    return distu

def theta_vec(shape):
    n = shape[0]
    m = shape[1]
    iu = np.triu_indices(n=n, m=m, k=0)


    thetau = np.arctan2((iu[0] - (n-1)/2), (iu[1] - (m-1)/2))

    return thetau



if __name__ == '__main__':

    img_folder = 'big'
    method = 'fft'

    highlight_keys = [
        'total_neurons',
        'num_layers',
        'omega',
        'alpha',
        'mu'
    ]

    corrs_means, corrs_quantiles, corr_vars, means_params = \
        architecture_correlation_means(img_folder, highlight_keys, method, forceCalculate=False)



    ########## ONE FIGURE ##########
    # highlight = 'total_neurons'
    # plot_correlations_all(corrs_means, means_params, highlight)
    # plt.show()
    ################################

    # ## All FIGURES ##
    # for highlight in highlight_keys:
    #     plot_correlations_all(corrs_means, means_params, highlight)
    # plt.show()
    # ################################
    #
    #
    #### ALL FIGURES IN SUBPLOT ####
    highlight = ['omega', 'mu']

    N = [None]  # total_neurons
    L = [None]  # num layers
    omega = [None]  # omega
    alpha = [None]  # alpha
    mu = [None]  # mu

    filters = {
        'total_neurons': N,
        'num_layers': L,
        'omega': omega,
        'alpha': alpha,
        'mu': mu,
    }

    plot_correlations_all(corrs_means, corrs_quantiles, corr_vars, means_params,
                          method=method, highlight_keys=highlight, filters=filters)
    plt.show()
    ################################
    #
    #



