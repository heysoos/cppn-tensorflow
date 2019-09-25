import numpy as np
import os
import sampler
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import colors
from orderedset import OrderedSet
import scipy.stats
from p_tqdm import p_imap
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import seaborn as sns
import matplotlib as mpl
from datetime import datetime

from tqdm import tqdm
import glob

from visualize_natimg_correlations import correlation_means



def architecture_correlation_means(dir, img_folder, filter_keys, method, forceCalculate=False):
    # load jsons, use to sort corr_path list, return corr_paths
    img_dir = os.path.join(dir, img_folder)

    corrs_folder = os.path.join(img_dir, 'corrs', method)
    # jsons_folder = os.path.join(img_dir, 'json')

    # generate savefiles for correlation means
    means_folder = os.path.join(corrs_folder, 'means')
    mean_filename = os.path.join(means_folder, img_folder + '_corr_means.txt')

    # first check if means have already been calculated, if so, load.
    exists = os.path.isfile(mean_filename)
    if exists and not forceCalculate:
        load_str = 'Loading correlation means: ' + mean_filename
        print(load_str)
        with open(mean_filename, 'rb') as fp:
            data = pickle.load(fp)
        corrs_means = [d[0] for d in data]
        corrs_quantiles = [d[1] for d in data]
        corrs_vars = [d[2] for d in data]
        means_params = [d[3] for d in data]
        # corrs_means = data['corrs_means']
        # corrs_quantiles = data['corrs_quantiles']
        # corrs_vars = data['corrs_vars']
        # means_params = data['means_params']
    else:
        calc_str = 'Calculating correlation means...'
        print(calc_str)

        corrs_paths, imgs_params, sort_list = sort_img_dirs_and_params(img_folder,
                                                                       method, filter_keys)

        max_iters = np.max([p[-1] for p in sort_list]) + 1
        arch_corr_paths = [[(corrs_paths[i:i+max_iters], imgs_params[i])]
                           for i in range(0, len(corrs_paths), max_iters)]

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

        elif method == 'fft' or method == 'fft_windowed' or method == 'fft_norm':
            #  should load files one by one since they are much larger

            # corrs = []
            # corrs_means = []
            # corrs_quantiles = []
            # corrs_vars = []
            # means_params = []

            num_cpus = 11
            f = lambda x: pickle_arch_corrs(*x)
            data = list(p_imap(f, arch_corr_paths, num_cpus=num_cpus))

        #     for i, corr_path in enumerate(corrs_paths):
        #         with open(corr_path, 'rb') as fp:
        #             corrs.append(pickle.load(fp))
        #
        #         if len(corrs) == max_iters:
        #             # calculate correlation means
        #             corrs = np.nan_to_num(corrs)  # set nan correlations to 0
        #
        #             corrs_means.append(np.mean(corrs, axis=0))
        #
        #             corrs_quantiles.append(
        #                 (np.quantile(corrs, 0.25, axis=0),
        #                  np.quantile(corrs, 0.75, axis=0)
        #                  ))
        #
        #             corrs_vars.append(np.var(corrs, axis=0))
        #
        #             # gather img params after removing redundancy in iterations
        #             means_params.append(imgs_params[i])
        #             imgs_name = os.path.splitext(os.path.basename(corr_path))[0][:-3]
        #             print('Finished {}...'.format(imgs_name))
        #             corrs = []
        #
        #
        # data = {
        #     'corrs_means': corrs_means,
        #     'corrs_quantiles': corrs_quantiles,
        #     'corrs_vars': corrs_vars,
        #     'means_params': means_params
        # }


        # save correlations
        if not os.path.exists(means_folder):
            os.makedirs(means_folder)

        with open(mean_filename, 'wb') as fp:
            pickle.dump(data, fp)

        corrs_means = [d[0] for d in data]
        corrs_quantiles = [d[1] for d in data]
        corrs_vars = [d[2] for d in data]
        means_params = [d[3] for d in data]

    return corrs_means, corrs_quantiles, corrs_vars, means_params

def pickle_arch_corrs(arch_corr_paths, means_params):
    corrs = []
    for i, corr_path in enumerate(arch_corr_paths):
        with open(corr_path, 'rb') as fp:
            corrs.append(pickle.load(fp))

    corrs = np.nan_to_num(corrs)  # set nan correlations to 0

    corrs_means = np.mean(corrs, axis=0)

    corrs_quantiles = (np.quantile(corrs, 0.25, axis=0), np.quantile(corrs, 0.75, axis=0))

    corrs_vars = np.var(corrs, axis=0)


    return (corrs_means, corrs_quantiles, corrs_vars, means_params)


def global_cppn_stats(dir, img_folder, method, forceCalculate=False):
    img_dir = os.path.join(dir, img_folder)

    corrs_folder = os.path.join(img_dir, 'corrs', method)
    # generate savefiles for correlation means
    stats_folder = os.path.join(corrs_folder, 'stats')
    stats_filename = os.path.join(stats_folder, img_folder + '_corr_stats.txt')

    # first check if means have already been calculated, if so, load.
    exists = os.path.isfile(stats_filename)
    if exists and not forceCalculate:
        load_str = 'Loading correlation means: ' + stats_filename
        print(load_str)
        with open(stats_filename, 'rb') as fp:
            data = pickle.load(fp)
        c_mean = data['corrs_mean']
        c_var = data['corrs_var']
    else:
        calc_str = 'Calculating correlation means...'
        print(calc_str)

        corrs_paths, _ = load_img_paths_and_params(dir, img_folder, method)
        distances = dist_vec((1023, 1023))

        c_sum = np.zeros(np.shape(distances))
        for corr_path in tqdm(corrs_paths):
            with open(corr_path, 'rb') as fp:
                c_sum += pickle.load(fp)
        c_mean = c_sum / len(corrs_paths)

        c_var_sum = np.zeros(np.shape(distances))
        for corr_path in tqdm(corrs_paths):
            with open(corr_path, 'rb') as fp:
                c_var_sum += (pickle.load(fp) - c_mean) ** 2
        c_var = c_var_sum / len(corrs_paths)

        data = {'corrs_mean': c_mean,
                'corrs_var': c_var}

        if not os.path.exists(stats_folder):
            os.makedirs(stats_folder)

        with open(stats_filename, 'wb') as fp:
            pickle.dump(data, fp)

    return c_mean, c_var


def corr_distance_stat(corrs_paths, d):

    corr_d = []
    for i, corr_path in enumerate(corrs_paths):
        with open(corr_path, 'rb') as fp:
            corr_d.append(pickle.load(fp)[d])

    corr_d_mean = np.mean(corr_d)
    corr_d_var = np.var(corr_d)

    return (corr_d_mean, corr_d_var)

def load_img_paths_and_params(dir, img_folder, method):
    img_dir = os.path.join(dir, img_folder)

    corrs_folder = os.path.join(img_dir, 'corrs', method)
    jsons_folder = os.path.join(img_dir, 'json')

    corrs_paths = [os.path.join(corrs_folder, f) for f in os.listdir(corrs_folder)
                   if f.endswith('.txt') ]

    imgs_params = []
    for corr in corrs_paths:
        img_name = os.path.splitext(os.path.basename(corr))[0]
        json_path = os.path.join(jsons_folder, img_name + '.json')

        imgs_params.append(sampler.loadJSON(json_path))

    return corrs_paths, imgs_params

def sort_img_dirs_and_params(img_folder, method, filter_keys):

    dir = 'save/img_architectures'
    corrs_paths, imgs_params = load_img_paths_and_params(dir, img_folder, method)

    # sort so similar params are together, with iteration number as well
    sort_keys = filter_keys[:]
    sort_keys.append('iteration')

    sort_list = [tuple((img_params[k]) for k in sort_keys) for img_params in imgs_params]
    sort_indices = sorted(enumerate(sort_list), key=itemgetter(1))
    sort_indices = [i[0] for i in sort_indices]

    # if you want to sort the indices with respect to different img_params then use:
    # sorted(enumerate(sort_list), key=lambda x: enum_sort_order(x, [1, 0, 2, 3, 4, 5]))

    # here p[-1] should correspond to the 'iteration' parameter which designates the
    # img number for the same seed. The idea is to sort, and then average, over imgs
    # generated from the same seed.
    # max_iters = np.max([p[-1] for p in sort_list]) + 1

    corrs_paths = [corrs_paths[i] for i in sort_indices]
    imgs_params = [imgs_params[i] for i in sort_indices]

    return corrs_paths, imgs_params, sort_list

def enum_sort_order(x, order):
    return tuple(x[1][i] for i in order)

def aesthetic_correlations(dir, img_folder, method, takeMean=False, forceCalculate=False):

    img_dir = os.path.join(dir, img_folder)

    corrs_folder = os.path.join(img_dir, 'corrs', method)
    aesthetics_folder = os.path.join(corrs_folder, 'aesthetic')
    aesthetic_filename = os.path.join(aesthetics_folder, img_folder + '_corr_aesthetics.txt')

    # first check if already been calculated, if so, load.
    exists = os.path.isfile(aesthetic_filename)
    if exists and not forceCalculate:
        with open(aesthetic_filename, 'rb') as fp:
            data = pickle.load(fp)
            aesthetic_corrs = data['aesthetic_corrs']
            aesthetic_corrs_vars = data['aesthetic_corrs_vars']
            aesthetic_params = data['aesthetic_params']
        load_str = 'Loading aesthetic correlation: ' + aesthetic_filename
        print(load_str)
    else:
        print('Gathering aesthetic correlations...')
        corrs_paths, aesthetic_params = load_img_paths_and_params(dir, img_folder, method)

        img_names = [os.path.splitext(os.path.basename(dir))[0] for dir in corrs_paths]
        aesthetic_img_names = [os.path.splitext(os.path.basename(dir))[0]
                               for dir in aesthetic_dirs[:]]
        aesthetic_index = set([img_names.index(a) for a in aesthetic_img_names])

        corrs_paths = [corrs_paths[i] for i in aesthetic_index]
        aesthetic_params = [aesthetic_params[i] for i in aesthetic_index]

        aesthetic_corrs = []
        for i, corr_path in enumerate(corrs_paths):
            with open(corr_path, 'rb') as fp:
                aesthetic_corrs.append(pickle.load(fp))

        aesthetic_corrs_vars = np.var(aesthetic_corrs, axis=0)
        if takeMean:
            aesthetic_corrs = np.mean(aesthetic_corrs, axis=0)

            data = {
                'aesthetic_corrs': aesthetic_corrs,
                'aesthetic_corrs_vars': aesthetic_corrs_vars,
                'aesthetic_params': aesthetic_params
            }
        else:
            data = {
                'aesthetic_corrs': aesthetic_corrs,
                'aesthetic_corrs_vars': aesthetic_corrs_vars,
                'aesthetic_params': aesthetic_params
            }

    # save correlations
    if not os.path.exists(aesthetics_folder):
        os.makedirs(aesthetics_folder)

    with open(aesthetic_filename, 'wb') as fp:
        pickle.dump(data, fp)

    return aesthetic_corrs, aesthetic_corrs_vars, aesthetic_params

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

    corrs_q25 = [c[0] for c in corrs_quantiles]
    corrs_q75 = [c[1] for c in corrs_quantiles]

    # TODO: make distances a function of img resolution...
    if method == 'fft' or method == 'fft_windowed' or method == 'fft_norm':
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

    fig, ax = plt.subplots(nrows=2, ncols=len(highlight_keys), figsize=(15.5, 7), dpi=100,
                           sharex=True, sharey='row', constrained_layout=True)

    fig.suptitle("".join(suptitle))
    # plt.xlabel('Distance')
    fig.text(0.5, 0.04, 'Distance', ha='center', va='center')

    for ii, highlight in enumerate(highlight_keys):

        # if ii == 1:
        #     cmap = plt.get_cmap('copper')

        values = list(np.sort(np.unique([v[highlight] for v in means_params])))
        # all_values = [tuple(v[h] for h in highlight_keys) for v in means_params]
        # values = sorted(np.unique(all_values, axis=0), key=itemgetter(0))
        norm = colors.Normalize(vmin=0, vmax=len(values))  # age/color mapping

        if ii == 0:
            ax[0, ii].set_ylabel(r'$\rho(x,x_d)$')
            ax[1, ii].set_ylabel(r'$\sigma^2_{\rho}$')
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
            elif method == 'fft' or method == 'fft_windowed' or method == 'fft_norm':
                # ax[0].plot(d, corr, alpha=alpha, linewidth=1, c=c, label=label)
                ax[0, ii].plot(d, corr, alpha=alpha, linewidth=1, c=c, label=label)
                ax[1, ii].plot(d, corr_var[i, :], alpha=alpha, linewidth=1, c=c, label=label)
                # ax[1, ii].scatter(d, corr_var[i, :], alpha=alpha, s=1,
                #                   c=[c], label=label)
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

        # ax[1, ii].set_yscale("log")
        # ax[1, 1].set_yscale("log")

    for i in range(np.shape(ax)[1]):
        ax[1, i].set_yscale('log')
        ax[1, i].set_ylim(1e-7, 0.1)

    for a in ax.flatten():
        a.set_xscale('log')
        # a.set_xlim([-5, 512])

        # plt.yscale('log')
        # plt.xscale('log')
    print('Done!')

def plot_correlations_cppn_aesthetic_together(
        corrs_means, corrs_quantiles, corrs_vars, means_params,
        aesthetic_corrs, aesthetic_corrs_vars, aesthetic_params,
        method, highlight_keys, filters=None):

    if filters is not None:
        filtered_index = filter_corrs(means_params, filters)

        corrs_means = [corrs_means[i] for i in filtered_index]
        corrs_quantiles = [corrs_quantiles[i] for i in filtered_index]
        corrs_vars = [corrs_vars[i] * corrs_means[i] for i in filtered_index]
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

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15.5, 8), dpi=100,
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
    if method == 'fft' or method == 'fft_windowed' or method == 'fft_norm':
        # bin the fft auto-corr results since they are very high resolution
        # distances = dist_vec((1023, 1023))  ## assumes a 512x512 image.
        distances = theta_vec((1023, 1023))  ## assumes a 512x512 image.

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

        aesthetic_corrs = np.mean(aesthetic_corrs, axis=0)

        aesthetic_corr, _, _ = scipy.stats.binned_statistic(
            distances, aesthetic_corrs, statistic='mean', bins=300)
        aesthetic_var, _, _ = scipy.stats.binned_statistic(
            distances, aesthetic_corrs_vars, statistic='mean', bins=300)



        d = (bin_edges[:-1] + bin_edges[1:]) / 2
    else:
        aesthetic_corr = np.mean(aesthetic_corrs, axis=0)
        aesthetic_var = aesthetic_corrs_vars

        d = np.arange(1, 300)

    # plot everything in one figure and make a legend for all combinations of params
    all_values = [tuple(v[h] for h in highlight_keys) for v in means_params]
    values = sorted(np.unique(all_values, axis=0), key=itemgetter(0))
    values = [tuple(v) for v in values[:]]
    norm = colors.Normalize(vmin=0, vmax=len(values))  # age/color mapping

    ax[0].set_ylabel(r'$\rho(x,x_d)$')
    ax[1].set_ylabel(r'$\sigma^2_{\rho}$')
    # fig.set_xlabel('Distance')
    # fig.set_ylabel('r')

    # PLOT ALL CORRELATIONS
    for i, corr in enumerate(corrs_means):

        corr_value = means_params[i]
        corr_value_index = values.index(tuple(corr_value[h] for h in highlight))

        c = cmap(norm(corr_value_index))

        ktxt = []
        vtxt = []
        for ih, h in enumerate(highlight):
            if ih > 0:
                ktxt.append(', ')
                vtxt.append(', ')
            ktxt.append(latex_dict[h])
            vtxt.append(str(corr_value[h]))

        ktxt = '$(' + ''.join(ktxt) + ')= '
        vtxt = '(' + ''.join(vtxt) + ')$'

        label = ktxt + vtxt

        if method == 'sample':
            ax[0].plot(np.arange(0, len(corr)),
                           corr, alpha=alpha, linewidth=1, c=c, label=label)
            ax[1].plot(np.arange(0, len(corr)),
                           corr_vars[i], alpha=alpha, linewidth=1, c=c, label=label)
        elif method == 'fft' or method == 'fft_windowed' or method == 'fft_norm':
            # ax[0].plot(d, corr, alpha=alpha, linewidth=1, c=c, label=label)
            # ax[0].plot(d, corr, alpha=alpha, linewidth=1, c=c, label=label)
            ax[0].scatter(d, corr, alpha=alpha, s=1, c=[c], label=label)
            ax[1].plot(d, corr_var[i, :], alpha=alpha, linewidth=1, c=c, label=label)
            # ax[1].scatter(d, corr_var[i, :], alpha=alpha, s=1,
            #                   c=[c], label=label)
            # ax[0].fill_between(d, corr_q75[i, :], corr_q25[i, :], alpha=0.03, color=c)

        else:
            raise Exception(r'Method must be either "fft", "fft_windowed" or "sample".')

    # PLOT AESTHETIC CORRELATIONS
    # for corr in aesthetic_corr:
    #     ax[0].plot(d, corr, alpha=alpha, linestyle='--', linewidth=1, c='r', label='Aesthetic')
    # ax[1].plot(d, aesthetic_var, alpha=1, linestyle='--', linewidth=3, c='k', label='Aesthetic')

    ax[0].plot(d, aesthetic_corr, linestyle='--', linewidth=3, c='k', label='Aesthetic')
    ax[1].plot(d, aesthetic_var, linestyle='--', linewidth=3, c='k', label='Aesthetic')

    handles, labels = unique_labels(ax[0])

    leg1 = ax[0].legend(handles, labels)  # draw the legend with the filtered handles and labels lists
    leg2 = ax[0].legend(handles, labels)
    for h in leg1.legendHandles:
        h.set_alpha(1)
    for h in leg2.legendHandles:
        h.set_alpha(1)

    for irow in range(0, 2):
        # xmax, xmin = ax[irow].get_xlim()
        # ymax, ymin = ax[irow].get_ylim()
        # ax[irow].set_aspect(np.max(xmax-xmin)/(ymax-ymin))
        ax[irow].set_axisbelow(True)
        ax[irow].minorticks_on()
        ax[irow].grid(which='major', linestyle='-', color='black', alpha=0.1)
        ax[irow].grid(which='minor', linestyle=':', color='black', alpha=0.1)

    # ax[0].set_xscale("log")
    # ax[0].set_yscale("log")
    #
    # ax[1].set_xscale("log")
    # ax[1].set_yscale("log")

    # plt.yscale('log')
    # plt.xscale('log')
    print('Done!')

def plot_correlations_all_together(data_cppn, data_aesthetic, data_nat):


    # plt.style.use('seaborn-dark')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['text.usetex'] = True
    alpha = np.tanh(0.05 + 1 / len(data_cppn['corrs_means']))  ## transparency related to number of lines

    if method == 'fft' or method == 'fft_windowed' or method == 'fft_norm':
        # bin the fft auto-corr results since they are very high resolution
        distances = dist_vec((1023, 1023))  ## assumes a 512x512 image.
        # distances = theta_vec((1023, 1023))

        #######################################################
        # playing around with how the variances are calculated. comment this for new calculations!!
        # corrs_vars = [corrs_vars[i] * corrs_means[i]
        #               for i in range(len(corrs_vars))]
        #######################################################

        corrs_means, bin_edges, _ = scipy.stats.binned_statistic(
            distances, data_cppn['corrs_means'], statistic='mean', bins=300)

        corr_var, _, _ = scipy.stats.binned_statistic(
            distances, data_cppn['corr_vars'], statistic='mean', bins=300)

        ################# AESTHETIC #################
        mac = np.mean(aesthetic_corrs, axis=0)

        mac, _, _ = scipy.stats.binned_statistic(
            distances, mac, statistic='mean', bins=300)

        aesthetic_corr, _, _ = scipy.stats.binned_statistic(
            distances, data_aesthetic['corrs'], statistic='mean', bins=300)

        aesthetic_var, _, _ = scipy.stats.binned_statistic(
            distances, data_aesthetic['var'], statistic='mean', bins=300)

        ################# NATURAL IMGS #################
        nat_corr = []
        nat_var = []

        for i in range(len(data_nat['corrs_means'])):
            nc, _, _ = scipy.stats.binned_statistic(
                distances, data_nat['corrs_means'][i], statistic='mean', bins=300)

            nv, _, _ = scipy.stats.binned_statistic(
                distances, data_nat['corr_var'][i], statistic='mean', bins=300)

            nat_corr.append(nc[0])
            nat_var.append(nv[0])



        d = (bin_edges[:-1] + bin_edges[1:]) / 2
    else:
        aesthetic_corr = np.mean(aesthetic_corrs, axis=0)
        aesthetic_var = aesthetic_corrs_vars

        d = np.arange(1, 300)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(17, 10), dpi=100,
                           sharex=True, sharey='row', constrained_layout=True)

    fig.text(0.5, 0.04, 'Distance', ha='center', va='center')

    ax[0].set_ylabel(r'$\rho(x,x_d)$')
    ax[1].set_ylabel(r'$\sigma^2_{\rho}$')
    # fig.set_xlabel('Distance')
    # fig.set_ylabel('r')

    # PLOT ALL CORRELATIONS
    for i, corr in enumerate(corrs_means):

        corr_value = means_params[i]
        # corr_value_index = values.index(tuple(corr_value[h] for h in highlight))

        # c = cmap(norm(corr_value_index))
        c = 'k'

        label = 'CPPN'

        if method == 'sample':
            ax[0].plot(np.arange(0, len(corr)),
                           corr, alpha=alpha, linewidth=1, c=c, label=label)
            ax[1].plot(np.arange(0, len(corr)),
                           corr_vars[i], alpha=alpha, linewidth=1, c=c, label=label)
        elif method == 'fft' or method == 'fft_windowed' or method == 'fft_norm':
            # ax[0].plot(d, corr, alpha=alpha, linewidth=1, c=c, label=label)
            ax[0].plot(d, corr, alpha=alpha, linewidth=1, c=c, label=label)
            # ax[0].scatter(d, corr, alpha=alpha, s=1, c=[c], label=label)
            # ax[1].plot(d, corr_var[i, :], alpha=alpha, linewidth=1, c=c, label=label)
            ax[1].scatter(d, corr_var[i, :], alpha=alpha, s=1,
                              c=[c], label=label)
            # ax[0].fill_between(d, corr_q75[i, :], corr_q25[i, :], alpha=0.03, color=c)

        else:
            raise Exception(r'Method must be either "fft", "fft_windowed" or "sample".')

    # PLOT AESTHETIC CORRELATIONS
    cmap = plt.get_cmap('Reds')
    norm = colors.Normalize(vmin=0, vmax=4)  # age/color mapping
    c = cmap(norm(2))
    for corr in aesthetic_corr:
        ax[0].plot(d, corr, alpha=2*alpha, linestyle='--', linewidth=0.75, c=c, label='Aesthetic')
    ax[1].plot(d, aesthetic_var, alpha=1, linestyle='--', linewidth=3, c=c, label='Aesthetic')

    cmap = plt.get_cmap('gnuplot')
    norm = colors.Normalize(vmin=0, vmax=len(nat_corr))  # age/color mapping
    for i in range(len(nat_corr)):
        c = cmap(norm(i))
        ax[0].plot(d, nat_corr[i], alpha=1, linestyle=':', linewidth=3, c=c,
                   label=data_nat['img_folders'][i])
        ax[1].plot(d, nat_var[i], alpha=1, linestyle=':', linewidth=3, c=c,
                   label=data_nat['img_folders'][i])

    ax[0].plot(d, mac, linestyle='--', linewidth=3, c='k', label='Mean Aesthetic')
    # ax[1].plot(d, aesthetic_var, linestyle='--', linewidth=3, c='k', label='Aesthetic')
    mccppn = np.mean(corrs_means, axis=0)
    mvcppn = np.mean(corr_var, axis=0)
    ax[0].plot(d, mccppn, linewidth=3, c='g', label='Mean CPPN')
    ax[1].plot(d, mvcppn, '.', linewidth=3, c='g', label='Mean CPPN')

    handles, labels = unique_labels(ax[0])

    leg1 = ax[0].legend(handles, labels)  # draw the legend with the filtered handles and labels lists
    leg2 = ax[0].legend(handles, labels)
    for h in leg1.legendHandles:
        h.set_alpha(1)
    for h in leg2.legendHandles:
        h.set_alpha(1)

    for irow in range(0, 2):
        # xmax, xmin = ax[irow].get_xlim()
        # ymax, ymin = ax[irow].get_ylim()
        # ax[irow].set_aspect(np.max(xmax-xmin)/(ymax-ymin))
        ax[irow].set_axisbelow(True)
        ax[irow].minorticks_on()
        ax[irow].grid(which='major', linestyle='-', color='black', alpha=0.1)
        ax[irow].grid(which='minor', linestyle=':', color='black', alpha=0.1)

    # ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    #
    # ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    plt.xlim([-10, 512])
    print('Done!')
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
    # e.g. if 'total_neurons': [5, 10] then only correlations corresponding to these values will
    # kept in the index. If a filter value is 'None' then all cases are indexed (no filtering).

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

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


if __name__ == '__main__':

    dir = 'save/img_architectures'
    img_folder = 'big'
    method = 'fft_norm'

    # global_cppn_stats(dir, img_folder, method, forceCalculate=False)

    ## DON'T CHANGE ##
    highlight_keys = [
        'total_neurons',
        'num_layers',
        'omega',
        'alpha',
        'mu'
    ]
    ##
    # with open('save/img_architectures/big/aesthetic_tag.txt', 'r') as f:
    #     aesthetic_dirs = f.read().splitlines()
    # takeMean = False
    # aesthetic_corrs, aesthetic_corrs_vars, aesthetic_params = \
    #     aesthetic_correlations(dir, img_folder, method, takeMean=takeMean, forceCalculate=False)
    #
    # data_aesthetic = {
    #     'corrs': aesthetic_corrs,
    #     'var': aesthetic_corrs_vars,
    #     'params': aesthetic_params
    # }


    corrs_means, corrs_quantiles, corr_vars, means_params = \
        architecture_correlation_means(dir, img_folder, highlight_keys,
                                       method, forceCalculate=False)
    data_cppn = {
        'corrs_means': corrs_means,
        'corr_vars': corr_vars,
    }



    ########## ONE FIGURE ##########
    highlight = 'total_neurons'
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
    highlight = ['total_neurons']

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

    # plot_correlations_all(corrs_means, corrs_quantiles, corr_vars, means_params,
    #                       method=method, highlight_keys=highlight, filters=filters)


    # plot_correlations_cppn_aesthetic_together(corrs_means, corrs_quantiles, corr_vars, means_params,
    #                                aesthetic_corrs, aesthetic_corrs_vars, aesthetic_params,
    #                                method=method, highlight_keys=highlight, filters=filters)

    # ### PLOT WITH NATURAL IMAGES ###
    # dir = 'natural_imgs/'
    # img_folders = ['Foliage', 'LandWater', 'Snow', 'Animals', 'Flowers', 'ManMade']
    # # img_folders = ['Animals', 'Flowers', 'ManMade']
    # method = 'fft_norm'  # 'fft_norm', 'sample'
    #
    #
    # corrs_means, _, corr_vars = \
    #     correlation_means(dir, img_folders, method, forceCalculate=False)
    #
    # data_nat = {
    #     'corrs_means': corrs_means,
    #     'corr_var': corr_vars,
    #     'img_folders': img_folders
    # }
    #
    #
    # plot_correlations_all_together(data_cppn, data_aesthetic, data_nat)

    plt.show()
    ################################
    #
    #



