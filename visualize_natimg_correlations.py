import numpy as np
import os
import sampler
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import colors
from orderedset import OrderedSet
import scipy.stats
from p_tqdm import p_map

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

def load_img_paths(dir, img_folder, method):
    img_dir = os.path.join(dir, img_folder)

    corrs_folder = os.path.join(img_dir, 'corrs', method)

    corrs_paths = [os.path.join(corrs_folder, f) for f in os.listdir(corrs_folder)
                   if f.endswith('.txt')]

    return corrs_paths

def load_corr(corr_paths):
    # used only in the parallel computing case
    for i, corr_path in enumerate(corr_paths):
        with open(corr_path, 'rb') as fp:
            if i==0:
                m_corr = (pickle.load(fp)) / len(corr_paths)
            else:
                m_corr += (pickle.load(fp)) / len(corr_paths)
    return m_corr

def correlation_means(dir, img_folders, method, forceCalculate=False):
    # calculates the mean correlations for all images in the img_folders into a list

    corrs_means = []
    corrs_quantiles = []
    corrs_vars = []

    for img_folder in img_folders:
        img_dir = os.path.join(dir, img_folder)
        corrs_folder = os.path.join(img_dir, 'corrs', method)

        means_folder = os.path.join(corrs_folder, 'means')
        mean_filename = os.path.join(means_folder, img_folder + '_corr_means.txt')

        exists = os.path.isfile(mean_filename)
        if exists and not forceCalculate:
            load_str = 'Loading correlation means: ' + mean_filename
            print(load_str)
            with open(mean_filename, 'rb') as fp:
                data = pickle.load(fp)
            corrs_means.append(data['corrs_means'])
            corrs_quantiles.append(data['corrs_quantiles'])
            corrs_vars.append(data['corrs_vars'])
        else:
            calc_str = 'Calculating correlation means...'
            print(calc_str)

            corrs_paths = load_img_paths(dir, img_folder, method)

            if method == 'fft_norm':
                #  should load files one by one since they are much larger

                folder_corrs_means = []
                folder_corrs_quantiles = []
                folder_corrs_vars = []
                corrs = []

                # if there are too many correlations, parallel compute
                if len(corrs_paths) > 5*1e3:
                    sub_size = 100  # number of corr paths to deal with per core
                    corrs_paths = corrs_paths[:-(len(corrs_paths) % sub_size)]  # truncate list
                    sub_corrs_paths = [ [corrs_paths[i] for i in range(start, start + sub_size)]
                                        for start in range(0, len(corrs_paths), sub_size) ]
                    num_cpus = 11
                    f = lambda x: load_corr(x)
                    corrs = p_map(f, sub_corrs_paths, num_cpus=num_cpus)

                else:
                    for i, corr_path in enumerate(corrs_paths):
                        with open(corr_path, 'rb') as fp:
                            corrs.append(pickle.load(fp))

                        imgs_name = os.path.splitext(os.path.basename(corr_path))[0][:-3]
                        print('Finished {}...'.format(imgs_name))
            elif method == 'sample':

                folder_corrs_means = []
                folder_corrs_quantiles = []
                folder_corrs_vars = []
                corrs = []

                # if there are too many correlations, parallel compute
                if len(corrs_paths) > 5*1e3:
                    sub_size = 100  # number of corr paths to deal with per core
                    corrs_paths = corrs_paths[:-(len(corrs_paths) % sub_size)]  # truncate list
                    sub_corrs_paths = [ [corrs_paths[i] for i in range(start, start + sub_size)]
                                        for start in range(0, len(corrs_paths), sub_size) ]
                    num_cpus = 11
                    f = lambda x: load_corr(x)
                    corrs = p_map(f, sub_corrs_paths, num_cpus=num_cpus)

                else:
                    for i, corr_path in enumerate(corrs_paths):
                        with open(corr_path, 'rb') as fp:
                            corrs.append(pickle.load(fp))

                        imgs_name = os.path.splitext(os.path.basename(corr_path))[0][:-3]
                        print('Finished {}...'.format(imgs_name))

            else:
                print("Incorrect method defined!")

            # calculate correlation means
            corrs = np.nan_to_num(corrs)  # set nan correlations to 0

            folder_corrs_means.append(np.mean(corrs, axis=0))

            folder_corrs_quantiles.append(
                (np.quantile(corrs, 0.25, axis=0),
                 np.quantile(corrs, 0.75, axis=0)
                 ))

            folder_corrs_vars.append(np.var(corrs, axis=0))

            data = {
                'corrs_means': folder_corrs_means,
                'corrs_quantiles': folder_corrs_quantiles,
                'corrs_vars': folder_corrs_vars,
            }

            # save correlations
            if not os.path.exists(means_folder):
                os.makedirs(means_folder)

            with open(mean_filename, 'wb') as fp:
                pickle.dump(data, fp)

            corrs_means.append(data['corrs_means'])
            corrs_quantiles.append(data['corrs_quantiles'])
            corrs_vars.append(data['corrs_vars'])


    return corrs_means, corrs_quantiles, corrs_vars

if __name__ == '__main__':

    dir = 'natural_imgs/'
    img_folders = ['Foliage', 'LandWater', 'Snow', 'Animals', 'Flowers', 'ManMade']
    # img_folders = ['Animals', 'Flowers', 'ManMade']
    method = 'fft_norm'  # 'fft_norm', 'sample'


    corrs_means, corrs_quantiles, corr_vars = \
        correlation_means(dir, img_folders, method, forceCalculate=False)


    # plot_correlations(corrs_means, corrs_quantiles, corr_vars, method=method)

    plt.show()
