import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import scipy.signal

from p_tqdm import p_imap

def fft_autocorr_norm(img, mass, I, clean=True):
    ###
    # calculate the auto-correlation of 'img' normalizing with respect to local means and variances
    # I: np.ones(np.shape(img))
    # mass: convolve(I, I)
    # these matrices are not included in this function to save computation
    ###
    local_mean = scipy.signal.fftconvolve(img, I) / mass
    sum_img_sqr = scipy.signal.fftconvolve(img ** 2, I)

    if clean:
        local_mean[local_mean < 0] = 0
        sum_img_sqr[sum_img_sqr < 0] = 0

    local_var = (sum_img_sqr - mass * local_mean ** 2) / mass

    if clean:
        local_var[local_var < 1e-8] = 1e-8
        local_var[np.isinf(local_var)] = np.nan


    G = scipy.signal.fftconvolve(img, img[::-1, ::-1])
    means_terms = mass * -1 * (local_mean * local_mean[::-1, ::-1])
    auto_corr = (G + means_terms) / (mass * np.sqrt(local_var * local_var[::-1, ::-1]))


    shape = np.shape(auto_corr)
    iu = np.triu_indices(n=shape[0], m=shape[1])

    return auto_corr[iu]

def load_image_directories(img_folder=None):
    # make a list of all imgs in image directory

    dir = 'natural_imgs'
    imgs_folder = os.path.join(dir, img_folder)
    if img_folder == 'Art':
        imgs_paths = [os.path.join(imgs_folder, f)
                      for f in os.listdir(imgs_folder)
                      if f.endswith('.jpg')]
    else:
        imgs_paths = [os.path.join(imgs_folder, f)
                      for f in os.listdir(imgs_folder)
                      if f.endswith('.TIF')]

    return imgs_paths

def show_image(img_dir):
    img = Image.open(img_dir)
    plt.imshow(img)
    plt.show()

def dist_vec(shape):
    n = shape[0]
    m = shape[1]
    iu = np.triu_indices(n=n, m=m, k=0)

    distu = np.sqrt((iu[0] - (n-1)/2)**2 + (iu[1] - (m-1)/2)**2)

    return distu

def top_rec(n, m):
    n = int(n/2)
    iu = [(y, x) for x in range(m) for y in range(n)]
    return [np.array([iu[i][j] for i in range(len(iu))]) for j in range(2)]

def top_dist_vec(shape):
    n = shape[0]
    m = shape[1]
    iu = top_rec(n, m)

    return np.sqrt((iu[0] - (n - 1) / 2) ** 2 + (iu[1] - (m - 1) / 2) ** 2)

def random_walk(x1, y1, d, res):
    # use random walk to find a second coordinate (x2, y2) from starting point (x1, y1)
    alpha = 2 * np.pi * np.random.random(len(x1))

    dx = np.round(d * np.sin(alpha))
    dy = np.round(d * np.cos(alpha))

    x2 = x1 + dx
    y2 = y1 + dy

    # find all values that are out of bounds (oob)
    oob = (x2 > res - 1) + (x2 < 0) + (y2 > res - 1) + (y2 < 0)

    # TODO: there has to be a better way to do this, no? This seems dangerous using a while loop...
    while any(oob):
        alpha = 2 * np.pi * np.random.random(np.sum(oob))

        dx = np.round(d * np.sin(alpha))
        dy = np.round(d * np.cos(alpha))

        x2[oob] = x1[oob] + dx
        y2[oob] = y1[oob] + dy

        oob = (x2 > res - 1) + (x2 < 0) + (y2 > res - 1) + (y2 < 0)

    return x2, y2

def find_N_pairs(d, N, res):
    # find N pairs of coordinates that are distance d apart
    x1 = np.random.randint(0, res, N)
    y1 = np.random.randint(0, res, N)

    p = x1 * res + y1

    x2, y2 = random_walk(x1, y1, d, res)

    q = x2 * res + y2

    return p.astype('int'), q.astype('int')

def correlation_distance(img, dmin=1, dmax=512, N=10e5):
    # Calculate correlation function (correlation as a function of distance).

    # Summary: for each distance bin, find N pairs of points in the image with that distance. Calculate correlation
    # coefficient for the vectors generated by these pairs. Loop through all distance bins to get correlations as a
    # function of distance.

    # --- PARAMETERS ---
    # imgs_dir: directory of images
    # dmin, dmax: minimum and maximum distances
    # N: number of point pairs samples
    # ------------------

    img = np.array(img).flatten()  # load img > convert to greyscale > flatten
    res = np.sqrt(img.shape[0])  # img should be square...

    # calculate correlation for each distance bin
    corr_d_img = []
    for d in np.arange(dmin, dmax):

        p, q = find_N_pairs(d, N, int(res))
        corr = np.corrcoef(img[p], img[q])[0, 1]

        corr_d_img.append(corr)

    return corr_d_img



def calc_and_save_corrs(method, img_path, corr_folder, dmin=1, dmax=512, N=1e4, takeLog=False):

    # try:
    img = Image.open(img_path).convert('L')
    width, height = img.size

    if width >= 512 and height >= 512:

        minwidth, minheight = 512, 512  # acts as both max/min since we eventually crop
        if width > minwidth or height > minheight:
            ratio = max(minwidth / img.width, minheight / img.height)
        # Resize image so smallest dimension is 512
        img.thumbnail((width * ratio, height * ratio), Image.ANTIALIAS)

        # Crop image down to 512x512
        width, height = img.size
        left = (width - minwidth) / 2
        top = (height - minheight) / 2
        right = (width + minwidth) / 2
        bottom = (height + minheight) / 2

        # Crop the center of the image
        img = img.crop((left, top, right, bottom))

        if method == 'fft_norm':

            #  TODO: logarithms behave weirdly....don't normalize properly and stuff
            if takeLog:
                img = np.log10(np.array(img) + 1)
            else:
                img = np.array(img) / 255

            I = np.ones(np.shape(img))
            mass = np.round(scipy.signal.fftconvolve(I, I))
            corr = fft_autocorr_norm(img, mass, I, clean=True)


        elif method == 'sample':
            corr = correlation_distance(img, dmin, dmax, N)
        else:
            raise Exception('Method must be either \'fft\' or \'sample\'.')

    else:
        print(f'\nImage {img_path} too small.\n')

    # save
    new_base = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
    savename = os.path.join(corr_folder, new_base)

    with open(savename, 'wb') as fp:
        pickle.dump(corr, fp)

    # except:
    #     print('Something went wrong!!')
    #     # print(f'\nFile {img_path} could not open.\n')

    # status_txt = 'Saving correlation: ' + new_base
    # print(status_txt)

def generate_params(method, img_folder, corr_folder, dmin=1, dmax=512, N=1e4, takeLog=False):

    img_paths = load_image_directories(img_folder)

    params = [ (method, img_path, corr_folder, dmin, dmax, N, takeLog) for img_path in img_paths ]

    return params


if __name__ == '__main__':

    method = 'sample'  # (DEFAULT: 'fft_norm'), 'sample'

    # img_folders = ['Foliage', 'LandWater', 'Snow', 'Animals', 'Flowers', 'ManMade', 'Art']
    img_folders = ['Foliage', 'LandWater', 'Snow']
    # img_folders = ['Animals', 'Flowers', 'ManMade']
    # img_folders = ['Art']

    for img_folder in img_folders:
        corr_folder = os.path.join('natural_imgs', img_folder, 'corrs', method)

        if not os.path.exists(corr_folder):
            os.makedirs(corr_folder)

        params = generate_params(method, img_folder, corr_folder,
                                 dmin=1, dmax=300, N=int(1e4), takeLog=False)


        ## P_TQDM EXPERIMENTS ##
        num_cpus = 11
        f = lambda x: calc_and_save_corrs(*x)
        list(p_imap(f, params, num_cpus=num_cpus))
