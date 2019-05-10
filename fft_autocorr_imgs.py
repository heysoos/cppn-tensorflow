import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from os import path, listdir
from PIL import Image


def fft_autocorr_imgs(img_paths):

    acorr = []
    # for i, img_path in enumerate(img_paths):
    #     img = np.array(Image.open(img_path).convert('L'))/255
    #     acorr.append( fft_autocorr(img )

    for i in range(500):
        img_path = img_paths[i]
        img = np.array(Image.open(img_path).convert('L'))/255
        acorr.append( fft_autocorr(img) )

    return acorr

def fft_autocorr(img):
    nm = np.product(np.shape(img))
    return scipy.signal.fftconvolve(img, img[::-1, ::-1]) / nm

def load_image_directories(img_folder=None):
    # make a list of all imgs in image directory

    dir = 'save/img_architectures'
    imgs_folder = path.join(dir, img_folder)
    imgs_paths = [path.join(imgs_folder, f) for f in listdir(imgs_folder) if f.endswith('.png')]

    return imgs_paths


if __name__ == '__main__':

    img_folder = 'big'
    img_paths = load_image_directories(img_folder)

    acorr = fft_autocorr(img_paths)

    plt.imshow(np.mean(acorr, axis=0))

    plt.show()