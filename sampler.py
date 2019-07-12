'''
Implementation of Compositional Pattern Producing Networks in Tensorflow

https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

@hardmaru, 2016

Sampler Class

This file is meant to be run inside an IPython session, as it is meant
to be used interacively for experimentation.

It shouldn't be that hard to take bits of this code into a normal
command line environment though if you want to use outside of IPython.

usage:

%run -i sampler.py

sampler = Sampler(z_dim = 4, c_dim = 1, scale = 8.0, net_size = 32)

'''

import numpy as np
import tensorflow as tf
import math
import random
# import PIL
from PIL import Image
import pylab
import matplotlib.pyplot as plt
import images2gif
from images2gif import writeGif
import imageio
from datetime import datetime
from skimage.color import *
from random import sample
from copy import deepcopy
from audio_loader import load_audio
from os import path, makedirs
import json

from model import CPPN


# mgc = get_ipython().magic
# mgc(u'matplotlib inline')
# pylab.rcParams['figure.figsize'] = (10.0, 10.0)

class Sampler():
    def __init__(self, z_dim=8, c_dim=1, scale=10.0, net_size=32, num_layers=8, seed=0, img=None):
        self.cppn = CPPN(z_dim=z_dim, c_dim=c_dim, scale=scale,
                         net_size=net_size, num_layers=num_layers, seed=seed, img=img)
        self.z = self.generate_z()  # saves most recent z here, in case we find a nice image and want the z-vec

    def reinit(self):
        self.cppn.reinit()

    def generate_f_params(self):
        # generate random frequency
        w = np.random.randn() + 1
        f_params = [w]
        # generate random parameters for input functions
        for i in range(len(self.cppn.f)):
            f_params.append(np.random.random())

        return f_params


    def generate_z(self):
        z = np.random.uniform(-1.0, 1.0, size=(1, self.cppn.z_dim)).astype(np.float32)
        return z


    def generate(self, z=None, x_dim=512, y_dim=512, scale=10.0, seed=0,
                 f_params=None, img=None):
        if z is None:
            z = self.generate_z()
        else:
            z = np.reshape(z, (1, self.cppn.z_dim))
        self.z = z
        return self.cppn.generate(z=z, x_dim=x_dim, y_dim=y_dim, scale=scale, seed=seed,
                                  f_params=f_params, img=img)[0]

    def generate_hires(self, z, res=512, x_res_factor=2, y_res_factor=2, scale=8.0, seed=0,
                       f_params=None, img=None):
        self.z = z
        return self.cppn.generate_hires(z, res=res, x_res_factor=x_res_factor, y_res_factor=y_res_factor, scale=scale,
                                        seed=seed, f_params=f_params, img=img)

    # def get_image(filepath):
    #     imageio.imread(filepath)
    #     #TODO: check color/size
    #     return img


    def show_image(self, image_data, method='rgb'):
        '''
        image_data is a tensor, in [height width depth]
        image_data is NOT the PIL.Image class
        '''
        ax = plt.subplot(1, 1, 1)
        y_dim = image_data.shape[0]
        x_dim = image_data.shape[1]
        c_dim = self.cppn.c_dim
        if c_dim > 1:
            image_data = self.image2rgb(image_data, method)

            ax.imshow(image_data, interpolation='nearest')
        else:
            ax.imshow(image_data.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')

        plt.axis('off')
        ax.set_aspect('equal', 'box')
        plt.show()

    def image2rgb(self, image_data, method):
        c_dim = self.cppn.c_dim
        if c_dim == 4:
            image_data = rgba2rgb(image_data)
        elif method == 'hsv':
            image_data = hsv2rgb(image_data)
        elif method == 'xyz':
            image_data = xyz2rgb(image_data)
        elif method == 'yuv':
            image_data = self.normalize_img(yuv2rgb(image_data))
        elif method == 'YPbPr':
            image_data = self.normalize_img(ypbpr2rgb(image_data))
        elif method == 'YDbDr':
            image_data = self.normalize_img(ydbdr2rgb(image_data))
        # if method == 'rgb' do nothing

        return image_data

    def normalize_img(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))


    def save_png(self, image_data, filename):
        # img_data = np.array(1 - image_data)
        y_dim = image_data.shape[0]
        x_dim = image_data.shape[1]
        c_dim = self.cppn.c_dim


        # img_data = (255 * self.generate(z, x_dim, y_dim, scale,
        #                                    f_params=f_params, img=img)).astype(np.uint8)
        img_data = image_data

        if c_dim > 1:
            img_data = np.array(img_data.reshape((y_dim, x_dim, c_dim)) * 255.0, dtype=np.uint8)
        else:
            img_data = np.array(img_data.reshape((y_dim, x_dim)) * 255.0, dtype=np.uint8)
        im = Image.fromarray(img_data)
        im.save(filename)


    def to_image(self, image_data):
        # convert to PIL.Image format from np array (0, 1)
        img_data = np.array(1 - image_data)
        y_dim = image_data.shape[0]
        x_dim = image_data.shape[1]
        c_dim = self.cppn.c_dim
        if c_dim > 1:
            img_data = np.array(img_data.reshape((y_dim, x_dim, c_dim)) * 255.0, dtype=np.uint8)
        else:
            img_data = np.array(img_data.reshape((y_dim, x_dim)) * 255.0, dtype=np.uint8)
        im = Image.fromarray(img_data)
        return im

    def save_hires_png_seq(self, z1, r, n_frame=10, res=512, x_res_factor=2, y_res_factor=2,
                           scale=1.0, seed=0, f_params=None, img=None, method='rgb'):
        '''
        this saves a hi-res png sequence by rotating 360 from latent vector z1 back to itself.
        r: radius of rotation
        n_frame: number of states in between z1 and z2 morphing effect, exclusive of z1 and z2
        '''


        time = datetime.now().strftime("%y-%m-%d-%H-%M-%S.%f")
        folder = 'save/png_seq/' + time
        if not path.exists(time):
            makedirs(folder)

        delta_theta = (2 * np.pi) / (n_frame + 1)

        z = deepcopy(z1)
        total_frames = n_frame + 2
        for i in range(total_frames):
            theta = np.tile(delta_theta * float(i), int(np.size(z)/2))

            delta_z1 = r * np.sin(theta)
            delta_z2 = r * np.cos(theta) - r  #  minus r so that frame 0 starts from initial z img

            delta_z = np.stack([x for t in zip(delta_z1, delta_z2) for x in t])

            z = z1 + delta_z

            image_data = self.generate_hires(z, res, x_res_factor, y_res_factor, scale, seed,
                                                    f_params, img)

            image_data = (255 * self.image2rgb(image_data, method)).astype(np.uint8)
            print('processing image ', i)
            figname = folder + '/' + time + '-' + str(i).zfill(4) + '.png'
            imageio.imwrite(figname, image_data, format='png')



    def save_anim_gif(self, z1, z2, filename, n_frame=10, duration1=0.5, \
                      duration2=1.0, duration=0.1, x_dim=512, y_dim=512, c_dim=1, scale=10.0,
                      reverse=True, f_params=None, img=None):
        '''
        this saves an animated gif from two latent states z1 and z2
        n_frame: number of states in between z1 and z2 morphing effect, exclusive of z1 and z2
        duration1, duration2, control how long z1 and z2 are shown.  duration controls frame speed, in seconds
        '''
        delta_z = (z2 - z1) / (n_frame + 1)
        total_frames = n_frame + 2
        images = []
        for i in range(total_frames):
            z = z1 + delta_z * float(i)
            # images.append(self.to_image(self.generate(z, x_dim, y_dim, scale)))
            images.append((255 * self.generate(z, x_dim, y_dim, scale,
                                               f_params=f_params, img=img)).astype(np.uint8))
            # if self.cppn.c_dim == 1:
            #     images.append(self.generate(z, x_dim, y_dim, scale).reshape(x_dim, y_dim))
            # else:
            #   images.append(self.generate(z, x_dim, y_dim, scale).reshape(x_dim, y_dim, self.cppn.c_dim))
            print('processing image ', i)
        durations = [duration1] + [duration] * n_frame + [duration2]
        if reverse == True:  # go backwards in time back to the first state
            revImages = list(images)
            revImages.reverse()
            revImages = revImages[1:]
            images = images + revImages
            durations = durations + [duration] * n_frame + [duration1]

        # images = np.array(images)
        print('writing gif file...')
        # imageio.imwrite(filename, images, format='gif', fps=60)
        imageio.mimsave(filename, images, format='gif', fps=30)
        # imageio.mimsave(filename, images, format='ffmpeg', fps=30, quality = 10)
        # writeGif(filename, images, duration = durations)

    def save_wave_anim_gif(self, z1, r, filename, n_frame=10, duration1=0.5, \
                      duration2=1.0, duration=0.1, x_dim=512, y_dim=512, c_dim=1, scale=10.0,
                      reverse=True, f_params=None, img=None, method='rgb'):
        '''
        this saves an animated gif by rotating 360 from latent vector z1 back to itself.
        r: radius of rotation
        n_frame: number of states in between z1 and z2 morphing effect, exclusive of z1 and z2
        duration1, duration2, control how long z1 and z2 are shown.  duration controls frame speed, in seconds
        '''

        rdim1, rdim2 = sample(range(0, np.size(z1)), 2)
        rdim = sample(range(0, np.size(z1)), 2)
        delta_theta = (2 * np.pi) / (n_frame +1)
        theta = 0

        z = deepcopy(z1)
        # delta_z = (z2 - z1) / (n_frame + 1)
        total_frames = n_frame + 2
        images = []
        for i in range(total_frames):
            theta = np.tile(delta_theta * float(i), int(np.size(z)/2))
            # z[rdim1] = 0.5 * r * np.sin(theta)
            # z[rdim2] = 0.5 * r * np.cos(2 * theta)

            delta_z1 = r * np.sin(theta) - r/2
            delta_z2 = r * np.cos(theta) - r/2

            delta_z = np.stack([x for t in zip(delta_z1, delta_z2) for x in t])

            z = z1 + delta_z


            # z = z1 + delta_z * float(i)
            # images.append(self.to_image(self.generate(z, x_dim, y_dim, scale)))
            image_data = (255 * self.generate(z, x_dim, y_dim, scale,
                                               f_params=f_params, img=img))
            image_data = self.image2rgb(image_data, method).astype(np.uint8)
            images.append(image_data)
            # if self.cppn.c_dim == 1:
            #     images.append(self.generate(z, x_dim, y_dim, scale).reshape(x_dim, y_dim))
            # else:
            #   images.append(self.generate(z, x_dim, y_dim, scale).reshape(x_dim, y_dim, self.cppn.c_dim))
            print('processing image ', i)
        durations = [duration1] + [duration] * n_frame + [duration2]
        if reverse == True:  # go backwards in time back to the first state
            revImages = list(images)
            revImages.reverse()
            revImages = revImages[1:]
            images = images + revImages
            durations = durations + [duration] * n_frame + [duration1]

        # images = np.array(images)
        print('writing gif file...')
        # imageio.imwrite(filename, images, format='gif', fps=60)
        imageio.mimsave(filename, images, format='gif', fps=30)
        # imageio.mimsave(filename, images, format='ffmpeg', fps=30, quality = 10)
        # writeGif(filename, images, duration = durations)

    def save_wave_cat_anim_gif(self, z1, r_global, filename, n_frame=10, duration1=0.5, \
                           duration2=1.0, duration=0.1, x_dim=512, y_dim=512, c_dim=1, scale=10.0,
                           reverse=True, f_params=None, img=None, method='rgb'):
        '''
        this saves an animated gif by rotating 360 from latent vector z1 back to itself as a chain of the chained
        'rotations' of each individual latent vector dimension, like a caterpillar
        r: radius of rotation
        n_frame: number of states in between z1 and z2 morphing effect, exclusive of z1 and z2
        duration1, duration2, control how long z1 and z2 are shown.  duration controls frame speed, in seconds
        '''

        z = deepcopy(z1)
        h_size = int(np.size(z) / 2)

        r = z * r_global
        r1 = r[::2]
        r2 = r[1::2]
        # delta_z = (z2 - z1) / (n_frame + 1)
        total_frames = n_frame + 2

        period = int(total_frames / h_size)
        total_frames = period * h_size
        delta_theta = np.zeros((total_frames, h_size))  # the matrix
        delta_theta_vec = 2 * np.pi / period
        # delta_theta_vec = np.tile(delta_theta_vec, np.size(z))


        for i in range(h_size):
            t_start = period * i
            t_end = t_start + period
            delta_theta[t_start:t_end, i] = delta_theta_vec
        theta = np.cumsum(delta_theta, axis=0)


        images = []
        for i in range(total_frames):
            # delta_z = r * np.sin(theta[i, :]) - r / 2

            theta_vec = theta[i, :]

            delta_z1 = r1 * np.sin(theta_vec) - r2/2
            delta_z2 = r2 * np.cos(theta_vec) - r1/2

            delta_z = np.stack([x for t in zip(delta_z1, delta_z2) for x in t])

            z = z1 + delta_z
            image_data = (255 * self.generate(z, x_dim, y_dim, scale,
                                               f_params=f_params, img=img)).astype(np.uint8)
            image_data = self.image2rgb(image_data, method)
            images.append(image_data)
            print('processing image ', i)
        durations = [duration1] + [duration] * n_frame + [duration2]
        if reverse == True:  # go backwards in time back to the first state
            revImages = list(images)
            revImages.reverse()
            revImages = revImages[1:]
            images = images + revImages
            durations = durations + [duration] * n_frame + [duration1]

        print('writing gif file...')
        imageio.mimsave(filename, images, format='gif', fps=30)


    def save_recursive_anim_png_seq(self, z1, img_initial, img_evo=0.1, n_frame=100,
                                    x_dim=512, y_dim=512, scale=10.0, f_params=None):
        '''
        this saves a png sequence of images generated by recursively using the generated images as input into the
        next image. The rate of change of the input image is controlled by the 'img_evo' rate.
        img_initial: the initial input image
        img_evo: image evolution rate
        '''

        time = datetime.now().strftime("%y-%m-%d-%H-%M-%S.%f")
        folder = 'save/png_seq/r-' + time
        if not path.exists(time):
            makedirs(folder)

        r_img = img_initial  # recursive input image
        for i in range(n_frame):

            image = self.generate(z1, x_dim, y_dim, scale, f_params=f_params, img=r_img)
            image_save = (255 * image).astype(np.uint8)

            figname = folder + '/' + time + '-' + str(i).zfill(4) + '.png'
            imageio.imwrite(figname, image_save, format='png')
            print('processing image ', i)
            r_img = image

    def save_music_anim_gif(self, z1, r_global, audiopath, n_frame=10, x_dim=512, y_dim=512, scale=10.0,
                            f_params=None, img=None, acceleration=False, exp_gain=1,
                            fps=30, w_scaler=2, normalize_w = True, f_amp_w=False):
        '''
        this saves an animated gif by rotating 360 from latent vector z1 back to itself.
        r: radius of rotation
        n_frame: number of states in between z1 and z2 morphing effect, exclusive of z1 and z2
        duration1, duration2, control how long z1 and z2 are shown.  duration controls frame speed, in seconds
        w_scaler: used to scale the number of periods in gif. more periods needed for longer audio
        normalize_w: is integral(d_omega) normalized such that motion is periodic?
        f_amp_w: is the amplitude coupled with d_omega (musical amplitude)?
        TODO: threshold, gamma, num_pad, freq
        '''

        delta_z = self.calculate_delta_z(z1, r_global, audiopath, n_frame,
                                    acceleration, exp_gain, fps, w_scaler, normalize_w, f_amp_w)
        total_frames = delta_z.shape[0]

        time = datetime.now().strftime("%y-%m-%d-%H-%M-%S.%f")
        folder = 'save/png_seq/' + time
        if not path.exists(time):
            makedirs(folder)

        for i in range(total_frames):

            z = z1 + delta_z[i, :]

            image = (255 * self.generate(z, x_dim, y_dim, scale,
                                               f_params=f_params, img=img)).astype(np.uint8)

            figname = folder + '/' + time + '-' + str(i).zfill(4) + '.png'
            imageio.imwrite(figname, image, format='png')
            print('processing image ', i)


    def calculate_delta_z(self, z1, r_global, audiopath, n_frame=10,
                          acceleration=False, exp_gain=1, fps=30, w_scaler=2, normalize_w=True, f_amp_w=False):
        total_frames = n_frame
        fps = fps
        sound, fs = self.load_sound(audiopath)
        amp = self.generate_amp(sound, fs, fps)[0:total_frames]
        start_frame = 0

        # z1 = z1[0]
        r = z1 / np.max(np.abs(z1))
        # r1 = z1[0:8] / np.max(z1[0:8]) * r_global
        # r2 = z1[8:16] / np.max(z1[8:16]) * r_global
        # r = r * r_global
        r = r_global

        # r = np.logspace(np.log(1), np.log(0.5), amp.shape[1])
        # r = r_global * (amp_sums - np.min(amp_sums) + 0.001) / np.max(amp_sums)  # radius of periodicity controlled by total amplitude
        # r = r * r_global

        #  CALCULATE delta_z
        if acceleration:
            gamma = 20  # damping constant
            thresh = 20  # threshold audio
            num_pad = 3  #  number of padding elements (zeros) to re-align delta_theta with beats

            amp, theta1, theta2 = self.calculate_force_thetas(amp=amp, exp_gain=exp_gain, gamma=gamma, w_scaler=w_scaler,
                                                         num_pad=num_pad, thresh=thresh, normalize_w=normalize_w)
            # total_frames += num_pad
        else:
            amp, theta1, theta2 = self.calculate_thetas(amp=amp, exp_gain=exp_gain, w_scaler=w_scaler, normalize_w=normalize_w)

        # if total_frames > theta1.shape[0]:
        #     total_frames = theta1.shape[0]
        r_amp = 3  # radius of amplitude fluctuations: another parameter to play with!
        if f_amp_w:
            # amp /= np.max(amp)
            # amp -= np.median(amp, axis=0)
            # amp = np.cumsum(amp, axis=0)
            # freq = [1, 1, 2, 2, 2, 3, 3, 3]
            freq = [1, 3, 6, 6, 3, 2, 1, 1]
            # freq = np.power(freq, 2)
            freq = np.multiply(w_scaler, freq)

            amp = -np.cumsum(amp, axis=0)
            amp = np.divide(amp, np.max(np.abs(amp), axis=0))
            amp = np.sin(amp * freq * np.pi)

            delta_z1 = r_amp * amp * np.sin(theta1)
            delta_z2 = r_amp * amp * np.cos(theta2)

        else:
            delta_z1 = np.sin(theta1)
            delta_z2 = np.cos(theta2)

        delta_z = []
        for i in range(delta_z1.shape[0]):
            delta_z.append(np.stack([x for t in zip(delta_z1[i, :], delta_z2[i, :]) for x in t]))
        delta_z = np.stack(delta_z)
        # delta_z = np.multiply(r, delta_z) - r/2
        delta_z = np.multiply(r, delta_z)

        return delta_z


    def calculate_thetas(self, amp, exp_gain, w_scaler, normalize_w):
        ##### Theta is a function of amplitude ######
        #  Includes exponential smoothing
        #  delta_theta in acts like a smoothed version of the amp. calculate_force_thetas just uses amp.
        amp = self.smooth_amps(amp, a=0.9)

        delta_theta = np.power(amp, exp_gain)
        theta = np.cumsum(delta_theta, axis=0)
        if normalize_w:
            theta /= theta[-1, :]
            delta_theta /= theta[-1, :]
        else:
            theta /= np.max(theta)
            delta_theta /= np.max(theta)

        # s_theta = 0
        # total_frames = amp.shape[0]
        # for i in range(total_frames):
        #     # delta_theta = np.power(amp[i, :], gain)
        #     s_theta = np.add(s_theta, delta_theta[i, :])
        #
        #     theta.append(s_theta)

        # theta = np.stack(theta)
        # theta = theta / theta[-1, :]

        # used to scale delta_thetas
        # freq = [1, 1, 2, 2, 2, 3, 3, 3]
        freq = [1, 3, 6, 6, 3, 2, 1, 1]
        freq = np.multiply(w_scaler, freq)
        ## THESE FREQUENCIES USED TO BE DIFFERENT, THATS WHY THERES A THETA1 AND THETA2
        rand_freq1 = freq
        rand_freq2 = freq

        theta1 = rand_freq1 * np.tile(2 * np.pi, theta.shape[1]) * theta
        theta2 = rand_freq2 * np.tile(2 * np.pi, theta.shape[1]) * theta

        return delta_theta, theta1, theta2

    def calculate_force_thetas(self, amp, exp_gain, gamma, w_scaler, num_pad, thresh, normalize_w):
        ##### ACCEELRATION MODE ####
        #  theta: Theta is a function of velocity and acceleration, which is a function of amplitude
        #  - Includes damping
        # amp = self.smooth_amps(amp, a=0.9)
        amp = self.percentile_threshold(amp, thresh)

        theta1 = np.zeros(amp.shape[1])
        v1 = np.zeros(amp.shape[1])
        a1 = np.zeros(amp.shape[1])
        x1 = 0

        delta_t = 0.01
        w_scaler = int(2)

        theta = []
        total_frames = amp.shape[0]
        for i in range(total_frames):
            a1_past = deepcopy(a1)
            v1_past = deepcopy(v1)

            a1 = np.power(amp[i, :], exp_gain) - gamma * v1_past

            dv1 = 0.5 * delta_t * (a1 + a1_past)
            v1 = np.add(v1, dv1)

            dx1 = delta_t * v1_past + 0.5 * delta_t ** 2 * a1_past
            x1 = np.add(x1, dx1)

            theta.append(x1)

        # pad thetas to re-align beats, usually 3 frames is ok, might need to tune this per song though
        theta = np.stack(theta)
        pad = np.tile(0, (num_pad, theta.shape[1]))
        theta = np.concatenate((pad, theta))
        amp = np.concatenate((pad, amp))

        # normalize so that the final frame has theta = 1
        if normalize_w:
            theta = theta / theta[-1, :]

        # generate rotation frequencies
        freq = np.multiply(w_scaler, [1, 2, 2, 3, 4, 4, 6, 8])
        rand_freq1 = freq
        rand_freq2 = freq
        # rand_freq1 = np.random.permutation(freq)
        # rand_freq2 = np.random.permutation(freq)
        # rand_freq1 = sample(freq, len(freq))
        # rand_freq2 = sample(freq, len(freq))
        # rand_freq1 = [1, 2, 2, 4, 4, 4, 4, 4]
        # rand_freq2 = [1, 1, 3, 3, 3, 3, 3, 5]
        # rand_freq1 = [1, 2, 3, 4, 4, 4, 5, 8]
        # rand_freq2 = [1, 2, 3, 4, 4, 4, 5, 8]

        theta1 = rand_freq1 * np.tile(2 * np.pi, theta.shape[1]) * theta
        theta2 = rand_freq2 * np.tile(2 * np.pi, theta.shape[1]) * theta
        ##################################

        return amp, theta1, theta2

    def percentile_threshold(self, amp, percentile):
        threshold = np.percentile(amp, percentile, axis=0)
        amp[amp < threshold] = 0

        return amp

    def generate_amp(self, sound, fs, fps):
        amps = self.do_stft(sound, fs, fps)
        amps = 0.5 * amps / np.median(amps, 0)

        amps[amps < 0.1] = 0.0
        amps = amps - np.min(amps, axis=0)

        return amps

    def load_sound(self, audiopath):
        sound, fs = load_audio(audiopath)

        return sound, fs

    def condense_spectrum(self, ampspectrum):
        #
        bands = np.zeros(8, dtype=np.float32)
        #
        bands[0] = np.sum(ampspectrum[0:4])
        bands[1] = np.sum(ampspectrum[4:12])
        bands[2] = np.sum(ampspectrum[12:28])
        bands[3] = np.sum(ampspectrum[28:60])
        bands[4] = np.sum(ampspectrum[60:124])
        bands[5] = np.sum(ampspectrum[124:252])
        bands[6] = np.sum(ampspectrum[252:508])
        bands[7] = np.sum(ampspectrum[508:])
        #
        return bands

    def do_stft(self, sound, fs, fps):
        #
        nsamples = len(sound)
        wsize = 2048
        stride = int(fs / fps)

        #
        amplitudes = []

        stop = False
        start = 0

        while not stop:
            #
            end = start + wsize
            if end > nsamples:
                end = nsamples
            #
            chunk = sound[start:end]

            if len(chunk) < 2048:
                padsize = 2048 - len(chunk)
                chunk = np.pad(chunk, (0, padsize), 'constant', constant_values=0)
            #
            freqspectrum = np.fft.fft(chunk)[0:1024]
            amplitudes.append(self.condense_spectrum(np.abs(freqspectrum)))
            #
            start = start + stride

            if start >= nsamples:
                stop = True

        #
        return np.stack(amplitudes).astype(np.float32)

    def smooth_amps(self, amps, a):
        '''
        smooths and returns audio amplitudes using an exponential smoothing


        :param amps: audio amplitudes
        :param a: averaging ratio
        :return: smooth_amps
        '''

        smooth_amps = a * amps[1:] + (1 - a) * amps[0:-1]

        return smooth_amps

    def zip_matrix(self, mat1, mat2):
        zip_mat = []
        for f in range(mat1.shape[1]):
            zip_mat.append(mat1[:, f])
            zip_mat.append(mat2[:, f])
        zip_mat = np.stack(zip_mat).transpose()

        return zip_mat


    # TODO: save NN weight data (maybe use JSON only for the other params?)
def saveJSON(data, filename):
    with open(filename, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4, sort_keys=True))

def loadJSON(filename):
    with open(filename) as json_file:
        data = json.load(json_file)

    return data

def generate_architecture(total_neurons, num_layers, omega, alpha, mu):

        # ensuring omega is a fraction of one full period
        omega = omega * (2 * np.pi) / num_layers
        # exp. decay factor, normalized such that last layer is around 2 neurons
        mu = mu * np.log(2 / total_neurons)
        l = np.arange(1, num_layers + 1, dtype=float)

        # net_size = C_n * exp( mu * l / L) * (a + sin(-w*l)
        net_size = total_neurons * np.exp(mu * l / num_layers) * (alpha + np.sin(-omega * l))

        # 'integration factor' to keep sum(net_size) as close to total_neurons as possible
        net_size = np.array([int(np.round(x)) for x in net_size])
        Cn = np.sum(net_size / total_neurons)
        net_size = net_size / Cn
        net_size = np.array([int(np.round(x)) for x in net_size])

        # ensure no layers with less than 2 neurons (makes the architecture trivial)
        # subtract the neurons added from the first layer (usually the largest layer)
        # deficit = np.sum(net_size[net_size < 2])
        net_size[net_size < 2] = 2
        # net_size[0] -= deficit
        # print(net_size)

        return net_size


########### MUSIC TESTS ##################
# seed = 12345678
# np.random.seed(seed=seed)
#
# total_neurons = 250
# num_layers = 5
# w = 1.25
# alpha = 5
# mu = 0.1*np.log(2/total_neurons)
# l = np.arange(0, num_layers, dtype=float)
#
# net_size = total_neurons * np.exp(mu*l/num_layers)*(alpha + np.sin(-w * l))
# Cn = np.sum(net_size / total_neurons)
# # net_size[net_size < 2] += 1
# net_size = net_size / Cn
# # net_size = np.array([int(np.round(x + 5*np.random.random())) for x in net_size])
# net_size = np.array([int(np.round(x)) for x in net_size])
# net_size[net_size < 2] += 1
#
#
# c_dim = 3
# img_null = tf.zeros((640, 640))
# sampler = Sampler(z_dim = 16, scale = 8, net_size = net_size,
#                   num_layers=num_layers, c_dim=c_dim, img=img_null)
#
# z2 = sampler.generate_z()[0]
# img = None
#
# z_scale = 10
# z_factor = np.random.normal(size=16) * z_scale
#
# sortidx = np.argsort(np.abs(z_factor))
# sortidy = np.argsort(np.abs(z2[0]))
#
# z_factor = z_factor[sortidx]
# z2 = z2[sortidy]
#
#
# zz = z2 * z_factor
# print(zz)
# #
# f_params = [15, 0.1, 0, 0, 0]
#
#
# # img_data = sampler.generate(z1)
# # sampler.show_image(img_data)
#
# audiopath = 'media/muy-tranquilo-short3.mp3'
# acceleration = False
# exp_gain = 2.5
# w_scaler = 1
# f_amp_w = True
# normalize_w = False
#
# time = datetime.now().strftime("%y-%m-%d-%H-%M-%S.%f")
# figname = 'save/' + time + '.gif'
# sampler.save_music_anim_gif(zz, r_global=0.015, audiopath=audiopath, filename=figname,
#                             x_dim = 640, y_dim = 640, scale=1, f_params=f_params,
#                             n_frame=322, img=None,
#                             acceleration=acceleration,
#                             exp_gain=exp_gain, w_scaler=w_scaler,
#                             normalize_w=normalize_w,
#                             f_amp_w=f_amp_w)

############ HIRES IMAGES ##################
# net_size = [32, 32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4]
# num_layers = len(net_size)
# c_dim = 3
# img_null = np.zeros((640, 640))
#
# sampler = Sampler(z_dim = 16, scale = 8, net_size = net_size,
#                   num_layers=num_layers, c_dim=c_dim, img=img_null)
# img = np.random.random((512, 512))
# z1 = sampler.generate_z()
#
# low = np.random.random()
# mid = np.random.random()
# high = np.random.random()
#
# C = np.random.random()
# z1 = np.concatenate((C + low * z1[0:4], mid * z1[4:12], high * z1[12:16]))
# perlin_params = (1, 0.05, 6, 0.5, 2.0)
# f_params = [2, 0, perlin_params, 0, 1]
#
# pseed=0
# img_data = sampler.generate(z1, x_dim=512, y_dim=512, scale=0.1, seed=pseed,
#                             f_params=f_params, img=img)
#
# # img_data = sampler.generate_hires(z1, res=512, x_res_factor=10, y_res_factor=10, seed=0,
# #                                   scale=5, f_params=f_params, img=img_null)
# sampler.show_image(img_data)
# time = datetime.now().strftime("%y-%m-%d-%H-%M-%S.%f")
# figname = 'save/hi_res/' + time + '.png'
# imageio.imwrite(figname, (img_data * 255).astype(np.uint8), format='png')

################# CATERPILLAR IMAGES ##################
# net_size = [32, 32, 32, 16, 8, 4]
# num_layers = 6
# c_dim = 3
# img_null = tf.zeros((640, 640))
# sampler = Sampler(z_dim = 8, scale = 8, net_size=net_size,
#                   num_layers=num_layers, c_dim=c_dim, img=img_null)
#
# z2 = sampler.generate_z()
# img = None
#
# global_scale = 0.1
#
# z_factor = np.random.random(3) * global_scale
# z_factor = np.sort(z_factor)[::-1]
#
# zz = np.concatenate((z_factor[0] * z2[0:4] ,
#                      z_factor[1] * z2[4:12],
#                      z_factor[2] * z2[12:16]))
#
# f_params = [2, 0, 0, 0, 0]
#
#
#
# time = datetime.now().strftime("%y-%m-%d-%H-%M-%S.%f")
# figname = 'save/' + time + '.gif'
# # z1 = np.array(z1)
# sampler.save_wave_cat_anim_gif(zz[0], 0.1, figname,
#                       x_dim = 640, y_dim = 640, scale=10, f_params=f_params,
#                       n_frame=240, reverse=False, img=None)
#
################# RECURSION TESTS ###################
# net_size = [32, 32, 32, 16, 8, 4]
# num_layers = len(net_size)
# c_dim = 3
# img_null = np.zeros((1080, 1080))
# sampler = Sampler(z_dim=16, scale=8, net_size=net_size,
#                   num_layers=num_layers, c_dim=c_dim, seed=0, img=img_null)
#
# i_r = 0
# ##################
# x_dim = 512
# y_dim = 512
# scale = 1
# if i_r == 0:
#     x, y = np.meshgrid(np.linspace(-scale, scale, x_dim), np.linspace(-scale, scale, y_dim))
#     img = np.random.random()*x + np.random.random()*y + np.random.random()*np.sqrt(x**2 + y**2)
# mu = 0.4  # image evolution rate
# ##################
#
# z2 = sampler.generate_z()[0]
# z_scale = 0
# z_factor = np.random.normal(size=16) * z_scale
#
# sortidx = np.argsort(np.abs(z_factor))
# sortidy = np.argsort(np.abs(z2[0]))
#
# z_factor = z_factor[sortidx]
# z2 = z2[sortidy]
# zz = z2 * z_factor
# print(zz)
# ##################
#
# ##################
# # # pseed = 0
# pseed = int(np.random.random() * 1e5 )
# perlin_params = (0, 0.7, 6, 0.5, 2.0)
# f_params = [0, 0, perlin_params, 0, 10]
# ##################
#
# sampler.save_recursive_anim_png_seq(zz, img, img_evo=0.1, n_frame=100,
#                             x_dim=x_dim, y_dim=y_dim, scale=scale, f_params=f_params)
#
