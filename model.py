'''
Implementation of Compositional Pattern Producing Networks in Tensorflow

https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

@hardmaru, 2016

'''

import numpy as np
import tensorflow as tf
from ops import *


class CPPN():
    def __init__(self, batch_size=1, z_dim=32, x_dim=512, y_dim=512, c_dim=1, img=None,
                 scale=8.0, net_size=32, num_layers=8, seed=0):
        """

        Args:
        z_dim: how many dimensions of the latent space vector (R^z_dim)
        c_dim: 1 for mono, 3 for rgb.  dimension for output space.  you can modify code to do HSV rather than RGB.
        net_size: number of nodes for each fully connected layer of cppn
        scale: the bigger, the more zoomed out the picture becomes

        """
        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        self.batch_size = batch_size
        self.net_size = net_size
        self.num_layers = num_layers
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scale = scale
        self.c_dim = c_dim
        self.z_dim = z_dim

        # tf Graph batch of image (batch_size, height, width, depth)
        self.batch = tf.placeholder(tf.float32, [batch_size, x_dim, y_dim, c_dim])

        n_points = x_dim * y_dim
        self.n_points = n_points

        self.x_vec, self.y_vec, self.r_vec, self.f_vec = self._coordinates(x_dim, y_dim, scale=scale, img=img)

        # latent vector
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        # inputs to cppn, like coordinates and radius from centre
        self.x = tf.placeholder(tf.float32, [self.batch_size, None, 1], name='x')
        self.y = tf.placeholder(tf.float32, [self.batch_size, None, 1], name='y')
        self.r = tf.placeholder(tf.float32, [self.batch_size, None, 1], name='r')

        # Sina's inputs
        self.f = [tf.placeholder(tf.float32, [self.batch_size, None, 1], name='f_' + str(i))
                  for i in range(0, len(self.f_vec))]

        # builds the generator network
        # self.G = self.generator(x_dim = self.x_dim, y_dim = self.y_dim)
        self.G = self.generator()

        self.init()
        #
        # # Initializing the tensor flow variables
        # # init = tf.initialize_all_variables()
        # init = tf.global_variables_initializer()
        # # Launch the session
        # self.sess = tf.InteractiveSession()
        # self.sess.run(init)

    def init(self):

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

    def reinit(self):
        # init = tf.initialize_variables(tf.trainable_variables())
        init = tf.global_variables_initializer()

        # TODO: tf.variables_initializer??
        self.sess.run(init)

    def _coordinates(self, x_dim=32, y_dim=32, scale=1.0,
                     f_params=None, img=None):
        '''
        calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
        '''
        n_points = x_dim * y_dim
        x_range = scale * (np.arange(x_dim) - (x_dim - 1) / 2.0) / (x_dim - 1) / 0.5
        y_range = scale * (np.arange(y_dim) - (y_dim - 1) / 2.0) / (y_dim - 1) / 0.5
        # x_range = 10 ** (scale * (np.arange(x_dim) - (x_dim - 1) / 2.0) / (x_dim - 1) / 0.5)
        # y_range = 10 ** (scale * (np.arange(y_dim) - (y_dim - 1) / 2.0) / (y_dim - 1) / 0.5)

        ## LOGARITHMIC COORDINATES
        # x_dim_h = x_dim / 2
        # x_pos = 10 ** (scale * (np.arange(x_dim_h) - (x_dim_h - 1) / 2.0) / (x_dim_h - 1) / 0.5 - scale/2)
        # x_neg = -x_pos[::-1]
        # x_range = np.concatenate((x_neg, x_pos))
        #
        # y_dim_h = y_dim / 2
        # y_pos = scale * 10 ** (scale * (np.arange(y_dim_h) - (y_dim_h - 1) / 2.0) / (y_dim_h - 1) / 0.5 - scale/2)
        # y_neg = -y_pos[::-1]
        # y_range = np.concatenate((y_neg, y_pos))
        #
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)

        # x_mat = np.power(x_mat, scale)
        # y_mat = np.power(y_mat, scale)
        # x_mat = np.sinh(x_mat)
        # y_mat = np.cosh(y_mat)
        # r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)

        ## LOG-POLAR COORDINATES
        # # theta_range = scale * ((np.arange(x_dim) - (x_dim - 1) / 2.0) / (x_dim - 1) + 0.5) * 2*np.pi
        # # theta_mat = np.matmul(np.ones((x_dim, 1)), theta_range.reshape((1, x_dim)))
        # rho_mat = np.log(r_mat)
        #
        # # x_mat = theta_mat
        #
        # x_range = 10 ** (scale * (np.arange(x_dim) - (x_dim - 1) / 2.0) / (x_dim - 1) / 0.5)
        # x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        # y_mat = rho_mat
        # r_mat = np.multiply(r_mat, 0)

        if f_params is not None:
            #  TODO: 2D frequency maps?
            # w = 5 * np.random.randn() + 1  # frequency
            w = f_params[0]

            af1 = f_params[1]
            af2 = f_params[2]
            af3 = f_params[3]

            if img is not None:
                af4 = f_params[-1]

        else:
            w = 1

            af1 = 1
            af2 = 1
            af3 = 1
            af4 = 1

        # w = 5 * np.random.randn() + 1

        # Sina's cheat input functions
        f1 = af1 * (np.cos(w * x_mat) + np.sin(w * y_mat))

        # r = 0.1
        # r2 = 0.9
        # offset = int(2 * scale / 4)
        # theta = np.arctan2(-y_mat - offset, x_mat)
        #
        # # theta = np.arctan2(y_mat, x_mat)
        # # m = 1 - r * np.sin(2 * w * theta)
        # # f1 = af1 * r_mat * m
        #
        # r_mat_new = np.sqrt(x_mat * x_mat + (-y_mat - offset) * (-y_mat - offset))
        # heart = (2 - 2 * np.sin(theta) + np.sin(theta) * np.sqrt(np.abs(np.cos(theta))) / (np.sin(theta) + 1.4)) ** r
        # heart *= 1 / np.max(heart)
        # f1 = af1 * r_mat_new * (1 - r2 * heart)

        # f1 = af1 * (np.tan(w * x_mat) + np.tan(w * y_mat))
        # f1 = af1 * np.cosh(w * r_mat)
        # f2 = af2 * np.tanh(5 + y_mat)
        # f3 = af3 * np.tanh(5 + x_mat)

        # f2 = af2 * np.tanh(x_mat)
        f2 = af2 * (np.sin(w * x_mat ** 2) * np.cos(w * y_mat ** 2))
        f3 = af3 * np.cosh(w * r_mat)
        f_mat = [f1, f2, f3]
        # x_mat = np.power(x_mat, w)
        # y_mat = np.power(y_mat, w)
        # r_mat = np.power(r_mat, w)

        if img is not None:
            if img.shape[0] == x_mat.shape[0] and img.shape[1] == x_mat.shape[1]:
                f_mat.append(af4 * img)
            else:
                print('Input image is not the same dims as x_mat/y_mat.')

        # Sina's mods
        # x_mat = np.power(x_mat, 3)
        # y_mat = np.power(y_mat, 3)

        x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        for i in range(0, len(f_mat)):
            f_mat[i] = np.tile(f_mat[i].flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        return x_mat, y_mat, r_mat, f_mat

    def generator(self, x_dim=512, y_dim=512, reuse=False):
        with tf.variable_scope("generator") as scope:

            if reuse:
               scope.reuse_variables()

            net_size = self.net_size
            if np.size(net_size) < 2:
                net_size = np.tile(net_size, self.num_layers)
            n_points = x_dim * y_dim

            # note that latent vector z is scaled to self.scale factor.
            z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) * \
                       tf.ones([n_points, 1], dtype=tf.float32) * self.scale
            z_unroll = tf.reshape(z_scaled, [self.batch_size * n_points, self.z_dim])
            x_unroll = tf.reshape(self.x, [self.batch_size * n_points, 1])
            y_unroll = tf.reshape(self.y, [self.batch_size * n_points, 1])
            r_unroll = tf.reshape(self.r, [self.batch_size * n_points, 1])

            f_unroll = [tf.reshape(self.f[0], [self.batch_size * n_points, 1])]
            for i in range(1, len(self.f)):
                f_unroll.append(tf.reshape(self.f[i], [self.batch_size * n_points, 1]))

            U = fully_connected(z_unroll, net_size[0], 'g_0_z') + \
                fully_connected(x_unroll, net_size[0], 'g_0_x', with_bias=False) + \
                fully_connected(y_unroll, net_size[0], 'g_0_y', with_bias=False) + \
                fully_connected(r_unroll, net_size[0], 'g_0_r', with_bias=False)

            for i in range(0, len(f_unroll)):
                U += fully_connected(f_unroll[i], net_size[0], 'g_0_f' + str(i), with_bias=False)

            '''
            Below are a bunch of examples of different CPPN configurations.
            Feel free to comment out and experiment!
            '''

            ###
            ### Example: 3 layers of tanh() layers, with net_size = 32 activations/layer
            ###
            # # '''
            H = tf.nn.tanh(U)
            for i in range(self.num_layers):
                # H = tf.nn.tanh(tf.math.pow(fully_connected(H, net_size[i], 'g_tanh_' + str(i)), 1))
                # H = tf.nn.tanh(tf.math.pow(
                #     fully_connected(H, net_size[i], 'g_tanh_' + str(i), clip = True, clip_min=-1, clip_max=1), 3))

                # H = tf.nn.relu(tf.math.pow(
                #     fully_connected(H, net_size[i], 'g_relu_' + str(i)), 1))

                H = tf.nn.tanh(tf.math.pow(
                    fully_connected(H, net_size[i], 'g_tanh_' + str(i)), 3))


                # ## convolution layers
                # H = tf.reshape(H, [self.batch_size, x_dim, y_dim, -1])
                # H = tf.layers.conv2d(
                #     inputs=H,
                #     filters=net_size[i],
                #     kernel_size=[5, 5],
                #     padding="same",
                #     activation=tf.nn.tanh,
                #     name='g_conv_' + str(i))

            H  = tf.reshape(H, [self.batch_size * n_points, - 1])

                # H = lrelu(H)
                # H = tf.nn.dropout(fully_connected(H, net_size[i], 'g_dropout_' + str(i)), keep_prob=0.9)
                # H = tf.nn.softplus(fully_connected(H, net_size[i], 'g_softplus_' + str(i)))
                # H = conv2d(tf.reshape(H, [1, x_dim, y_dim, net_size]), net_size[i], name='g_conv_' +str(i), d_h=1, d_w=1)
                # H = tf.reshape(H, [n_points, net_size[i]])
                # H = lrelu(conv2d(tf.reshape(H, [1, x_dim, y_dim, net_size]), 20, name = 'g_conv_' + str(i)))

            # if the last layer is not added to residuals, do so (to ensure matrices are same as size as other layers)
            # if (self.num_layers % layer_skip) is not 0:
            #     Hr += tf.tanh(fully_connected(H, net_size[i], 'g_tanh_r_' + str(i)))

            # Hr = tf.sigmoid(fully_connected(Hr, c_dim, 'g_tanh_r_' + str(i)))

            # output = tf.sigmoid(fully_connected(H, self.c_dim, 'g_final'))
            # output = 0.5 * tf.sin(fully_connected(H, self.c_dim, 'g_final')) + 0.5
            # output = tf.sigmoid(fully_connected(H, self.c_dim, 'g_final'))
            output = tf.sigmoid(fully_connected(H, self.c_dim, 'g_final'))
            # output = 0.5 * tf.nn.tanh(fully_connected(H, self.c_dim, 'g_final')) + 0.5
            # '''

            ###
            ### Sina's weird residuals
            # H = tf.nn.tanh(U)
            # Hr = tf.nn.tanh(fully_connected(H, self.c_dim, 'g_r_init_' + str(i)))
            # layer_skip = 1  # residal layers per normal layer
            # for i in range(self.num_layers):
            #     if (i + 1) % layer_skip is 0:  # accumulate residuals every layer_skip
            #         Hr  += tf.nn.tanh(tf.math.pow(
            #             fully_connected(H, self.c_dim, 'g_tanh_r_' + str(i)), 3))
            #     H = tf.nn.tanh(tf.math.pow(
            #         fully_connected(H, net_size[i], 'g_tanh_' + str(i), clip = True, clip_min=-1e2, clip_max=1), 3))
            #
            # Hr = tf.nn.tanh(tf.math.pow(Hr, 3))
            # output = tf.sigmoid(tf.math.add(fully_connected(H, self.c_dim, 'g_final'), Hr))
            ###


            ###
            ### Similar to example above, but instead the output is
            ### a weird function rather than just the sigmoid
            '''
            H = tf.nn.tanh(U)
            for i in range(self.num_layers):
              H = tf.nn.tanh(fully_connected(H, net_size, 'g_tanh_'+str(i)))
            output = tf.sqrt(1.0-tf.abs(tf.tanh(fully_connected(H, self.c_dim, 'g_final'))))
            '''

            ###
            ### Example: mixing softplus and tanh layers, with net_size = 32 activations/layer
            ###
            '''
            H = tf.nn.tanh(U)
            H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_1'))
            H = tf.nn.tanh(fully_connected(H, net_size, 'g_tanh_2'))
            H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_2'))
            H = tf.nn.tanh(fully_connected(H, net_size, 'g_tanh_2'))
            H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_2'))
            output = tf.sigmoid(fully_connected(H, self.c_dim, 'g_final'))
            '''

            ###
            ### Example: mixing sinusoids, tanh and multiple softplus layers
            ###
            '''
            H = tf.nn.tanh(U)
            H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_1'))
            H = tf.nn.tanh(fully_connected(H, net_size, 'g_tanh_2'))
            H = tf.nn.softplus(fully_connected(H, net_size, 'g_softplus_2'))
            output = 0.5 * tf.sin(fully_connected(H, self.c_dim, 'g_final')) + 0.5
            '''

            ###
            ### Example: residual network of 4 tanh() layers
            ###
            '''
            H = tf.nn.tanh(U)
            for i in range(3):
              H = H+tf.nn.tanh(fully_connected(H, net_size, g_tanh_'+str(i)))
            output = tf.sigmoid(fully_connected(H, self.c_dim, 'g_final'))
            '''

            '''
            The final hidden later is pass thru a fully connected sigmoid later, so outputs -> (0, 1)
            Also, the output has a dimention of c_dim, so can be monotone or RGB
            '''
            result = tf.reshape(output, [self.batch_size, y_dim, x_dim, self.c_dim])

        return result

    def generate(self, z=None, x_dim=26, y_dim=26, scale=8.0, f_params=None, img=None):
        """ Generate data by sampling from latent space.

        If z is not None, data for this point in latent space is
        generated. Otherwise, z is drawn from prior in latent
        space.
        TODO: offset, remove null img if possible...
        """
        if z is None:
            z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
            # Note: This maps to mean of distribution, we could alternatively
            # sample from Gaussian distribution

        G = self.generator(x_dim=x_dim, y_dim=y_dim, reuse=True)
        x_vec, y_vec, r_vec, f_vec = self._coordinates(x_dim, y_dim, scale=scale, f_params=f_params, img=img)
        # create dict for tf.sess
        dict_data = {self.z: z, self.x: x_vec, self.y: y_vec, self.r: r_vec}
        # add custom inputs
        dict_data.update({i: d for i, d in zip(self.f, f_vec)})

        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        image = self.sess.run(G, feed_dict=dict_data, options=run_opts)

        return image

    def generate_hires(self, z, res=512, x_res_factor=2, y_res_factor=2, scale=8.0,
                       f_params=None, img=None):
        """ Generate data by sampling from latent space.
        If z is not None, data for this point in latent space is
        generated. Otherwise, z is drawn from prior in latent
        space.
        """
        x_dim = res
        y_dim = res
        n_points = x_dim * y_dim
        x_dim_big = res * x_res_factor
        y_dim_big = res * y_res_factor
        image_hires = np.zeros((x_dim_big, y_dim_big, self.c_dim))


        G = self.generator(x_dim=x_dim, y_dim=y_dim, reuse=True)
        x_vec, y_vec, r_vec, f_vec = self._coordinates(x_dim_big, y_dim_big,
                                                       scale=scale, f_params=f_params, img=img)

        # unflatten input vectors
        x_vec = x_vec.reshape(x_dim_big, y_dim_big)
        y_vec = y_vec.reshape(x_dim_big, y_dim_big)
        r_vec = r_vec.reshape(x_dim_big, y_dim_big)
        for ii in range(len(f_vec)):
            f_vec[ii] = f_vec[ii].reshape(x_dim_big, y_dim_big)

        image=[]
        for ix in range(x_res_factor):
            for iy in range(y_res_factor):
                x_start = ix * res
                x_end = x_start + res

                y_start = iy * res
                y_end = y_start + res

                # select smaller window
                x_vec_small = x_vec[x_start:x_end, y_start:y_end]
                y_vec_small = y_vec[x_start:x_end, y_start:y_end]
                r_vec_small = r_vec[x_start:x_end, y_start:y_end]

                f_vec_small = []
                for ii in range(len(f_vec)):
                    f_vec_small.append(f_vec[ii][x_start:x_end, y_start:y_end])

                # re-flatten input
                x_vec_small = np.tile(x_vec_small.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
                y_vec_small = np.tile(y_vec_small.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
                r_vec_small = np.tile(r_vec_small.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
                for i in range(0, len(f_vec_small)):
                    f_vec_small[i] = np.tile(f_vec_small[i].flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)

                # create dict for tf.sess
                dict_data = {self.z: z,
                             self.x: x_vec_small,
                             self.y: y_vec_small,
                             self.r: r_vec_small}
                # add custom inputs
                dict_data.update({i: d for i, d in zip(self.f, f_vec_small)})

                run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
                image = (self.sess.run(G, feed_dict=dict_data, options=run_opts))[0]

                image_hires[x_start:x_end, y_start:y_end, :] = image

                print('Preparing image section ' + str())


        return image_hires

    def close(self):
        self.sess.close()
