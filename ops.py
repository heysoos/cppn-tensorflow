import math
import numpy as np
import tensorflow as tf

'''
some helper code borrowed from:
https://github.com/carpedm20/DCGAN-tensorflow
'''

def linear(input_, output_size, scope=None, stddev=1.0, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def fully_connected(input_, output_size, scope=None, stddev=1.0, with_bias=True,
                    clip=False, clip_min=None, clip_max=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "FC"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 # tf.constant_initializer(1))
            tf.random_normal_initializer(stddev=stddev))
            # tf.random_uniform_initializer(minval=-0.5, maxval=0.5))

        if clip:
            matrix = tf.clip_by_value(matrix, clip_min, clip_max)

        result = tf.matmul(input_, matrix)

        if with_bias:
            bias = tf.get_variable("bias", [1, output_size],
                initializer=tf.random_normal_initializer(stddev=stddev))
            result += bias*tf.ones([shape[0], 1], dtype=tf.float32)

        return result

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
