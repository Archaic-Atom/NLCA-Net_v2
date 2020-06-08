# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from JackBasicStructLib.Basic.Define import *

"""
    Contains the implementation of Convolutional Spatial Propagation Network.
    As described in "Learning Depth with Convolutional Spatial Propagation Network
    " https://arxiv.org/abs/1810.02695.
"""


def CSPN(x, sparse_depth, guidance, kernel_size, num_layers):
    num_channels = int(x.shape[3])
    num_guid_chanels = int(guidance.shape[3] // num_channels)

    # sparse_mask = tf.cast(tf.greater(sparse_depth, 0.6), tf.float32)

    propagateds = [x]
    # imm_result = propagateds[0]

    # final_result_layer = imm_result #(1 - sparse_mask) * imm_result + sparse_mask * x
    # propagateds.append(final_result_layer)

    for i in range(num_layers):
        elemwise_max_gates = []
        for j in range(num_guid_chanels):
            guid_idx = j * num_channels
            elemwise_max_gates.append(eight_way_propagation(
                guidance[:, :, :, guid_idx: guid_idx +
                         num_channels], propagateds[i], kernel_size, num_channels
            ))

        concated = tf.stack(elemwise_max_gates, axis=-1)
        result_propagation = tf.reduce_max(concated, axis=-1)

        final_result_layer = result_propagation  # + propagateds[i]
        #((1 - sparse_mask) * result_propagation) + sparse_mask * sparse_depth

        propagateds.append(final_result_layer)

    # gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.2))
    # result = gamma * propagateds[-1] + propagateds[0]
    result = propagateds[-1]  # + propagateds[0]

    return result


def eight_way_propagation(weight_matrix, blur_input, kernel, num_channels):

    weight_matrix = tf.abs(weight_matrix)
    weight = np.ones((kernel, kernel, num_channels, num_channels))
    weight[(kernel - 1) // 2, (kernel - 1) // 2, :, :] = 0
    avg_conv_weight = tf.convert_to_tensor(weight, dtype=tf.float32)

    sum_weight = tf.constant(1.0, shape=[kernel, kernel, num_channels, num_channels])
    weight_sum = tf.nn.conv2d(weight_matrix, sum_weight,
                              strides=[1, 1, 1, 1], padding='SAME')

    avg_sum = tf.nn.conv2d((weight_matrix * blur_input),
                           avg_conv_weight, strides=[1, 1, 1, 1],
                           padding='SAME')

    out = (tf.divide(weight_matrix, weight_sum)) * blur_input  # + tf.divide(avg_sum, weight_sum)

    return out


def normalize_gate(guidance):
    guidance_abs_sum = tf.reduce_sum(tf.abs(guidance), 3)
    guidance_abs_sum = guidance_abs_sum + 0.0000001
    guidance_div = tf.div(guidance, tf.expand_dims(guidance_abs_sum, 3))
    guidance_sum = tf.reduce_sum(guidance_div, 3)
    guidance_ones = tf.ones_like(guidance_sum)
    guidance = tf.subtract(guidance_ones, guidance_sum)
    guidance = tf.expand_dims(guidance, axis=3)

    guidance = tf.concat([guidance, guidance_div], 3)

    return guidance


if __name__ == '__main__':

    # Just use the same set of convolutional kernels for now
    # x here stands for blur_depth in the original implementation, which is
    # the output of the Resnet 5th guidance up project layer
    # it is also used as input for depth refinement
    w, h = None, None
    num_channels = 1
    num_guid_channels = 4
    x = tf.placeholder(tf.float32, shape=[None, w, h, num_channels],
                       name='cnn_output')
    sparse_depth = tf.placeholder(tf.float32,
                                  shape=[None, w, h, 1])
    # guidance is the original Resnet 6th guidance layer in the original
    # pytorch implementation, it has 8 channels output and same width,
    # height as the input images
    guidance = tf.placeholder(tf.float32,
                              shape=[None, w, h, num_guid_channels * num_channels],
                              name='cnn_output')

    out = CSPN(x, sparse_depth, guidance, kernel_size=3, num_layers=8)
    print(out)
