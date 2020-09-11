#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
model.py
============
File with graph layout for the model used in confidence weighted regularisation by denoising.
Authors: Dari Kimanius
"""

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

_REG_SCALE = 0.01

_L2_LOSS = True
_RESIDUAL_LOSS = True


def batch_norm(inputs, training):
  # Set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=4,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def conv(inputs, filters, kernel_size=3, strides=1):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    return tf.layers.conv3d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        kernel_regularizer=regularizer)


def upconv_3D(tensor, n_filter):
    return tf.layers.conv3d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))


def upconv_concat(inputA, input_B, n_filter):
    up_conv = upconv_3D(inputA, n_filter)
    return tf.concat([up_conv, input_B], axis=-1)


def pool(inputs):
    return tf.layers.average_pooling3d(
        inputs, 2, 2,
        padding='same',
        data_format='channels_last',
    )


def model(inputs, is_train):
    with tf.variable_scope('multiscale', reuse=tf.AUTO_REUSE):
        down = []

        block_sizes = [1]*5
        n_filter = 4

        inputs = tf.layers.conv3d(
            inputs,
            n_filter, 3,
            activation=None,
            padding='same',
            name="pre")

        for i, num_blocks in enumerate(block_sizes):
            nf = n_filter * (2 ** (i + 2))
            s = inputs.shape
            down.append(inputs)

            inputs = batch_norm(inputs, is_train)
            inputs = tf.nn.relu(inputs)
            inputs = conv(inputs=inputs, filters=nf)

            for b in range(num_blocks):
                inputs = batch_norm(inputs, is_train)
                inputs = tf.nn.relu(inputs)
                inputs = conv(inputs=inputs, filters=nf // 2)

            inputs = pool(inputs)

            print(i, s, "->", inputs.shape)

        down = down[::-1]
        block_sizes = block_sizes[::-1]

        for i, num_blocks in enumerate(block_sizes):
            nf = n_filter * (2 ** (len(block_sizes) - i - 1))
            s1 = inputs.shape
            s2 = down[i].shape
            inputs = upconv_concat(inputs, down[i], nf)
            s3 = inputs.shape

            inputs = batch_norm(inputs, is_train)
            inputs = tf.nn.relu(inputs)
            inputs = conv(inputs=inputs, filters=nf)

            for b in range(num_blocks):
                inputs = batch_norm(inputs, is_train)
                inputs = tf.nn.relu(inputs)
                inputs = conv(inputs=inputs, filters=nf // 2)

            print(len(block_sizes) - i, s1, "+", s2, "->", s3, "->", inputs.shape)

        inputs = tf.layers.conv3d(
            inputs,
            1, 1,
            activation=None,
            padding='same',
            name="post")

        return inputs


def ops(input, output, model):
    reg_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    if _RESIDUAL_LOSS:
        l1_loss = tf.reduce_mean(tf.abs(model - (input - output)))
        l2_loss = tf.reduce_mean((model - (input - output)) ** 2)
        denoised = input - model
    else:
        l1_loss = tf.reduce_mean(tf.abs(model - output))
        l2_loss = tf.reduce_mean((model - output) ** 2)
        denoised = model

    if _L2_LOSS:
        train_loss = l2_loss
    else:
        train_loss = l1_loss

    train_loss += reg_losses * _REG_SCALE

    return train_loss, reg_losses, l1_loss, l2_loss, denoised
    
