#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
model_manager.py
============

For training, loading, storing and evaluating NN model.
Specifically for usage in Relion external reconstruct.
Can be used as a runnable for a single-pass model evaluation.

Authors: Dari Kimanius

"""

import tensorflow as tf
import numpy as np
import os
import logging
from shutil import copyfile
import sys
import mrcfile as mrc
from denoiser.model import model, ops


def standardize(input):
    mean = np.mean(input, axis=(1, 2, 3, 4))
    mean = np.resize(mean, (input.shape[0], 1, 1, 1, 1))
    std = np.std(input, axis=(1, 2, 3, 4)) + 1e-12
    std = np.resize(std, (input.shape[0], 1, 1, 1, 1))
    return mean, std


def normalize(input):
    norm = np.sqrt(np.sum(np.square(input), axis=(1, 2, 3, 4))) + 1e-12
    norm = np.resize(norm, (input.shape[0], 1, 1, 1, 1))
    return norm


class ModelManager(object):
    def __init__(self, config=None):
        self.path = None
        self.is_train = None
        self.sess = None
        self.input = None
        self.output = None
        self.reg_losses = None
        self.global_step = None
        self.learning_rate = None
        self.optimizer = None
        self.train_op = None
        self.denoised = None
        self.residual = None
        self.merged_summary = None
        self.writer = None
        self.saver = None
        self.write_meta_graph = False
        self.normalize = False
        self.standardize = False
        self.config = config

    def setup(self, image_size, path,
              normalize=True,
              standardize=False):

        self.normalize = normalize
        self.standardize = standardize
        self.path = path

        self.sess = tf.Session(config=self.config)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        with tf.device('/gpu:0'):

            self.input = tf.placeholder(shape=image_size, dtype=tf.float32)
            self.output = tf.placeholder(shape=image_size, dtype=tf.float32)

            self.residual = model(self.input, self.is_train)

            loss, self.reg_losses, l1_loss, l2_loss, self.denoised = \
                ops(self.input, self.output, self.residual)

            self.learning_rate = tf.placeholder(dtype=tf.float32)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(loss, global_step=self.global_step)

        with tf.name_scope('Loss'):
            tf.summary.scalar('Total', loss)
            tf.summary.scalar('L1', l1_loss)
            tf.summary.scalar('L2', l2_loss)
            tf.summary.scalar('Regularization', self.reg_losses)
        with tf.name_scope('Norms'):
            tf.summary.scalar('Input', tf.norm(self.input))
            tf.summary.scalar('Output', tf.norm(self.output))
        with tf.name_scope('Stds'):
            tf.summary.scalar('Input', tf.math.reduce_std(self.input))
            tf.summary.scalar('Output', tf.math.reduce_std(self.output))
        with tf.name_scope('Means'):
            tf.summary.scalar('Input', tf.reduce_mean(self.input))
            tf.summary.scalar('Output', tf.reduce_mean(self.output))
        with tf.name_scope('Maximum'):
            tf.summary.image('Input', tf.reduce_max(self.input, axis=3), max_outputs=1)
            tf.summary.image('Ground_Truth', tf.reduce_max(self.output, axis=3), max_outputs=1)
            tf.summary.image('Unet', tf.reduce_max(self.residual, axis=3), max_outputs=1)
            tf.summary.image('Output', tf.reduce_max(self.denoised, axis=3), max_outputs=1)
        with tf.name_scope('Slice'):
            slice = int(image_size[3] / 2)
            tf.summary.image('Input', self.input[..., slice, :], max_outputs=1)
            tf.summary.image('Ground_Truth', self.output[..., slice, :], max_outputs=1)
            tf.summary.image('Unet', self.residual[..., slice, :], max_outputs=1)
            tf.summary.image('Output', self.denoised[..., slice, :], max_outputs=1)

        self.merged_summary = tf.summary.merge_all()

        # set up the logger
        summary_path = os.path.join(self.path, "summary")
        self.writer = tf.summary.FileWriter(summary_path)

        self.writer.add_graph(self.sess.graph)
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        root_src_path = os.path.dirname(os.path.realpath(__file__))
        model_src_path = os.path.join(root_src_path, "model.py")

        copyfile(model_src_path, os.path.join(self.path, "model.py"))

        tf.add_to_collection('residual', self.residual)
        tf.add_to_collection('denoised', self.denoised)
        tf.add_to_collection('input', self.input)
        tf.add_to_collection('output', self.output)
        tf.add_to_collection('is_train', self.is_train)
        tf.add_to_collection('learning_rate', self.learning_rate)
        tf.add_to_collection('global_step', self.global_step)

        preprocess = tf.constant([self.standardize, self.normalize], dtype=bool)
        tf.add_to_collection('preprocess', preprocess)

    def evaluate(self, data, residual=False):
        if self.standardize:
            mean, std = standardize(data)
            data = (data - mean) / std

        if self.normalize:
            norm = normalize(data)
            data /= norm

        if residual:
            denoised = self.sess.run(self.residual, feed_dict={self.input: data,
                                                               self.is_train: False})
        else:
            denoised = self.sess.run(self.denoised, feed_dict={self.input: data,
                                                               self.is_train: False})

        if self.normalize:
            denoised *= norm
        if self.standardize:
            denoised = denoised * std + mean

        return denoised

    def train(self, input, output, learning_rate):
        if self.standardize:
            mean, std = standardize(input)
            input = (input - mean) / std
            output = (output - mean) / std

        if self.normalize:
            norm = normalize(input)
            input /= norm
            output /= norm

        self.sess.run(self.train_op, feed_dict={self.input: input,
                                                self.output: output,
                                                self.learning_rate: learning_rate,
                                                self.is_train: True})

    def run_summary(self, input, output):
        if self.standardize:
            mean, std = standardize(input)
            input = (input - mean) / std
            output = (output - mean) / std

        if self.normalize:
            norm = normalize(input)
            input /= norm
            output /= norm

        merged, step = self.sess.run([self.merged_summary, self.global_step],
                                     feed_dict={self.input: input,
                                                self.output: output,
                                                self.is_train: False})
        self.writer.add_summary(merged, global_step=step)

    def get_input_size(self):
        s = self.input.get_shape().as_list()
        return s[1:4]

    def save(self, write_meta_graph=False):
        md_path = os.path.join(self.path, "model")
        if write_meta_graph:
            self.saver.export_meta_graph(
                md_path + ".meta",
                clear_devices=True,
                strip_default_attrs=True)
            logging.info("Meta file save to file: " + md_path + ".meta")
        self.saver.save(self.sess, md_path,
                        write_meta_graph=False,
                        global_step=self.global_step)
        logging.info("Checkpoint file output to: " + md_path)

    def load(self, path, for_training=True):
        self.sess = tf.Session(config=self.config)
        self.path = path

        meta_fn = os.path.join(path, "model.meta")
        self.saver = tf.compat.v1.train.import_meta_graph(
            meta_fn,
            clear_devices=True)
        logging.info("Loading meta file: " + meta_fn)
        checkpint_fn = tf.compat.v1.train.latest_checkpoint(path)
        self.saver.restore(self.sess, checkpint_fn)

        residual = tf.compat.v1.get_collection('residual')
        if len(residual) > 0:
            self.residual = residual
        self.denoised = tf.compat.v1.get_collection('denoised')[0]
        self.input = tf.compat.v1.get_collection('input')[0]
        self.is_train = tf.compat.v1.get_collection('is_train')[0]

        if for_training:
            self.output = tf.compat.v1.get_collection('output')[0]
            self.learning_rate = tf.compat.v1.get_collection('learning_rate')[0]
            self.train_op = tf.compat.v1.get_collection('train_op')[0]
            self.global_step = tf.compat.v1.get_collection('global_step')[0]
            preprocess = tf.compat.v1.get_collection('preprocess')
            self.merged_summary = tf.summary.merge_all()
            if len(preprocess) > 0:
                preprocess = self.sess.run(preprocess[0])
                if len(preprocess) == 2:
                    [self.standardize, self.normalize] = preprocess
                else:
                    print("WARNING: Preprocess settings not the right size")
                    self.standardize = True  # Backward comp
            else:
                print("WARNING: Preprocess settings not found")
                self.standardize = True  # Backward comp
        logging.info('Save restored')

        summary_path = os.path.join(self.path, "summary")
        self.writer = tf.compat.v1.summary.FileWriter(summary_path)

    def end(self):
        tf.reset_default_graph()
        self.sess.close()

