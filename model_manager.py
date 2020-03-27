#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
model_manager.py
============

For loading, storing and evaluating NN model.
Specifically for usage in Relion external reconstruct.
Can be used as a runnable for a single-pass model evaluation.

Authors: Dari Kimanius

Example usage:
    model_manager.py path/to/model path/to/denisty.mrc path/to/output.mrc
"""

import tensorflow as tf
import numpy as np
import os
import sys
import mrcfile as mrc


class ModelManager(object):
    def __init__(self, config=None):
        self.path = None
        self.is_train = None
        self.sess = None
        self.input = None
        self.denoised = None
        self.config = config

    def load(self, path):
        self.sess = tf.Session(config=self.config)
        self.path = path

        meta_fn = os.path.join(path, "model.meta")
        saver = tf.compat.v1.train.import_meta_graph(meta_fn, clear_devices=True)
        checkpint_fn = tf.compat.v1.train.latest_checkpoint(path)
        saver.restore(self.sess, checkpint_fn)

        self.denoised = tf.compat.v1.get_collection('denoised')[0]
        self.input = tf.compat.v1.get_collection('input')[0]
        self.is_train = tf.compat.v1.get_collection('is_train')[0]

    def get_input_size(self):
        s = self.input.get_shape().as_list()
        return s[1:4]

    def evaluate(self, data):
        mean = np.mean(data, axis=(1, 2, 3, 4))
        mean = np.resize(mean, (data.shape[0], 1, 1, 1, 1))
        std = np.std(data, axis=(1, 2, 3, 4)) + 1e-12
        std = np.resize(std, (data.shape[0], 1, 1, 1, 1))
        data = (data - mean) / std

        denoised = self.sess.run(self.denoised, feed_dict={self.input: data,
                                                           self.is_train: False})
        return denoised * std + mean

    def end(self):
        tf.reset_default_graph()
        self.sess.close()


if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]
    MAP_PATH = sys.argv[2]
    OUT_PATH = sys.argv[3]

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    net = ModelManager(config)
    net.load(MODEL_PATH)
    net_size = net.get_input_size()

    input_mrc = mrc.open(MAP_PATH)
    voxel_size = input_mrc.voxel_size.x
    origin = input_mrc.header['origin']

    data = np.resize(np.copy(input_mrc.data),
                     (1, net_size[0], net_size[1], net_size[2], 1))
    denoised = net.evaluate(data)[0, ..., 0]
    net.end()

    (z, y, x) = denoised.shape
    o = mrc.new(OUT_PATH, overwrite=True)
    o.header['cella'].x = x * voxel_size
    o.header['cella'].y = y * voxel_size
    o.header['cella'].z = z * voxel_size
    o.header['origin'].x = origin.x
    o.header['origin'].y = origin.y
    o.header['origin'].z = origin.z
    out_box = np.reshape(denoised, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.flush()
    o.update_header_stats()
    o.close()
