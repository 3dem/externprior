#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
exrecon_cwred.py
============

Runnable for applying confidence weighted Regularisation by Denoising in Relion
external reconstruct.

This file should only be called by Relion. This can be done by specifying the path to
this file through the environmental variable RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE.

Authors: Dari Kimanius

Example usage:
    export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python /path/to/exrecon_cwred.py /path/to/model"
    mpirun -n 3 relion_refine_mpi <flags> --external_reconstruct
"""

import tensorflow as tf
import numpy as np
import os
import mrcfile as mrc
import sys
from scipy import interpolate

from relion_fixed_it import load_star
from model_manager import ModelManager

# Suppress too much output from TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

EPS = 1e-10  # A tiny value

# For now we assign a constant,
# but this should be determined by the loaded NN model
VOXEL_SIZE = 1.5


def save_mrc(box, voxel_size, origin, filename):
    """Shortcut for writing out MRC files."""
    (z, y, x) = box.shape
    o = mrc.new(filename, overwrite=True)
    o.header['cella'].x = x * voxel_size
    o.header['cella'].y = y * voxel_size
    o.header['cella'].z = z * voxel_size
    o.header['origin'].x = origin[0]
    o.header['origin'].y = origin[1]
    o.header['origin'].z = origin[2]
    out_box = np.reshape(box, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.flush()
    o.update_header_stats()
    o.close()


def get_confidence_from_fsc(fsc):
    """Applies a simple fixed linear model to determine the confidence in the NN model."""
    res = np.ones(len(fsc))
    res[0] = 999.
    res[1:] = len(fsc) * 2 * VOXEL_SIZE / np.arange(1, len(fsc))
    threshold = np.argmax(fsc < 1/7) - 1
    if threshold > 0:
        threshold = len(fsc)-1
    current_res = res[threshold]
    w = 1.8 - 0.18 * current_res
    if w < 0:
        w = 0
    if w > 1:
        w = 1

    return w


if __name__ == "__main__":

    # Parse commandline
    paths = sys.argv
    assert len(paths) == 3
    nn_model_fn = paths[1]
    star_fn = paths[2]

    # Load external reconstruct start file
    star_file = load_star(star_fn)

    # Load the real part of the data
    with mrc.open(star_file['external_reconstruct_general']['rlnExtReconsDataReal']) as m:
        data_real = m.data

    # Load the imaginary part of the data
    with mrc.open(star_file['external_reconstruct_general']['rlnExtReconsDataImag']) as m:
        data_imag = m.data

    # Load weights
    with mrc.open(star_file['external_reconstruct_general']['rlnExtReconsWeight']) as m:
        weights = m.data

    # Extract relevant data statistics from the star file
    original_size = int(star_file['external_reconstruct_general']['rlnOriginalImageSize'])
    spec_index = np.array(star_file['external_reconstruct_tau2']['rlnSpectralIndex'], dtype=np.float)
    tau2_value = np.array(star_file['external_reconstruct_tau2']['rlnReferenceTau2'], dtype=np.float)
    tau2_inter = interpolate.interp1d(spec_index, tau2_value, fill_value=(1, 0), bounds_error=False)

    # Load FSC to determine current nominal resolution for confidence weighting
    fsc_value = np.array(star_file['external_reconstruct_tau2']['rlnGoldStandardFsc'], dtype=np.float)
    confidence = get_confidence_from_fsc(fsc_value)

    # Assign indices of each Fourier component
    (z, y, x) = weights.shape
    Z, Y, X = np.meshgrid(np.linspace(-z // 2, z // 2 - 1, z),
                          np.linspace(-y // 2, y // 2 - 1, y),
                          np.linspace(0, x - 1, x))
    R = np.round(np.sqrt(X ** 2 + Y ** 2 + Z ** 2)).astype(np.int)
    R = np.fft.ifftshift(R, axes=(0, 1))

    # Assign tau2 values to each Fourier component
    tau2 = tau2_inter(R)

    # Re-weights the data, without regularisation
    data_ft = (data_real + 1j * data_imag) / (weights + EPS)

    # Go to real space
    data = np.fft.fftshift(np.fft.irfftn(data_ft))

    # Load the NN model
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    net = ModelManager(config)
    net.load(nn_model_fn)
    nets = net.get_input_size()

    # Reshape the array to match the input shape
    data = np.resize(data, (1, nets[0], nets[1], nets[2], 1))

    # Run the network and unload the NN model
    denoised = net.evaluate(data)[0, ..., 0]
    net.end()

    # Back to Fourier space
    denoised_ft = np.fft.rfftn(np.fft.fftshift(denoised))

    # Apply confidence weighting
    denoised_ft *= confidence

    # We define a lower regime for tau2 where the
    # numerical accuracy in the division becomes an issue
    tau2lim = 1e-8
    imix = tau2 >= tau2lim

    # Define the results container
    result_ft = np.zeros(data_ft.shape, dtype=np.complex)

    # Apply Regularization by Denoising (RED)
    data_ft = data_real + 1j * data_imag
    result_ft[imix] = (data_ft[imix] + denoised_ft[imix] / tau2[imix]) / (weights[imix] + 1. / tau2[imix])
    result_ft[tau2 < tau2lim] = denoised_ft[tau2 < tau2lim]

    # Back to real space again for file-storage
    result = np.fft.fftshift(np.fft.irfftn(result_ft))

    # Relion and Numpy apply different scale factors for FT,
    # re-scale the output to match
    result = result * weights.shape[0] ** 2

    # Store to file
    target_path = star_file['external_reconstruct_general']['rlnExtReconsResult']
    save_mrc(result, VOXEL_SIZE, np.array([0, 0, 0]), target_path)
    print('Ouput to file', target_path)
