#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
model_manager.py
============

For training the NN model using gaussian noise.

Authors: Dari Kimanius
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import glob
import argparse
import mrcfile as mrc
from concurrent.futures import ThreadPoolExecutor
import time
import pickle
import matplotlib.pylab as plt
import logging

parser = argparse.ArgumentParser()
parser.add_argument('input_list', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--noLPGT', action="store_true")
parser.add_argument('--epocs', type=int, default=int(1e9))
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--block_size', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--checkpoint_steps', type=int, default=500)
parser.add_argument('--box_size', type=int, default=96)
parser.add_argument('--preload', action="store_true")
parser.add_argument('--norm', action="store_true")
parser.add_argument('--std', action="store_false")
parser.add_argument('--max_res', type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(filename=args.output_dir + "/model.log", level=logging.INFO)

from utils.grid import grid_rot90, apply_fsc, save_mrc, rescale_real
from denoiser.model_manager import ModelManager
from dataset import Dataset, get_entry_res, filter_entries
from scipy import interpolate
import tensorflow as tf

PIXEL_SIZE = 1.5 * (96. / float(args.box_size))
IMAGE_SIDE = args.box_size
IMAGE_SIZE = (None, IMAGE_SIDE, IMAGE_SIDE, IMAGE_SIDE, 1)

config = tf.ConfigProto(
    intra_op_parallelism_threads=5,
    inter_op_parallelism_threads=5,
    allow_soft_placement=False,
    log_device_placement=True,
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1))

net = ModelManager()
if len(glob.glob(os.path.join(args.output_dir, "model.meta"))) > 0:
    print("Checkpoint files found in directory: " + args.output_dir)
    logging.info("Checkpoint files found in directory: " + args.output_dir)
    net.load(args.output_dir)
else:
    net.setup(IMAGE_SIZE, args.output_dir, args.norm, args.std)
    net.save(True)

print("Building training data dictionary...")
logging.info("Building training data dictionary...")
dataset_list_file_raw = open(args.input_list, 'r').readlines()
dataset = []
dataset_list_file = []
data_ref = np.zeros((0, 2), dtype=np.int)
root_path = os.path.dirname(args.input_list)
last_min_percent = 0
filtered_count = 0
for i, d in enumerate(dataset_list_file_raw):
    d = d.strip()
    d = os.path.join(root_path, d)
    ds = pickle.load(open(d, "br"))
    ds_size = len(ds.model_stars)

    ref = np.zeros((ds_size, 2), dtype=np.int)
    ref[:, 0] = len(dataset_list_file)
    ref[:, 1] = np.arange(ds_size)

    if args.max_res:
        keep_entries = []
        for j in range(ds_size):
            res = get_entry_res(ds, j)
            if res > args.max_res:
                keep_entries.append(j)
        keep_entries = np.array(keep_entries)
        filtered_count += ds_size - len(keep_entries)

        if len(keep_entries) == 0:
            continue

        ref = ref[keep_entries]
        if args.preload:
            ds = filter_entries(ds, keep_entries)

    data_ref = np.concatenate((data_ref, ref), axis=0)
    dataset_list_file.append(d)
    if args.preload:
        dataset.append(ds)

    percent_finished = round(float(i)/len(dataset_list_file_raw)*100)
    if percent_finished % 10 == 0 and percent_finished > last_min_percent:
        last_min_percent = percent_finished
        logging.info(str(percent_finished)+'%')

print("Training data size", len(data_ref))
print("Discarded count", filtered_count)
logging.info("Training data size " + str(len(data_ref)))
logging.info("Discarded count " + str(filtered_count))
np.random.shuffle(data_ref)
current_idx = 0


def running_mean(x, N):
    avg = np.copy(x)
    count = np.ones(x.shape)

    for i in range(N):
        avg[1+i:-2-i] += x[0:-3-2*i] + x[2+2*i:-1]
        count[1+i:-2-i] += 2

    avg /= count
    return avg


def get_fsc(star_file):
    fsc_index = np.array(star_file['model_class_1']['rlnSpectralIndex'], dtype=np.float)
    fsc_value = np.array(star_file['model_class_1']['rlnGoldStandardFsc'], dtype=np.float)
    fsc_value = running_mean(fsc_value, 6)
    fsc_inter = interpolate.interp1d(fsc_index, fsc_value,
                                     fill_value=(fsc_value[0], fsc_value[-1]),
                                     bounds_error=False)
    return fsc_inter


timing = [[], [], []]

(fiz, fiy, fix) = (IMAGE_SIDE, IMAGE_SIDE, IMAGE_SIDE//2 + 1)
Z, Y, X = np.meshgrid(np.linspace(-fiz // 2, fiz // 2 - 1, fiz),
                      np.linspace(-fiy // 2, fiy // 2 - 1, fiy),
                      np.linspace(0, fix - 1, fix))

R = np.round(np.sqrt(X ** 2 + Y ** 2 + Z ** 2))
R = np.fft.ifftshift(R, axes=(0, 1))


def get_batch(batch_size):
    global current_idx
    b_in = []
    b_out = []
    for b in range(batch_size):
        (ii, oi) = data_ref[current_idx]
        if args.preload:
            ds = dataset[ii]
        else:
            ds = pickle.load(open(dataset_list_file[ii], "br"))

        ts = time.time()

        out_box = ds.ground_truth

        fsc_inter = get_fsc(ds.model_stars[oi])
        out_box_ft = np.fft.rfftn(np.fft.fftshift(out_box))
        fsc = fsc_inter(R)

        timing[0].append(time.time() - ts)
        ts = time.time()

        lp_gt_ft = out_box_ft * 1 / (1+np.exp((R-np.random.uniform(0, 20))/5))
        lp_gt = np.fft.fftshift(np.fft.irfftn(lp_gt_ft))
        mask = np.ones(lp_gt.shape) * np.random.uniform(0, 0.6)
        mask[lp_gt > lp_gt.mean()*np.random.uniform(1, 8)] = 1

        mask_ft = np.fft.rfftn(np.fft.fftshift(mask))
        mask_ft = mask_ft * 1 / (1+np.exp((R-np.random.uniform(0, 10))/5))
        mask = np.fft.fftshift(np.fft.irfftn(mask_ft))
        mask = np.clip(mask, 0, 1)

        rand = np.random.normal(scale=mask)
        rand_ft = np.fft.rfftn(np.fft.fftshift(rand))

        ds.data[oi].seek(0)
        real_in_box_ft = np.load(ds.data[oi])['arr_0']
        in_box_ft *= in_box_ft.shape[0]**2

        diff_box = (np.abs(out_box_ft - real_in_box_ft))
        rand_ft = rand_ft * diff_box * np.random.uniform(0.01, 1)
        in_box_ft = (out_box_ft + rand_ft) * fsc

        mask2 = mask * np.clip(mask + np.random.uniform(0, 0.6), 0, 1)
        in_box = np.fft.fftshift(np.fft.irfftn(in_box_ft)) * mask2
        out_box = np.fft.fftshift(np.fft.irfftn(out_box_ft * fsc))

        timing[1].append(time.time() - ts)
        ts = time.time()

        rot = np.random.randint(24)
        in_box = grid_rot90(in_box, rot)
        out_box = grid_rot90(out_box, rot)

        b_in.append(in_box)
        b_out.append(out_box)

        timing[2].append(time.time() - ts)

        current_idx += 1
        if current_idx >= len(data_ref):
            np.random.shuffle(data_ref)
            current_idx = 0

    b_in = np.resize(np.array(b_in), (batch_size, IMAGE_SIDE, IMAGE_SIDE, IMAGE_SIDE, 1))
    b_out = np.resize(np.array(b_out), (batch_size, IMAGE_SIDE, IMAGE_SIDE, IMAGE_SIDE, 1))

    return b_in, b_out

nbatch_in, nbatch_out = get_batch(args.batch_size)
preprocess_executor = ThreadPoolExecutor(max_workers=1)

for epoch in range(args.epocs):

    ts = time.time()

    (batch_in, batch_out) = (nbatch_in, nbatch_out)
    f = preprocess_executor.submit(get_batch, args.batch_size)

    ts1 = time.time()
    net.train(batch_in, batch_out, args.learning_rate)
    te1 = time.time() - ts1

    ts2 = time.time()
    nbatch_in, nbatch_out = f.result()
    te2 = time.time() - ts2

    te = time.time() - ts

    if epoch == 0:
        print("Training has begun...")

    if (epoch % args.checkpoint_steps == 0 and epoch > 0) or \
            (epoch % 10 == 0 and epoch < 100):
        logging.info("%d\t%1.3f\t%1.3f\t%1.3f" % (epoch, te1, te2, te))
        net.run_summary(nbatch_in, nbatch_out)

    if epoch % args.checkpoint_steps == 0 and epoch > 0:
        net.save()


logging.info(np.mean(np.array(timing), axis=1))
