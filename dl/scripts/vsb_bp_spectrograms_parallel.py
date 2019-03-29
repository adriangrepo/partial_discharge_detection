
# coding: utf-8

import logging
import time
import datetime
import uuid
import pyageng
from scipy import fftpack, signal
from skimage import util
import copy
from multiprocessing import Process
import multiprocessing as mp

import pyarrow.parquet as pq
import os
import seaborn as sns
import numpy.fft as fft

import matplotlib
import matplotlib.pyplot as plt

from fastai import *
from fastai.tabular import *
from fastai.utils import *

import fastai
print(fastai.__version__)


torch.cuda.set_device(1)

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


NUM_CORES = (mp.cpu_count() / 2) - 7
IMG_SIZE = 300
print(f'mp.cpu_count(): {mp.cpu_count()}, using: {NUM_CORES*2}')

DATE = datetime.datetime.today().strftime('%Y%m%d')
UID = str(uuid.uuid4())[:8]

print(f'DATE: {DATE}')
print(f'uID: {UID}')


# ## Data preparation

path = Path('../../input/')
train_path = path / 'train_300_bp_500Hz-40MHz/'
test_path = path / 'test_300_bp_500Hz-40MHz/'

samples = 800000

period = 0.02
time_step = 0.02 / 800000.
time_vec = np.arange(0, 0.02, time_step)
f_sampling = 1 / time_step
print(f'Sampling Frequency = {f_sampling / 1e6} MHz')

# 800000 samples every 20 millisecond

total_time_sec = 20 / 1000
sr = total_time_sec / samples

sample_freq = fftpack.fftfreq(samples, d=time_step)


def read_meta():
    meta_train = pd.read_csv(path / 'metadata_train.csv')
    features = meta_train.columns
    meta_test = pd.read_csv(path / 'metadata_test.csv')
    return meta_train, features, meta_test


def read_parquet():
    start_time = time.time()
    train_df = pq.read_pandas(path / 'train.parquet').to_pandas()
    end_time = time.time()
    print(f'loading took {end_time-start_time} secs')
    train_df = train_df.T
    print(f'train_df.shape: {train_df.shape}')

    start_time = time.time()
    test_df = pq.read_pandas(path / 'test.parquet').to_pandas()
    end_time = time.time()
    print(f'loading took {end_time-start_time} secs')
    test_df = test_df.T
    print(f'test_df.shape: {test_df.shape}')

    return train_df, test_df


def y_to_spec_sk(y):
    # For each slice, calculate the DFT, which returns both positive and negative frequencies
    #(more on that in “Frequencies and Their Ordering”), so we slice out the positive M2 frequencies for now.
    M = 1024

    slices = util.view_as_windows(y, window_shape=(M,), step=100)
    #print(f'y shape: {y.shape}, Sliced y shape: {slices.shape}')
    win = np.hanning(M + 1)[:-1]
    slices = slices * win
    slices = slices.T
    #print('Shape of `slices`:', slices.shape)
    spectrum = np.fft.fft(slices, axis=0)[:M // 2 + 1:-1]
    spectrum = np.abs(spectrum)
    f, ax = plt.subplots(figsize=(12, 6))

    S = np.abs(spectrum)
    S = 20 * np.log10(S / np.max(S))

    ax.imshow(S, origin='lower', cmap='viridis',
              extent=(0, len(y), 0, sr / 2 / 1000))
    ax.axis('tight')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_xlabel('Time [s]')


def y_to_sprectrogram(y, i, is_test, out_pixels, plot_labels=False):
    M = 1024
    dpi = 100

    freqs, times, Sx = signal.spectrogram(y, fs=sr, window='hanning',
                                          nperseg=1024, noverlap=M - 100,
                                          detrend=False, scaling='spectrum')
    f, ax = plt.subplots(figsize=(out_pixels / dpi, out_pixels / dpi), dpi=dpi)
    # TODO set colour to use max/min limit
    # Sx : ndarray; Spectrogram of x. By default, the last axis of Sx
    # corresponds to the segment times.

    # print(
    # f'Sx.shape: {Sx.shape}, max:{max(Sx[0])}, max log: {max(10 *
    # np.log10(Sx[0]))}, min log: {min(10 * np.log10(Sx[0]))}')
    vmin = -110
    vmax = 10
    ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx),
                  cmap='viridis', vmin=vmin, vmax=vmax)
    if plot_labels:
        ax.set_ylabel('Frequency [kHz]')
        ax.set_xlabel('Time [s]')
    else:
        plt.ioff()
        plt.axis('off')
    ax.set_ylim(top=1.2499999999999999e-11)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    if is_test:
        plt.savefig(test_path / f'{i}.jpg', dpi=dpi)
        plt.close()
    else:
        plt.savefig(train_path / f'{i}.jpg', dpi=dpi)
        plt.close()


def y_bp_to_sprectrogram(y, i, is_test, out_pixels, plot_labels=False):
    M = 1024
    dpi = 100

    freqs, times, Sx = signal.spectrogram(y, fs=sr, window='hanning',
                                          nperseg=1024, noverlap=M - 100,
                                          detrend=False, scaling='spectrum')
    f, ax = plt.subplots(figsize=(out_pixels / dpi, out_pixels / dpi), dpi=dpi)
    # TODO set colour to use max/min limit
    # Sx : ndarray; Spectrogram of x. By default, the last axis of Sx corresponds to the segment times.
    #print(f'Sx.shape: {Sx.shape}, max freq:{max(freqs)}, min freqs:{min(freqs)}, max Sx log: {max(10 * np.log10(Sx[0]))}, min Sx log: {min(10 * np.log10(Sx[0]))}')
    # vmin=-110
    # vmax=30
    vmin = -110
    vmax = -50
    ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx),
                  cmap='viridis', vmin=vmin, vmax=vmax)
    if plot_labels:
        ax.set_ylabel('Frequency [kHz]')
        ax.set_xlabel('Time [s]')
    else:
        plt.ioff()
        plt.axis('off')
    ax.set_ylim(bottom=-1.25e-11, top=0)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    if is_test:
        plt.savefig(test_path / f'{i}.jpg', dpi=dpi)
        plt.close()
    else:
        plt.savefig(train_path / f'{i}.jpg', dpi=dpi)
        plt.close()


def bandpassfilter(spec, sample_freq, lowcut, highcut):
    # a digital bandpass filter with a infinite roll off.
    # Note that we will keep the frequency point right at low cut-off and high
    # cut-off frequencies.
    spec1 = spec.copy()
    spec1[np.abs(sample_freq) < lowcut] = 0
    spec1[np.abs(sample_freq) > highcut] = 0
    filtered_sig = fftpack.ifft(spec1)
    return filtered_sig


def proc_chunk(df, is_test, out_pixels, plot_labels=False):
    assert isinstance(df, pd.DataFrame)
    for index, y in df.iterrows():
        assert len(y) == samples
        y_to_sprectrogram(y.values, index, is_test, out_pixels, plot_labels)


def proc_bp_chunk(df, is_test, out_pixels, lowcut, highcut, plot_labels=False):
    assert isinstance(df, pd.DataFrame)
    for index, y in df.iterrows():
        assert len(y) == samples
        filtered_sig = bandpassfilter(y, sample_freq, lowcut, highcut)
        y_bp_to_sprectrogram(filtered_sig, index, is_test,
                             out_pixels, plot_labels)


if __name__ == '__main__':
    #meta_train, features, meta_test = read_meta()
    train_df, test_df = read_parquet()
    print(f'test_df.shape: {test_df.shape}, cols: {len(list(test_df))}')
    jobs = []
    bp = True
    lowcut, highcut = 500, 40e6
    idxs = test_df.index.values
    #idxs = [ int(x) for x in idxs ]
    chunked = np.array_split(idxs, NUM_CORES)
    for i, chunk in enumerate(chunked):
        #chunk = [ int(x) for x in chunk ]
        part_df = test_df.loc[test_df.index.isin(chunk)]
        assert(len(part_df) > 0)
        if bp:
            p = Process(target=proc_bp_chunk, args=[
                        part_df, True, IMG_SIZE, lowcut, highcut])
        else:
            p = Process(target=proc_chunk, args=[part_df, True, IMG_SIZE])
        jobs.append(p)
        p.start()
        print(f'test process {i} started total: {len(jobs)}')

    # print(train_df.head(n=2))
    # create_train_spectrograms(train_df)
    idts = train_df.index.values
    #idts = [ int(x) for x in idts ]
    print(f'min idx: {min(idts)}, max: {max(idts)}, len: {len(idts)}')
    ckd = np.array_split(idts, NUM_CORES)
    for i, ck in enumerate(ckd):
        #ck = [ int(x) for x in ck ]
        pt_df = train_df.loc[train_df.index.isin(ck)]
        assert(len(part_df) > 0)
        if bp:
            p = Process(target=proc_bp_chunk, args=[
                        part_df, False, IMG_SIZE, lowcut, highcut])
        else:
            p = Process(target=proc_chunk, args=[part_df, False, IMG_SIZE])
        jobs.append(p)
        p.start()
        print(f'train process {i} started total: {len(jobs)}')
