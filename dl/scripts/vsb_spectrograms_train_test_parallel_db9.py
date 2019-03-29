
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

NUM_CORES = mp.cpu_count() - 4

print(f'mp.cpu_count(): {mp.cpu_count()}, using: {NUM_CORES}')

DATE = datetime.datetime.today().strftime('%Y%m%d')
UID = str(uuid.uuid4())[:8]

print(f'DATE: {DATE}')
print(f'uID: {UID}')

IMG_SIZE = 600

non_decimal = re.compile(r'[^\d.]+')

# ## Data preparation

bp_path = Path('../../input/bp_signals/')
path = Path('../../input/')
test_bp_path = Path('../../data/ML_Data/kaggle/VSB_Power_Line_Fault_Detection/bp_signals/')
train_path = path / 'train_320_hf/'
test_path = path / 'test_320_hf/'

ts_aug_300_path=path / 'train_300_ts_aug/'
ts_aug_600_path=path / 'train_600_ts_aug/'

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


def read_parquet(file_name):
    df = pq.read_pandas(bp_path / f'{file_name}.parquet').to_pandas()
    df = df.T
    return df


def read_test_parquet(file_name):
    df = pq.read_pandas(test_bp_path / f'{file_name}.parquet').to_pandas()
    df = df.T
    return df


def y_to_sprectrogram(y, i, is_test, out_pixels, descr, out_path, vmin, vmax, plot_labels=False):
    M = 1024
    dpi = 100

    freqs, times, Sx = signal.spectrogram(y, fs=sr, window='hanning',
                                          nperseg=1024, noverlap=M - 100,
                                          detrend=False, scaling='spectrum')
    f, ax = plt.subplots(figsize=(out_pixels / dpi, out_pixels / dpi), dpi=dpi)
    #print(f'Sx.shape: {Sx.shape}, max freq:{max(freqs)}, min freqs:{min(freqs)}, max Sx log: ' \
    #      f'{max(10 * np.log10(Sx[0]))}, min Sx log: {min(10 * np.log10(Sx[0]))}')
    # TODO set colour to use max/min limit
    # Sx : ndarray; Spectrogram of x. By default, the last axis of Sx
    # corresponds to the segment times.
    # print(
    # f'Sx.shape: {Sx.shape}, max:{max(Sx[0])}, max log: {max(10 *
    # np.log10(Sx[0]))}, min log: {min(10 * np.log10(Sx[0]))}')
    #-110 to 20 fr initial round
    vmin = vmin
    vmax = vmax
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
        plt.savefig(out_path / f'{i}{descr}.jpg', dpi=dpi)
        plt.close()
    else:
        plt.savefig(out_path / f'{i}{descr}.jpg', dpi=dpi)
        plt.close()

def proc_chunk(df, is_test, out_pixels, descr, out_path, vmin, vmax, plot_labels=False):
    assert isinstance(df, pd.DataFrame)
    for index, y in df.iterrows():
        assert len(y) == samples
        y_to_sprectrogram(y.values, index, is_test, out_pixels, descr, out_path, vmin, vmax, plot_labels)

def similar_lf_phased_sig(sig_id):
    '''find a different signal but in same LF phase window'''
    alt_sig = None
    for k,v in windows_dict.items():
        assert len(v)>0
        for sig in v:
            j = non_decimal.sub('', sig)
            if str(sig_id)==str(j):
                #return original signal sometimes
                alt_sig = random.choice(v)
                alt_sig = non_decimal.sub('', alt_sig)
    assert alt_sig is not None
    return int(alt_sig)

def recombine(lf_df, y, flip, index):
    '''recombine the HF component with a similar phase LF component'''
    if flip:
        v = (y.values) * -1
    else:
        v = y.values
    alt_sig=similar_lf_phased_sig(index)
    y_lf = lf_df.iloc[[alt_sig]]
    y_lf = y_lf.T.values.flatten()
    assert len(v) == samples
    assert len(y_lf) == samples
    v=y_lf+v
    return v

def aug_proc(lf_df, n, polarity, flip, is_test, out_pixels, out_path, plot_labels):
    if polarity=='neg':
        df = read_parquet(f'shift_neg_sig_df_{n}')
        descr = 'neg_aug_'+n
    else:
        df = read_parquet(f'shift_pos_sig_df_{n}')
        descr = 'pos_aug_'+n
    for index, y in df.iterrows():
        v = recombine(lf_df, y, flip, index)
        y_to_sprectrogram(v, index, is_test, out_pixels, descr, out_path, plot_labels)

def get_lf_components():
    '''keep only those we need to free memory'''
    df_lf = pq.read_table(bp_path/'train_lf_sig.parquet').to_pandas()
    df_lf=df_lf.T
    bad_sigs=get_any_bad()
    #note we use iloc
    df = df_lf.iloc[bad_sigs]
    print(f'{len(df)} signals out of {len(list(df_lf.T))} where at lease 1 phase is bad')
    df_lf=None
    return df

def get_any_bad():
    '''returns list of signal id's where any of the 3 phases are bad'''
    train_meta = pd.read_csv(path/'metadata_train.csv')
    train_meta_error = train_meta[train_meta.target == 1]
    train_error_idms = train_meta_error['id_measurement'].unique()
    all_phase_error = train_meta.loc[train_meta['id_measurement'].isin(train_error_idms)]
    return all_phase_error['signal_id'].values

if __name__ == '__main__':
    # meta_train, features, meta_test = read_meta()

    jobs = []

    # train data
    df_train = pq.read_pandas(path /'train.parquet').to_pandas()
    df_train = df_train.T
    idxs = df_train.index.values
    chunked = np.array_split(idxs, 4)
    descr=''
    vminl=[-110, -80, -70]
    vmaxl=[20,20,20]
    for vmin, vmax in zip(vminl, vmaxl):
        out_path = path / f'train_600_ts_viridis{vmin}-{vmax}/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i, chunk in enumerate(chunked):
            # chunk = [ int(x) for x in chunk ]
            part_df = df_train.loc[df_train.index.isin(chunk)]
            assert (len(part_df) > 0)
            p = Process(target=proc_chunk, args=[part_df, False, IMG_SIZE, descr, out_path, vmin, vmax])
            jobs.append(p)
            p.start()
            print(f' process {i} (train) started total: {len(jobs)}')

    # Wait for all of them to finish
    for p in jobs:
        p.join()

    #Test
    tjobs = []
    df_test = pq.read_pandas(path / f'test.parquet').to_pandas()
    df_test = df_test.T
    idxs = df_test.index.values
    tchunked = np.array_split(idxs, 4)
    tdescr=''
    for vmin, vmax in zip(vminl, vmaxl):
        tout_path=path/f'test_600_ts_viridis{vmin}-{vmax}/'
        if not os.path.exists(tout_path):
            os.makedirs(tout_path)
        for i, tchunk in enumerate(tchunked[:2]):
            # chunk = [ int(x) for x in chunk ]
            part_tdf = df_test.loc[df_test.index.isin(tchunk)]
            assert (len(part_tdf) > 0)
            p = Process(target=proc_chunk, args=[part_tdf, True, IMG_SIZE, tdescr, tout_path, vmin, vmax])
            tjobs.append(p)
            p.start()
            print(f' process {i} (test) started total: {len(jobs)}')
        for j, jtchunk in enumerate(tchunked[2:]):
            # chunk = [ int(x) for x in chunk ]
            part_jtdf = df_test.loc[df_test.index.isin(jtchunk)]
            assert (len(part_jtdf) > 0)
            p = Process(target=proc_chunk, args=[part_jtdf, True, IMG_SIZE, tdescr, tout_path, vmin, vmax])
            tjobs.append(p)
            p.start()
            print(f' process {i} (test) started total: {len(jobs)}')

