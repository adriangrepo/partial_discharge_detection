
# coding: utf-8

import logging
import time
import datetime
import uuid
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

''' Combine the HF shifted component of the time series with a similarly phased LF component, and convert to spectrogram'''

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

NUM_CORES = (mp.cpu_count() / 2) - 8

print(f'mp.cpu_count(): {mp.cpu_count()}, using: {NUM_CORES*2}')

DATE = datetime.datetime.today().strftime('%Y%m%d')
UID = str(uuid.uuid4())[:8]

print(f'DATE: {DATE}')
print(f'uID: {UID}')

IMG_SIZE = 600

non_decimal = re.compile(r'[^\d.]+')

# ## Data preparation

data_path= Path('../../data/ML_Data/kaggle/VSB_Power_Line_Fault_Detection/bp_signals/')
bp_path = Path('../../input/bp_signals/')
path = Path('../../input/')
test_bp_path = Path('../../data/ML_Data/kaggle/VSB_Power_Line_Fault_Detection/bp_signals/')
train_path = path / 'train_320_hf/'
test_path = path / 'test_320_hf/'

#ts_aug_300_path=path / 'train_300_ts_aug_viridis-70_20/'
ts_aug_600_path=path / 'train_600_ts_aug_viridis-70-20/'
signal_path=path / 'signal_analysis/good_signals/'
bp_good_path=Path('../../data/ML_Data/kaggle/VSB_Power_Line_Fault_Detection/bp_signals/')
signal_out_path=Path('../../data/ML_Data/kaggle/VSB_Power_Line_Fault_Detection/signal_analysis/good_signals/')

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

def get_sig_sorted(sig_windows_d):
    sig_vals = list(sig_windows_d.values())
    sig_vals = [item for sublist in sig_vals for item in sublist]
    # sorts in place
    sig_vals.sort()
    return sig_vals

def check_sig_windows(sig_windows_d, meta_train, target=0):
    sig_vals = get_sig_sorted(sig_windows_d)
    type_sigs = meta_train.loc[meta_train['target'] == target]['signal_id'].values.tolist()
    print(f'{len(type_sigs)} target signals in training data, windowed: {len(sig_vals)}')
    print(f'type_sigs[0]: {type_sigs[0]}, sig_vals[0]: {sig_vals[0]}')
    assert sig_vals == type_sigs

def any_sig_bad(meta_train):
    # get id_measurement of errors
    bad_ids = meta_train.loc[meta_train['target'] == 1]['id_measurement'].values.tolist()
    # get all phases even if only 1 has error
    all_sig_ds = meta_train.loc[meta_train['id_measurement'].isin(bad_ids)]['signal_id']
    return all_sig_ds


def get_windows(file_path='signal_analysis/bad_signals/sig_80_windows.csv'):
    '''find other LF signals of similar phase
    returns dict of window number, and signals within that window
    see vsb_signal_analysis_hf_lf_vlf..ipynb'''
    sig_windows_d = {}
    windows_dict = {}
    with open(path/file_path, 'r') as f:
        reader = csv.reader(f)
        for k, v in reader:
            windows_dict[k] = v
    for k,v in windows_dict.items():
        slig_l=[]
        assert len(v)>0
        vs = v.replace(']', '')
        vs = vs.replace('[', '')
        vs = vs.split()
        for sig in vs:
            j = non_decimal.sub('', sig)
            slig_l.append(int(j))
        sig_windows_d[k]=slig_l
    return sig_windows_d


def read_meta():
    meta_train = pd.read_csv(path / 'metadata_train.csv')
    features = meta_train.columns
    meta_test = pd.read_csv(path / 'metadata_test.csv')
    return meta_train, features, meta_test


def read_parquet(file_name):
    df = pq.read_pandas(data_path / f'{file_name}.parquet').to_pandas()
    df = df.T
    return df


def read_test_parquet(file_name):
    df = pq.read_pandas(test_bp_path / f'{file_name}.parquet').to_pandas()
    df = df.T
    return df


def y_to_sprectrogram(y, i, is_test, out_pixels, descr, out_path, plot_labels=False):
    #print(f'y type: {type(y)}, shape: {y.shape}, [:10] {y[:10]}')
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
    #vmin = -60
    #vmax = 10
    vmin = -70
    vmax = 20
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
        plt.savefig(out_path / f'{i}_{descr}.jpg', dpi=dpi)
        plt.close()
    else:
        plt.savefig(out_path / f'{i}_{descr}.jpg', dpi=dpi)
        plt.close()

def proc_chunk(df, is_test, out_pixels, plot_labels=False):
    assert isinstance(df, pd.DataFrame)
    for index, y in df.iterrows():
        assert len(y) == samples
        y_to_sprectrogram(y.values, index, is_test, out_pixels, plot_labels)

def similar_lf_phased_sig(sig_id, sig_windows_d):
    '''find a different signal but in same LF phase window'''
    alt_sig = None
    for k,v in sig_windows_d.items():
        if int(sig_id) in v:
            alt_sig = random.choice(v)
        #for sig in v:
        #    if str(sig_id)==str(sig):
                #return at random other or original signal
        #        alt_sig = random.choice(v)
    if alt_sig is None:
        print(f'--similar_lf_phased_sig() sig_id: {sig_id} not found')
    return alt_sig

def recombine(lf_df, y, flip, index, sig_windows_d,):
    '''recombine the HF component with a similar phase LF component'''
    if flip:
        v = (y.values) * -1
    else:
        v = y.values
    alt_sig =similar_lf_phased_sig(index, sig_windows_d)
    if alt_sig is not None:
        alt_sig = str(alt_sig)
        idxs = list(lf_df.index.values)
        if alt_sig in idxs:
            y_lf = lf_df.loc[[alt_sig]]
            assert y_lf is not None
            y_lf = y_lf.T.values.flatten()
            v = y_lf + v
            assert len(v) == samples
        else:
            return None
    else:
        return None

    return v

def aug_proc(lf_df, n, polarity, flip, is_test, out_pixels, out_path, sig_windows_d, plot_labels):
    if polarity=='neg':
        df = read_parquet(f'shift_neg_sig_df_{n}')
        print(f'neg df: {len(df)}')
        descr = 'neg_aug_'+n
    else:
        df = read_parquet(f'shift_pos_sig_df_{n}')
        print(f'pos df: {len(df)}')
        descr = 'pos_aug_'+n
    for index, y in df.iterrows():
        #index of lf_df is signal_id
        v = recombine(lf_df, y, flip, index, sig_windows_d)
        if v is not None:
            y_to_sprectrogram(v, index, is_test, out_pixels, descr, out_path, plot_labels)

def ts_signals(lf_df_name, lf_df, n, polarity, flip, out_path, sig_windows_d):
    if polarity=='neg':
        df = read_parquet(f'shift_neg_sig_df_{n}')
        print(f'neg df: {len(df)}')
        descr = 'neg_aug_'+n
    else:
        df = read_parquet(f'shift_pos_sig_df_{n}')
        print(f'pos df: {len(df)}')
        descr = 'pos_aug_'+n
    ts_sigs = {}
    i=0
    for index, y in df.iterrows():
        #index of lf_df is signal_id
        v = recombine(lf_df, y, flip, index, sig_windows_d)
        ts_sigs[index]=v
    df=pd.DataFrame.from_dict(ts_sigs)
    df.to_parquet(out_path/f'ts_{lf_df_name}_{descr}.parquet')

def create_spectrogram(lf_df_name, data_path, n, polarity, is_test, out_pixels, out_path, plot_labels):
    #lf_df_name, n, 'pos', False, signal_path
    print(f'lf_df_name: {lf_df_name}, data_path: {data_path}, n: {n}, polarity: {polarity}, is_test: {is_test}, out_pixels: {out_pixels}, out_path: {out_path}, plot_labels: {plot_labels}')
    descr = polarity+'_aug_'+n
    df_lf = pq.read_table(data_path / f'{lf_df_name}.parquet').to_pandas()
    df_lf = df_lf.T
    for index, y in df_lf.iterrows():
        y=y.astype(float)
        y_to_sprectrogram(y, index, is_test, out_pixels, descr, out_path, plot_labels)

def create_subset():
    '''for testing only'''
    df_lf = pq.read_table(data_path / 'train_lf_sig.parquet').to_pandas()
    df = df_lf.head(1000)
    df.to_parquet(data_path / 'train_lf_sig_1000sampl_testing.parquet')
    print('done')

def get_lf_components(meta_train):
    '''keep only those we need to free memory'''
    df_lf = pq.read_table(bp_path/'train_lf_sig.parquet').to_pandas()
    #NB length is 800000 - so take Transpose to put 1 signal per column
    df_lf=df_lf.T
    g_sigs=get_any_good()
    #note we use iloc
    df = df_lf.iloc[g_sigs]
    print(f'{len(df)} g signals out of {len(list(df_lf.T))} ')
    df_lf=None
    #other_phase_bad = any_sig_bad(meta_train)
    #df_o = df_lf.iloc[other_phase_bad]
    #print(f'{len(df_o)} good signals where another phase bad')
    return df

def get_any_good():
    '''returns list of signal id's where any of the 3 phases are bad'''
    train_meta = pd.read_csv(path/'metadata_train.csv')
    train_meta_g = train_meta[train_meta.target == 0]
    train_g_idms = train_meta_g['id_measurement'].unique()
    all_phase_g = train_meta.loc[train_meta['id_measurement'].isin(train_g_idms)]
    print(f'<<get_any_good: {len(all_phase_g)}')
    return all_phase_g['signal_id'].values

def get_any_bad():
    '''returns list of signal id's where any of the 3 phases are bad'''
    train_meta = pd.read_csv(path/'metadata_train.csv')
    train_meta_error = train_meta[train_meta.target == 1]
    train_error_idms = train_meta_error['id_measurement'].unique()
    all_phase_error = train_meta.loc[train_meta['id_measurement'].isin(train_error_idms)]
    print(f'<<get_any_bad: {len(all_phase_error)}')
    return all_phase_error['signal_id'].values

'''
#ts_shift only
if __name__ == '__main__':
    jobs = []
    meta_train, features, meta_test = read_meta()
    sig_windows_d = get_windows(file_path='signal_analysis/good_signals/sig_good_80_windows.csv')
    check_sig_windows(sig_windows_d, meta_train, target=0)

    lf_df = get_lf_components(meta_train)
    lf_df_name='good_sig_80_wind_'

    #for bad signals using parallel procs is OK, for good, need to run in series
    #for n in ['125', '375', '50','625', '75', '875']:
        #ts_signals(lf_df_name, lf_df, n, 'pos', False, signal_out_path, sig_windows_d)

    for n in ['25', '125', '375']:
        # lf_df_name, lf_df, n, polarity, flip, out_path, sig_windows_d
        ts_signals(lf_df_name, lf_df, n, 'neg', False, signal_out_path, sig_windows_d)

    #Augmented
    for n in ['125', '375', '50']:
        #lf_df_name, lf_df, n, polarity, flip, out_path, sig_windows_d
        p = Process(target=ts_signals, args=[lf_df_name, lf_df, n, 'neg', True, signal_path, sig_windows_d])
        jobs.append(p)
        p.start()
        print(f' process {n} 300 started total: {len(jobs)}')

    #Augmented
    for n in ['625', '75', '875']:
        #in 60_10 series, these were flipped (flip=True)
        #n, polarity, is_test, out_pixels, out_path, plot_labels
        p = Process(target=ts_signals, args=[lf_df_name, lf_df, n, 'neg', True, signal_path, sig_windows_d])
        jobs.append(p)
        p.start()
        print(f' process {n} 300 started total: {len(jobs)}')


'''

#spectrograms only
if __name__ == '__main__':
    print(f'outputting spectrograms to: {ts_aug_600_path}')
    jobs = []

    meta_train, features, meta_test = read_meta()
    sig_windows_d = get_windows(file_path='signal_analysis/good_signals/sig_good_80_windows.csv')
    check_sig_windows(sig_windows_d, meta_train, target=0)

    #lf_df_name='ts_good_sig_80_wind__'

    for n in ['125', '375', '50']:
        #lf_df_name, n, polarity, is_test, out_pixels, out_path, plot_labels
        df_name = 'shift_pos_sig_df_'+n
        p = Process(target=create_spectrogram, args=[df_name, bp_good_path, n, 'pos', False, IMG_SIZE, ts_aug_600_path, False])
        jobs.append(p)
        p.start()
        print(f' process {n} 300 started total: {len(jobs)}')

    # Wait for all of them to finish
    for p in jobs:
        p.join()

    jobs_b = []
    for n in ['625', '75', '875']:
        df_name = 'shift_pos_sig_df_' + n
        p1 = Process(target=create_spectrogram, args=[df_name, bp_good_path, n, 'pos', False, IMG_SIZE, ts_aug_600_path, False])
        jobs_b.append(p1)
        p1.start()
        print(f' process {n} 300 started total: {len(jobs_b)}')

    # Wait for all of them to finish
    for p in jobs_b:
        p.join()

    jobs_c = []
    for n in ['125', '25', '375']:
        #lf_df_name, n, polarity, is_test, out_pixels, out_path, plot_labels
        df_name = 'shift_neg_sig_df_' + n
        p = Process(target=create_spectrogram, args=[df_name, bp_good_path, n, 'neg', False, IMG_SIZE, ts_aug_600_path, False])
        jobs_c.append(p)
        p.start()
        print(f' process {n} 300 started total: {len(jobs_c)}')



'''
#ts shift and spectrograms
if __name__ == '__main__':
    #TODO write out TS first - then sep to conv to spectrograms
    jobs = []
    meta_train, features, meta_test = read_meta()
    #sig_windows_gd = get_windows(file_path='signal_analysis/good_signals/sig_good_80_windows.csv')
    sig_windows_d = get_windows(file_path='signal_analysis/bad_signals/sig_80_windows.csv')
    #check_sig_windows(sig_windows_gd, meta_train, target=0)
    check_sig_windows(sig_windows_d, meta_train, target=1)

    lf_df = get_lf_components(meta_train)
    #Augmented spectrograms
    for n in ['125', '375', '50']:
        #df, n, polarity, flip, is_test, out_pixels, out_path, plot_labels
        p = Process(target=aug_proc, args=[lf_df, n, 'pos', False, True, 300, ts_aug_300_path, sig_windows_d,False])
        jobs.append(p)
        p.start()
        p = Process(target=aug_proc, args=[lf_df, n, 'neg', False, True, 300, ts_aug_300_path, sig_windows_d,False])
        jobs.append(p)
        p.start()
        print(f' process {n} 300 started total: {len(jobs)}')

    #Augmented spectrograms
    for n in ['625', '75', '875']:
        #in 60_10 series, these were flipped (flip=True)
        #n, polarity, is_test, out_pixels, out_path, plot_labels
        p = Process(target=aug_proc, args=[lf_df, n, 'pos', True, True, 300, ts_aug_300_path, sig_windows_d, False])
        jobs.append(p)
        p.start()
        p.join(4000)
        p = Process(target=aug_proc, args=[lf_df, n, 'neg', True, True, 300, ts_aug_300_path, sig_windows_d, False])
        jobs.append(p)
        p.start()
        p.join(4000)
        print(f' process {n} 300 started total: {len(jobs)}')

    #600x600 image size
    for n in ['125', '375', '50']:
        p = Process(target=aug_proc, args=[lf_df, n, 'pos', False, False, 600, ts_aug_600_path, sig_windows_d,False])
        jobs.append(p)
        p.start()
        p.join(8000)
        p = Process(target=aug_proc, args=[lf_df, n, 'neg', False, False, 600, ts_aug_600_path, sig_windows_d, False])
        jobs.append(p)
        p.start()
        p.join(8000)
        print(f' process {n} 600 started total: {len(jobs)}')

    for n in ['625', '75', '875']:
        p = Process(target=aug_proc, args=[lf_df, n, 'pos', True, False, 600, ts_aug_600_path, sig_windows_d,False])
        jobs.append(p)
        p.start()
        p.join(14000)
        p = Process(target=aug_proc, args=[lf_df, n, 'neg', True, False, 600, ts_aug_600_path, sig_windows_d, False])
        jobs.append(p)
        p.start()
        p.join(14000)
        print(f' process {n} 600 started total: {len(jobs)}')
'''

