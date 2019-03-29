#!/usr/bin/env python
# coding: utf-8

# ## HF/LF analysis
# 
# See vsb_analyze-power-line-signals_parallel.py for HF/LF signal division


import csv
import numpy as np
from scipy.interpolate import interp1d
from scipy import fftpack, signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal import sosfilt, sosfreqz
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pyarrow.parquet as pq
import pickle
import re
import random
from random import randint
from concurrent.futures import ProcessPoolExecutor
from collections import Collection
from multiprocessing import Process
import multiprocessing as mp
import psutil
import os
import time
import gc



''' Shift the HF component of the time series'''



period = 0.02
time_step = 0.02 / 800000.
time_vec = np.arange(0, 0.02, time_step)
f_sampling = 1 / time_step
print(f'Sampling Frequency = {f_sampling / 1e6} MHz')
# print (str(50* 800000 /1e6) + ' MHz')


# Read signals (variable x). 

non_decimal = re.compile(r'[^\d-]+')
def data_as_float(data_list):
    data_flt=[]
    for i in data_list:
        j = non_decimal.sub('', i)
        data_flt.append(float(j))
    return data_flt


def create_sig_dict(sig_df, column_name, split_signal_name=True):
    sig_d={}
    data_counts={}
    for index, row in sig_df.iterrows():
        if split_signal_name:
            sig_id = row['signal_name'].split('_')[0]
        else:
            sig_id = row['signal_name']
        datas = row[column_name]
        datas = datas.split(",")
        data_flt = data_as_float(datas)
        sig_d[sig_id]=data_flt
    return sig_d


def create_sig_list(sig_df, column_name, split_signal_name=True):
    sig_list=[]
    data_counts={}
    for index, row in sig_df.iterrows():
        if split_signal_name:
            sig_id = row['signal_name'].split('_')[0]
        else:
            sig_id = row['signal_name']
        datas = row[column_name]
        datas = datas.split(",")
        data_counts[sig_id]=len(datas)
        sig_list.extend(datas)
    return sig_list, data_counts


samples = 800000
def shift_time(signal1: pd.Series, signal2: pd.Series, signal3: pd.Series, n=0.25, positive=True, random = 0.01):
    #defaults to 200000 id shift (Lambda/4), positve is move right, with +- up to 20000 (Lambda/400)
    rand_num = (1/random)*randint(1, 100)
    shift = int((samples*n)+((samples*n)/rand_num))
    if positive:
        start_shifted_sig = samples-shift
        #print(f'--shift_time() start_shifted_sig: {start_shifted_sig}, samples: {samples}')
        chopped1=signal1.iloc[start_shifted_sig : samples]
        leftover1 = signal1.iloc[0 : start_shifted_sig]
        chopped2=signal2.iloc[start_shifted_sig : samples]
        leftover2 = signal2.iloc[0 : start_shifted_sig]
        chopped3=signal3.iloc[start_shifted_sig : samples]
        leftover3 = signal3.iloc[0 : start_shifted_sig]

        shifted_sig1 = np.append(chopped1.values,leftover1.values)
        shifted_sig2 = np.append(chopped2.values, leftover2.values)
        shifted_sig3 = np.append(chopped3.values, leftover3.values)
    else:
        chopped1=signal1.iloc[0 : shift]
        leftover1 = signal1.iloc[shift: samples]
        chopped2=signal2.iloc[0 : shift]
        leftover2 = signal2.iloc[shift: samples]
        chopped3=signal3.iloc[0 : shift]
        leftover3 = signal3.iloc[shift: samples]

        shifted_sig1 = np.append(leftover1.values, chopped1.values)
        shifted_sig2 = np.append(leftover2.values, chopped2.values)
        shifted_sig3 = np.append(leftover3.values, chopped3.values)
    assert len(shifted_sig1)==samples
    return shifted_sig1, shifted_sig2, shifted_sig3

def ampl_adjust(signal, max_change=0.1, random=0.5):
    #i = 0
    adj_sig=[]
    for v in signal:
        p=None
        rand_num = randint(0, 10)*max_change
        rand_chance = randint(0,1/random)
        if rand_chance== 1:
            p=v+(v*(rand_num/10))
        elif rand_chance== 2:
            p=v-(v*(rand_num/10))
        else:
            p=v
        adj_sig.append(p)
    return adj_sig

# ### get 3 signals for each measuement
train_meta = pd.read_csv('../../input/metadata_train.csv')

#first round did bad signals where==1
train_meta_error = train_meta[train_meta.target == 0]

#get id_measurement of errors
train_error_idms=train_meta_error['id_measurement'].unique()
all_phase_error=train_meta.loc[train_meta['id_measurement'].isin(train_error_idms)]
all_phase_error.head()
train_meta_error_3_phase_0=all_phase_error.loc[all_phase_error['phase'] == 0]
train_meta_error_3_phase_0.head()

good_meas_ids = list(train_meta_error_3_phase_0['id_measurement'].unique())

print('reading training data')
start=time.time()
data_bp = '../../data/ML_Data/kaggle/VSB_Power_Line_Fault_Detection/bp_signals/'
df_hf = pq.read_table(data_bp+'train_hf_sig.parquet').to_pandas()
end=time.time()
print(f'train_hf_sig read time: {end-start}')
print(df_hf.head())

def calc_shift(n, j, meas_ids, shift_pos, shift_neg):
    pos_sig_d={}
    neg_sig_d={}
    for id in meas_ids:
        sig_ids = all_phase_error.loc[all_phase_error['id_measurement']==id]['signal_id'].values
        assert len(sig_ids)==3

        sig_hf1 = df_hf[[str(sig_ids[0])]]
        sig_hf2 = df_hf[[str(sig_ids[1])]]
        sig_hf3 = df_hf[[str(sig_ids[2])]]
        #convert to series
        sig_hf1 = sig_hf1.iloc[:,0]
        sig_hf2 = sig_hf2.iloc[:,0]
        sig_hf3 = sig_hf3.iloc[:,0]

        if shift_pos:
            shifted_pos1, shifted_pos2, shifted_pos3 = shift_time(sig_hf1, sig_hf2, sig_hf3, n=n, positive=True, random = 0.01)
            adj_signal_pos1=ampl_adjust(shifted_pos1, max_change=0.1, random=0.5)
            adj_signal_pos2=ampl_adjust(shifted_pos2, max_change=0.1, random=0.5)
            adj_signal_pos3=ampl_adjust(shifted_pos3, max_change=0.1, random=0.5)

            pos_sig_d[str(sig_ids[0])]=adj_signal_pos1
            pos_sig_d[str(sig_ids[1])]=adj_signal_pos2
            pos_sig_d[str(sig_ids[2])]=adj_signal_pos3

        if shift_neg:
            shifted_neg1, shifted_neg2, shifted_neg3 = shift_time(sig_hf1, sig_hf2, sig_hf3, n=n, positive=False, random = 0.01)
            adj_signal_neg1=ampl_adjust(shifted_neg1, max_change=0.1, random=0.5)
            adj_signal_neg2=ampl_adjust(shifted_neg2, max_change=0.1, random=0.5)
            adj_signal_neg3=ampl_adjust(shifted_neg3, max_change=0.1, random=0.5)

            neg_sig_d[str(sig_ids[0])]=adj_signal_neg1
            neg_sig_d[str(sig_ids[1])]=adj_signal_neg2
            neg_sig_d[str(sig_ids[2])]=adj_signal_neg3

        sig_hf1=None
        sig_hf2=None
        sig_hf3=None
        sig_ids=None
    return pos_sig_d, neg_sig_d

def shift_batch(n=0.25, shift_neg=True, shift_pos=True):
    print(f'>>shift_batch() n: {n}, shift_neg: {shift_neg}, shift_pos: {shift_pos}')
    pid = os.getpid()
    py = psutil.Process(pid)
    i=0
    #splitinto n parts as too much memory used
    n_meas_ids=np.array_split(good_meas_ids, 10)
    for j,meas_ids in enumerate(n_meas_ids):
        start=time.time()
        memoryUse = py.memory_info()[0] / 2. ** 30
        print(f'--shift_batch chunk: {j} memory use: {memoryUse}')
        pos_sig_d, neg_sig_d= calc_shift(n, j, meas_ids, shift_pos, shift_neg)
        n_factor = str(n).split('.')[1]
        if shift_pos:
            shift_pos_sig_df = pd.DataFrame.from_dict(pos_sig_d)
            print(f'n: {n}, len(shift_pos_sig_df): {len(shift_pos_sig_df)}')
            shift_pos_sig_df.to_parquet(f'../../data/ML_Data/kaggle/VSB_Power_Line_Fault_Detection/bp_signals/shift_pos_sig_df_{n_factor}_{j}.parquet')
        if shift_neg:
            shift_neg_sig_df = pd.DataFrame.from_dict(neg_sig_d)
            shift_neg_sig_df.to_parquet(f'../../data/ML_Data/kaggle/VSB_Power_Line_Fault_Detection/bp_signals/shift_neg_sig_df_{n_factor}_{j}.parquet')
        
        shift_pos_sig_df = None
        shift_neg_sig_df = None
        gc.collect()
        end=time.time()
        print(f'per sub batch elapsed: {end-start}')

def proc_in_series():
    #for good signals, too memory intensive to run in parallel
    #~6.25 hours per n; ~3.6 days in total`
    for n in [0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]:
        print(f'processing batch pos: {n}')
        start=time.time()
        shift_batch(n, False, True)
        end=time.time()
        print(f'elapsed: {end-start}')
    for n in [0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]:
        print(f'processing batch neg: {n}')
        shift_batch(n, True, False)


# combine these with actuals into new training set
# check ratio we actually need

if __name__ == '__main__':
    #proc_in_series()

    jobs = []
    for n in [0.125, 0.25]:
        p = Process(target=shift_batch, args=[n, False, True])
        jobs.append(p)
        p.start()
        print(f'pos process {n} started total: {len(jobs)}')

    # Wait for all of them to finish
    for p in jobs:
        p.join()

    jobs1 = []
    for n in [0.375]:
        p = Process(target=shift_batch, args=[n, False, True])
        jobs1.append(p)
        p.start()
        print(f'pos process {n} started total: {len(jobs1)}')

    # Wait for all of them to finish
    for p in jobs1:
        p.join()

    jobsb = []
    for nb in [0.50, 0.625]:
        pb = Process(target=shift_batch, args=[nb, False, True])
        jobsb.append(pb)
        pb.start()
        print(f'pos process {nb} started total: {len(jobsb)}')

    for pb in jobsb:
        pb.join()

    jobsb1 = []
    for nb1 in [0.75, 0.875]:
        pb1 = Process(target=shift_batch, args=[nb1, False, True])
        jobsb1.append(pb1)
        pb1.start()
        print(f'pos process {nb1} started total: {len(jobsb1)}')

    for pb1 in jobsb1:
        pb1.join()

    jobsc = []
    for nc in [0.125, 0.25]:
        pc = Process(target=shift_batch, args=[nc, True, False])
        jobsc.append(pc)
        pc.start()
        print(f'neg process {nc} started total: {len(jobsc)}')

    for pc in jobsc:
        pc.join()

    jobsc1 = []
    for nc in [0.375]:
        pc = Process(target=shift_batch, args=[nc, True, False])
        jobsc1.append(pc)
        pc.start()
        print(f'neg process {nc} started total: {len(jobsc1)}')

    for pc in jobsc1:
        pc.join()


    jobsd = []
    for nd in [0.50, 0.625]:
        pd = Process(target=shift_batch, args=[nd, True, False])
        jobsd.append(pd)
        pd.start()
        print(f'neg process {nd} started total: {len(jobsd)}')

    for pd in jobsd:
        pd.join()

    jobse = []
    for ne in [0.75, 0.875]:
        pe = Process(target=shift_batch, args=[ne, True, False])
        jobse.append(pe)
        pe.start()
        print(f'neg process {ne} started total: {len(jobse)}')
