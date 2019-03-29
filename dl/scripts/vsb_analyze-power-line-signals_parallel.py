
# coding: utf-8

import numpy as np
from scipy import fftpack, signal
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pyarrow.parquet as pq
import pickle
from multiprocessing import Process
import multiprocessing as mp

# NUM_CORES=(mp.cpu_count()/2)-2
# print(f'mp.cpu_count(): {mp.cpu_count()}, using: {NUM_CORES*2}')

period = 0.02
time_step = 0.02 / 800000.
time_vec = np.arange(0, 0.02, time_step)
f_sampling = 1 / time_step
print(f'Sampling Frequency = {f_sampling / 1e6} MHz')
# print (str(50* 800000 /1e6) + ' MHz')


def get_train_data():
    # xs = pq.read_table('../input/train.parquet', columns=[str(i) for i in range(999)]).to_pandas()
    xs = pq.read_pandas('../../input/train.parquet').to_pandas()
    xs = xs.T
    print(xs.shape)
    return xs


def get_test_data():
    xt = pq.read_pandas('../../input/test.parquet').to_pandas()
    xt = xt.T
    print(xt.shape)
    return xt


def get_meta_data():
    # Read labels (variable y). 
    train_meta = pd.read_csv('../../input/metadata_train.csv')
    print(train_meta.shape)
    train_meta.head(6)
    train_meta_good = train_meta[train_meta.target == 0]
    train_meta_error = train_meta[train_meta.target == 1]
    id_error = train_meta_error.groupby('id_measurement')['target'].count()
    print(id_error.head(6))
    return train_meta


def show_bad():
    # Examine how many phases with one ID were labelled problematic. 
    # For 80.4% faulty lines, three phases were all labelled faulty, 
    # while one-faulty-phase and two-faulty-phase lines contribute about 10% and 10%, respectively.  
    id_error_c = id_error.astype('category')
    print(id_error_c.value_counts())
    print(id_error_c.value_counts() / id_error_c.value_counts().sum())
    id_error_c.value_counts().plot(kind='bar')


def get_fft_vars(xs):
    # Fetch one signal from xs
    idx = 1
    sig = xs.iloc[idx, :]
    idx_error = 3
    sig_error = xs.iloc[idx_error, :]
    print(f'sig: {sig.shape}')
    
    # https://www.scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
    # The FFT of the signal
    sig_fft = fftpack.fft(sig)
    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)
    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(sig.size, d=time_step)
    
    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq >= 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]
    print(f'peak_freq: {peak_freq}')
    return sig_fft, power, sample_freq, pos_mask, freqs, peak_freq


def plot_power(pos_mask, sample_freq, power):
    # The inset figure above shows that peak frequency is 50 Hz, as we expected. 
    # Note that the frequency step in fft spectrum is 50 Hz, limited by the total duration of the signal. 

    plt.figure(figsize=(6, 5))
    # plt.plot(sample_freq[pos_mask], power[pos_mask])
    plt.semilogy(sample_freq[pos_mask], power[pos_mask])
    plt.ylim([1e-0, 1e8])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power [A.U./Hz]')
    # Check that it does indeed correspond to the frequency that we generate
    # the signal with
    np.allclose(peak_freq, 1. / period)
    
    # An inner plot to show the peak frequency
    axes = plt.axes([0.55, 0.6, 0.3, 0.2])
    plt.title('Peak frequency')
    plt.plot(freqs[:8], power[:8])
    plt.setp(axes, yticks=[])


def plot_ps(sig, f_sampling, label='sig', style='loglog'):
    # Next we use a slightly different method (signal.periodogram) and plot the power spectrum in log-log scale, 
    # which make more sense. The unit in the y axis is different by a fixed factor. It is OK. 

    f, Pxx_den = signal.periodogram(sig, f_sampling)
    if style == 'semilogy':
        plt.semilogy(f, Pxx_den, label=label)
    else:
        plt.loglog(f, Pxx_den, label=label)
    plt.ylim([1e-9, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [A.U./Hz]')

    
def plot_good_bad_sig(sig):
    # plot_good_bad_sig(sig)
    plot_ps(sig, f_sampling, 'Good sig')
    plot_ps(sig_error, f_sampling, 'Bad sig')
    plt.legend(loc='best')
    plt.show()
    # The horizontal line at 50 Hz is artificial, as the log scale in the x axis cannot show 0 Hz. 


def plot_ps(sig, f_sampling, label='sig', style='loglog'):
    f, Pxx_den = signal.periodogram(sig, f_sampling)
    print(f'{max(f)}, {min(f)}, {max(Pxx_den)}, {min(Pxx_den)}')
    if style == 'semilogy':
        plt.semilogy(f, Pxx_den, label=label)
    else:
        plt.loglog(f, Pxx_den, label=label)
    plt.ylim([1e-9, 1e2])
    plt.xlim([0, 20000000.0])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [A.U./Hz]')

    
def plot_ps_vs_freq():
    plt.rcParams["figure.figsize"] = [18, 4] 
    plot_ps(sig, f_sampling, 'Good sig')
    plt.legend(loc='best')
    plt.show()
    
    plot_ps(sig_error, f_sampling, 'Bad sig')
    plt.legend(loc='best')
    plt.show()


def plot_ps_diff(sig, Pxx_den, f_sampling, label='sig', style='loglog'):
    plt.loglog(sig, Pxx_den, label=label)
    plt.ylim([1e-9, 1e-2])
    plt.xlim([0, 20000000.0])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [A.U./Hz]')


def plot_difference(sig, sig_error):
    f_g, Pxx_den_g = signal.periodogram(sig, f_sampling)
    f_b, Pxx_den_b = signal.periodogram(sig_error, f_sampling)
    g = f_g * Pxx_den_g
    b = f_b * Pxx_den_b
    diff = g - b
    P_diff = diff / f_g
    plot_ps_diff(f_g, P_diff, f_sampling, 'Difference')


def bandpassfilter(spec, sample_freq, lowcut, highcut):
    # a digital bandpass filter with a infinite roll off. 
    # Note that we will keep the frequency point right at low cut-off and high cut-off frequencies. 
    # print(f'>>bandpassfilter() spec.shape: {spec.shape}, sample_freq: {sample_freq.shape}')
    spec1 = spec.copy()
    spec1[np.abs(sample_freq) < lowcut] = 0
    spec1[np.abs(sample_freq) > highcut] = 0
    filtered_sig = fftpack.ifft(spec1)
    return filtered_sig


def digital_filtering(sig_fft):
    # peak_freq should be 50 Hz. 
    # We demonstrated differnt low-pass filtered signals. You can see 10-1000 Hz can capture a lot of low frequency features. 
    lowcut, highcut = 10, 100
    filtered_sig0 = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
    lowcut, highcut = 10, 300
    filtered_sig1 = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
    lowcut, highcut = 10, 1000
    filtered_sig2 = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
    
    plt.figure(figsize=(6, 5))
    plt.plot(time_vec, sig, label='Original signal')
    plt.plot(time_vec, filtered_sig0, linewidth=3, label='10-100 Hz')
    plt.plot(time_vec, filtered_sig1, linewidth=3, label='10-300 Hz')
    plt.plot(time_vec, filtered_sig2, linewidth=3, label='10-1000 Hz')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend(loc='best')
    
    # We also demonstrate a band-pass filtered and a high-pass filtered signals. 
    lowcut, highcut = 1000, 1e6
    filtered_sig3 = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
    lowcut, highcut = 1000, 40e6
    filtered_sig4 = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
    plt.figure(figsize=(6, 5))
    plt.plot(time_vec, sig, label='Original signal')
    plt.plot(time_vec, filtered_sig4, linewidth=3, label='Above 1 kHz')
    plt.plot(time_vec, filtered_sig3, linewidth=3, label='1 kHz-1 MHz')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend(loc='best')


def bp_hf_signals(sig_fft):
    # We also demonstrate a band-pass filtered and a high-pass filtered signals. 
    lowcut, highcut = 0, 1000
    filtered_sig1 = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
    lowcut, highcut = 500, 1e4
    filtered_sig2 = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
    lowcut, highcut = 1000, 1e6
    filtered_sig3 = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
    lowcut, highcut = 1000, 40e6
    filtered_sig4 = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
    plt.figure(figsize=(6, 5))
    plt.plot(time_vec, sig, label='Original signal')
    plt.plot(time_vec, filtered_sig1, label='0-1KHz')
    plt.plot(time_vec, filtered_sig2, label='500 Hz-10 KHz')
    # plt.plot(time_vec, filtered_sig4, linewidth=3, label='1 KHz-40 MHz')
    # plt.plot(time_vec, filtered_sig3, linewidth=3, label='1 kHz-1 MHz')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend(loc='best')


def save_file(df, file_name):
    with open(f'../../input/bp_signals/{file_name}', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sep_hf_lf(xs_chunk, sample_freq, f_prefix, lowcut, highcut):
    # loop over all signals, sep into HF and LF and save to file
    df_hf = {}
    df_lf = {}
    for idx, sig in xs_chunk.iterrows():
        sig_fft = fftpack.fft(sig)
        filtered_sig = bandpassfilter(sig_fft, sample_freq, lowcut, highcut)
        hf_sig = sig - filtered_sig
        # keep only real parts
        df_hf[idx] = hf_sig.real
        df_lf[idx] = filtered_sig.real
    return df_hf, df_lf


def proc_chunk(xs, ck, chunk_num, sample_freq, f_prefix, lowcut, highcut , vlf=False):
    df_hf_all = {}
    df_lf_all = {}
    for sub_c in ck:
        xs_chunk = xs.loc[xs.index.isin(sub_c)]
        assert(len(xs_chunk) > 0)
        df_hf, df_lf = sep_hf_lf(xs_chunk, sample_freq, f_prefix, lowcut, highcut)
        df_hf_all.update(df_hf)
        df_lf_all.update(df_lf)
        
    print(f'saving chunk_num {chunk_num}, len(ck): {len(ck)}')
    columns = [str(x) for x in list(df_hf_all.keys())]
    df_hf = pd.DataFrame(df_hf_all, columns=columns)
    if vlf:
        df_hf.to_parquet(f'../../input/bp_signals/{f_prefix}_vlf_hf_sig_{chunk_num}.parquet')
    else:
        df_hf.to_parquet(f'../../input/bp_signals/{f_prefix}_hf_sig_{chunk_num}.parquet')
    df_lf = pd.DataFrame(df_lf_all, columns=columns)
    if vlf:
        df_lf.to_parquet(f'../../input/bp_signals/{f_prefix}_vlf_lf_sig_{chunk_num}.parquet')
    else:
        df_lf.to_parquet(f'../../input/bp_signals/{f_prefix}_lf_sig_{chunk_num}.parquet')

    
if __name__ == '__main__':
    # not really multiprocessing
    # tricky to balance procs vs ram use so running mostly serially
    jobs = []
    lowcut = 40
    highcut = 1000
    vlf_lowcut = 10
    vlf_highcut = 300
    vlf = True
    '''
    # train
    xs = get_train_data()
    sig_fft, power, sample_freq, pos_mask, freqs, peak_freq = get_fft_vars(xs)
    f_prefix = 'train'
    # divide df into chunks
    ids = xs.index.values 
    print(f'len(ids): {len(ids)}, max: {max(ids)}, min: {min(ids)}')
    num_cols = 100
    ckd = np.array_split(ids, num_cols)
    ck_batch = int(len(ckd) / 10)
    for chunk_num in range(ck_batch):
        start_ck = (chunk_num) * 10
        end_ck = (chunk_num + 1) * 10
        ck = ckd[start_ck:end_ck]
        # print(f'chunk_num: {chunk_num}, ck: {len(ck)}, start_ck: {start_ck}, end_ck: {end_ck}')
        if vlf:
            p = Process(target=proc_chunk, args=[xs, ck, chunk_num, sample_freq, f_prefix, vlf_lowcut, vlf_highcut, vlf])
        else:
            p = Process(target=proc_chunk, args=[xs, ck, chunk_num, sample_freq, f_prefix, lowcut, highcut])
        jobs.append(p)
    '''
    # Test data
    f_prefix = 'test'
    xt = get_test_data()
    sig_fft, power, sample_freq, pos_mask, freqs, peak_freq = get_fft_vars(xt)
    idts = xt.index.values 
    print(f'test data, len(idts): {len(idts)}, max: {max(idts)}, min: {min(idts)}')
    num_cols = 100
    ckn = np.array_split(idts, num_cols)
    ck_b = int(len(ckn) / 10)
    for c_num in range(ck_b):
        if c_num == 6:
            start_ck = (c_num) * 10
            end_ck = (c_num + 1) * 10
            ck = ckn[start_ck:end_ck]
            # print(f'c_num: {c_num}, ck: {len(ck)}, start_ck: {start_ck}, end_ck: {end_ck}')
            if vlf:
                p = Process(target=proc_chunk, args=[xt, ck, c_num, sample_freq, f_prefix, vlf_lowcut, vlf_highcut, vlf])
            else:
                p = Process(target=proc_chunk, args=[xt, ck, c_num, sample_freq, f_prefix, lowcut, highcut])
            jobs.append(p)
        
    # Run processes
    for i, p in enumerate(jobs):
        print(f'starting job: {i}')
        p.start()
        # 60 for train at least 2x that for test
        p.join()
        
    print('All finished')

