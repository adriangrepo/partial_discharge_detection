
# coding: utf-8

# ## Signal Analysis
# 
# Initial kernel based on Kaggle kernel: 'Analyze Power Line Signal Like a Physicist'
# 
# See vsb_analyze-power-line-signals_parallel.py for HF/LF signal division

# In[2]:


import csv
import numpy as np
from numpy.fft import rfft, rfftfreq, irfft
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


# In[ ]:


period = 0.02
time_step = 0.02 / 800000.
time_vec = np.arange(0, 0.02, time_step)
f_sampling = 1 / time_step
print(f'Sampling Frequency = {f_sampling / 1e6} MHz')
# print (str(50* 800000 /1e6) + ' MHz')


# Read signals (variable x). 
# I would like to thank https://www.kaggle.com/xhlulu/exploring-signal-processing-with-scipy and the host: https://www.kaggle.com/sohier/reading-the-data-with-python

# In[23]:


#xs = pq.read_table('../input/train.parquet', columns=[str(i) for i in range(999)]).to_pandas()
xs = pq.read_table('../../input/train.parquet').to_pandas()
print((xs.shape))
xs.head(2)


# Read labels (variable y). 

# In[69]:


train_meta = pd.read_csv('../../input/metadata_train.csv')
print(train_meta.shape)
train_meta.head(6)
train_meta_good = train_meta[train_meta.target == 0]
train_meta_error = train_meta[train_meta.target == 1]
id_error = train_meta_error.groupby('id_measurement')['target'].count()
id_error.head(6)


'''


train_meta.head()


# ### test set

# In[7]:


xt = pq.read_table('../input/test.parquet').to_pandas()
print((xt.shape))
xt.head(2)


# In[8]:


test_meta = pd.read_csv('../input/metadata_test.csv')


# In[9]:


test_phase_sigs=len(test_meta)/3
test_phase_sigs


# In[10]:


test_meta.head()


# In[11]:


test_idms=test_meta['id_measurement'].unique()


# #### Train data 3 phase signals

# In[51]:


three_phase_sigs=len(train_meta)/3
three_phase_sigs


# In[48]:


train_meta_good.head()


# In[63]:


train_meta_error.head()


# In[66]:


#get id_measurement of errors
train_error_idms=train_meta_error['id_measurement'].unique()


# In[67]:


#get all phases even if only 1 has error
all_phase_error=train_meta.loc[train_meta['id_measurement'].isin(train_error_idms)]


# In[68]:


train_meta_error_3_phase_0=all_phase_error.loc[all_phase_error['phase'] == 0]


# In[70]:


train_meta_error_3_phase_0.head()


# #### good

# In[73]:


train_meta_error_3_phase_ids=train_meta_error_3_phase_0['id_measurement']


# In[75]:


#get id_measurement of goods

all_idms=train_meta['id_measurement'].unique()


# In[89]:


train_good_id_measurement = list(set(all_idms)-set(train_meta_error_3_phase_ids))


# In[90]:


train_good_id_measurement[0]


# #### bad

# In[68]:


train_meta_error_3_phase_0=all_phase_error.loc[all_phase_error['phase'] == 0]


# In[70]:


train_meta_error_3_phase_0.head()


# In[86]:


train_meta_error_id_measurement=train_meta_error_3_phase_0['id_measurement']


# In[87]:


train_meta_error_id_measurement=list(train_meta_error_id_measurement)


# In[91]:


train_meta_error_id_measurement[1]


# ### 1, 2 or 3 faulty phases

# Examine how many phases with one ID were labelled problematic. For 80.4% faulty lines, three phases were all labelled faulty, while one-faulty-phase and two-faulty-phase lines contribute about 10% and 10%, respectively.  

# In[4]:


id_error_c = id_error.astype('category')
print(id_error_c.value_counts())
print(id_error_c.value_counts() / id_error_c.value_counts().sum())
id_error_c.value_counts().plot(kind='bar')


# ### which phases labelled faulty

# In[71]:


phase_error=train_meta_error['phase']
phase_error.value_counts()


# In[72]:


phase_error.value_counts().plot(kind='bar')


# We look into this towards end of this notebook - but phase 1 ('central') phase has less PD's identified

# ### Single Signal Analysis

# In[7]:


# Fetch one signal from xs
idx = 1
sig = xs.iloc[:, idx]
idx_error = 3
sig_error = xs.iloc[:, idx_error]
print(sig.shape)


# In[8]:


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

plt.figure(figsize=(6, 5))
# plt.plot(sample_freq[pos_mask], power[pos_mask])
plt.semilogy(sample_freq[pos_mask], power[pos_mask])
plt.ylim([1e-0, 1e8])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [A.U./Hz]')


# Check that it does indeed correspond to the frequency that we generate
# the signal with
np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.6, 0.3, 0.2])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])


# The inset figure above shows that peak frequency is 50 Hz, as we expected. Note that the frequency step in fft spectrum is 50 Hz, limited by the total duration of the signal. 
# Next we use a slightly different method (signal.periodogram) and plot the power spectrum in log-log scale, which make more sense. The unit in the y axis is different by a fixed factor. It is OK. 

# In[8]:


def plot_ps(sig, f_sampling, label='sig', style='loglog'):
    f, Pxx_den = signal.periodogram(sig, f_sampling)
    if style == 'semilogy':
        plt.semilogy(f, Pxx_den, label=label)
    else:
        plt.loglog(f, Pxx_den, label=label)
    plt.ylim([1e-9, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [A.U./Hz]')
    
plot_ps(sig, f_sampling, 'Good sig')
plot_ps(sig_error, f_sampling, 'Bad sig')
plt.legend(loc='best')
plt.show()
# The horizontal line at 50 Hz is artificial, as the log scale in the x axis cannot show 0 Hz. 


# In[9]:


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
    
plt.rcParams["figure.figsize"] = [18,4] 
plot_ps(sig, f_sampling, 'Good sig')
plt.legend(loc='best')
plt.show()


# In[10]:


plot_ps(sig_error, f_sampling, 'Bad sig')
plt.legend(loc='best')
plt.show()


# In[11]:


def plot_ps_diff(sig, Pxx_den, f_sampling, label='sig', style='loglog'):
    plt.loglog(sig, Pxx_den, label=label)
    plt.ylim([1e-9, 1e-2])
    plt.xlim([0, 20000000.0])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [A.U./Hz]')


# In[12]:


f_g, Pxx_den_g = signal.periodogram(sig, f_sampling)
f_b, Pxx_den_b = signal.periodogram(sig_error, f_sampling)


# In[13]:


g=f_g*Pxx_den_g
b=f_b*Pxx_den_b


# In[14]:


diff=g-b
P_diff = diff/f_g


# In[15]:


plot_ps_diff(f_g, P_diff, f_sampling, 'Difference')


# In[127]:


def bandpassfilter(spec, sample_freq, lowcut, highcut):
    # a digital bandpass filter with a infinite roll off. 
    # Note that we will keep the frequency point right at low cut-off and high cut-off frequencies. 
    #print(f'sample_freq: {sample_freq}')
    spec1 = spec.copy()
    spec1[np.abs(sample_freq) < lowcut] = 0
    spec1[np.abs(sample_freq) > highcut] = 0
    filtered_sig = fftpack.ifft(spec1)
    return filtered_sig


# In[17]:


# Digital filtering
# peak_freq should be 50 Hz. 
# We demonstrated differnt low-pass filtered signals. You can see 10-1000 Hz can capture a lot of low frequency features. 
lowcut, highcut = 10, 100
filtered_sig0 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 10, 300
filtered_sig1 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 10, 1000
filtered_sig2 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)

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
filtered_sig3 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 1000, 40e6
filtered_sig4 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig4, linewidth=3, label='Above 1 kHz')
plt.plot(time_vec, filtered_sig3, linewidth=3, label='1 kHz-1 MHz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')


# In[18]:


# We also demonstrate a band-pass filtered and a high-pass filtered signals. 
lowcut, highcut = 0, 1000
filtered_sig1 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 500, 1e4
filtered_sig2 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 1000, 1e6
filtered_sig3 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 1000, 40e6
filtered_sig4 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig1, label='0-1KHz')
plt.plot(time_vec, filtered_sig2, label='500 Hz-10 KHz')
#plt.plot(time_vec, filtered_sig4, linewidth=3, label='1 KHz-40 MHz')
#plt.plot(time_vec, filtered_sig3, linewidth=3, label='1 kHz-1 MHz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')


# In[ ]:


# We also demonstrate a band-pass filtered and a high-pass filtered signals. 
lowcut, highcut = 1000, 1e6
filtered_sig3 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 1000, 40e6
filtered_sig4 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
plt.figure(figsize=(6, 5))
#plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig4, linewidth=3, label='1 KHz-40 MHz')
plt.plot(time_vec, filtered_sig3, linewidth=3, label='1 kHz-1 MHz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')


# In[ ]:


# We also demonstrate a band-pass filtered and a high-pass filtered signals. 
lowcut, highcut = 1e6, 40e6
filtered_sig3 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 1000, 40e6
filtered_sig4 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
#plt.plot(time_vec, filtered_sig4, linewidth=3, label='1 KHz-40 MHz')
plt.plot(time_vec, filtered_sig3, linewidth=3, label='1 MHz - 40 MHz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')


# In[ ]:


# We also demonstrate a band-pass filtered and a high-pass filtered signals. 
lowcut, highcut = 500, 40e6
filtered_sig3 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
nlog_3 = np.log(filtered_sig3)
plt.figure(figsize=(18,4))
plt.plot(time_vec, nlog_3, linewidth=3, label='500 Hz - 40 MHz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')


# In[ ]:


# We also demonstrate a band-pass filtered and a high-pass filtered signals. 
lowcut, highcut = 40, 2500
filtered_sig3 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
plt.figure(figsize=(18,4))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig3, linewidth=3, label='50 Hz - 500 Hz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')


# In[ ]:


hf_sig = sig-filtered_sig3


# In[ ]:



plt.figure(figsize=(18,4))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, hf_sig, linewidth=3, label='50 Hz - 500 Hz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')


# In[130]:


sample_freq = fftpack.fftfreq(sig.size, d=time_step)


# In[ ]:


sample_freq[:10]


# In[ ]:


sig.size


# ### Frequency domain

# In[18]:


train_set=xs
meta_train=train_meta


# In[15]:


#FFT to filter out HF components and get main signal profile
def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)


# In[16]:


###Filter out low frequencies from the signal to get HF characteristics
def high_pass(s, threshold=1e7):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies < threshold] = 0
    return irfft(fourier)


# In[17]:


def phase_indices(three_phase_sig):
    phase1 = 3*three_phase_sig
    phase2 = 3*three_phase_sig + 1
    phase3 = 3*three_phase_sig + 2
    print(phase1,phase2,phase3)
    return phase1,phase2,phase3


# #### good 3 phase signals

# In[155]:


def print_targets(id_meas):
    targets=train_meta.loc[train_meta['id_measurement']==id_meas]
    print(targets.head())


# In[156]:


good_idm=train_good_id_measurement[5]
print_targets(good_idm)


# In[157]:


#2904 three_phase_sigs  

three_phase_sig = good_idm
p1,p2,p3 = phase_indices(three_phase_sig)


# In[158]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(train_set.iloc[:,p1])
plt.plot(train_set.iloc[:,p2])
plt.plot(train_set.iloc[:,p3])


# In[159]:


lf_signal_1 = low_pass(train_set.iloc[:,p1])
lf_signal_2 = low_pass(train_set.iloc[:,p2])
lf_signal_3 = low_pass(train_set.iloc[:,p3])


# In[160]:


plt.figure(figsize=(10,5))
plt.title('De-noised Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(lf_signal_1)
plt.plot(lf_signal_2)
plt.plot(lf_signal_3)


# In[161]:


plt.figure(figsize=(10,5))
plt.title('Signal %d AbsVal / Target: %d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(np.abs(lf_signal_1))
plt.plot(np.abs(lf_signal_2))
plt.plot(np.abs(lf_signal_3))


# In[162]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target: %d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(lf_signal_1)
plt.plot(lf_signal_2)
plt.plot(lf_signal_3)
plt.plot((np.abs(lf_signal_1)+np.abs(lf_signal_2)+np.abs(lf_signal_3)))
plt.legend(['phase 1','phase 2','phase 3','DC Component'],loc=1)


# In[163]:


hf_signal_1 = high_pass(train_set.iloc[:,p1])
hf_signal_2 = high_pass(train_set.iloc[:,p2])
hf_signal_3 = high_pass(train_set.iloc[:,p3])


# In[164]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(hf_signal_1, alpha=0.8)
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(hf_signal_2, alpha=0.8)
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(hf_signal_3, alpha=0.8)
plt.ylim(-40,40)
plt.show()


# In[165]:


diff_1_2=hf_signal_1-hf_signal_2
diff_2_3=hf_signal_2-hf_signal_3
diff_1_3=hf_signal_1-hf_signal_3


# In[166]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(diff_1_2, alpha=0.8, color='r')
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(diff_2_3, alpha=0.8, color='r')
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(diff_1_3, alpha=0.8, color='r')
plt.ylim(-40,40)
plt.show()


# In[209]:


def plot_n_goods(start=0, n=10):
    for i in range(n):
        id_m=train_good_id_measurement[start+i]
        targets=list(train_meta.loc[train_meta['id_measurement']==id_m]['target'])
        assert 1 not in targets
        p1,p2,p3 = phase_indices(id_m)
        hf_signal_1 = high_pass(train_set.iloc[:,p1])
        hf_signal_2 = high_pass(train_set.iloc[:,p2])
        hf_signal_3 = high_pass(train_set.iloc[:,p3])
        
        lf_signal_1 = low_pass(train_set.iloc[:,p1], threshold=100)
        lf_signal_2 = low_pass(train_set.iloc[:,p2], threshold=100)
        lf_signal_3 = low_pass(train_set.iloc[:,p3], threshold=100)
        
        plt.subplot(311)
        plt.plot(hf_signal_1)
        plt.plot(lf_signal_1)
        plt.ylim(-40,40)
        plt.subplot(312)
        plt.plot(hf_signal_2)
        plt.plot(lf_signal_2)
        plt.ylim(-40,40)
        plt.subplot(313)
        plt.plot(hf_signal_3)
        plt.plot(lf_signal_3)
        plt.ylim(-40,40)
        plt.show()


# In[ ]:


plot_n_goods(start=0, n=100)


# In[ ]:


plot_n_goods(start=100, n=100)


# ### bad 3 phase plots

# In[78]:


#good ids:
#train_good_idms
#bad ids
#train_meta_error_3_phase_sig_ids
len(train_good_idms), len(train_meta_error_3_phase_sig_ids)


# In[81]:


train_meta_error_3_phase_sig_ids[:10]


# In[204]:


bad_id_m=train_meta_error_id_measurement[3]
print_targets(bad_id_m)


# In[205]:


#get a new phase set
bthree_phase_sig = train_meta_error_id_measurement[3]
bp1,bp2,bp3 = phase_indices(bthree_phase_sig)


# In[206]:


bhf_signal_1 = high_pass(train_set.iloc[:,bp1])
bhf_signal_2 = high_pass(train_set.iloc[:,bp2])
bhf_signal_3 = high_pass(train_set.iloc[:,bp3])


# In[207]:


blf_signal_1 = low_pass(train_set.iloc[:,bp1], threshold=100)
blf_signal_2 = low_pass(train_set.iloc[:,bp2], threshold=100)
blf_signal_3 = low_pass(train_set.iloc[:,bp3], threshold=100)


# In[208]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(bhf_signal_1, alpha=0.8)
plt.plot(blf_signal_1, alpha=0.8)
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(bhf_signal_2, alpha=0.8)
plt.plot(blf_signal_2, alpha=0.8)
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(bhf_signal_3, alpha=0.8)
plt.plot(blf_signal_3, alpha=0.8)
plt.ylim(-40,40)
plt.show()


# Interesting that the HF component of all 3 phases is pretty similar

# In[180]:


bdiff_1_2=bhf_signal_1-bhf_signal_2
bdiff_2_3=bhf_signal_2-bhf_signal_3
bdiff_1_3=bhf_signal_1-bhf_signal_3


# In[181]:


btotal_diff = abs(bdiff_1_2)+abs(bdiff_2_3)+abs(bdiff_1_3)


# In[182]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(btotal_diff, alpha=0.8, color='g')
plt.show()


# In[183]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(bdiff_1_2, alpha=0.8, color='r')
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(bdiff_2_3, alpha=0.8, color='r')
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(bdiff_1_3, alpha=0.8, color='r')
plt.ylim(-40,40)
plt.show()


# ### plot lots of bads

# In[212]:


def plot_n_bads(start=0, n=10):
    for i in range(n):
        bad_id_m=train_meta_error_id_measurement[start+i]
        targets=list(train_meta.loc[train_meta['id_measurement']==bad_id_m]['target'])
        assert 1 in targets
        bp1,bp2,bp3 = phase_indices(bad_id_m)
        bhf_signal_1 = high_pass(train_set.iloc[:,bp1])
        bhf_signal_2 = high_pass(train_set.iloc[:,bp2])
        bhf_signal_3 = high_pass(train_set.iloc[:,bp3])
        
        lf_signal_1 = low_pass(train_set.iloc[:,bp1], threshold=100)
        lf_signal_2 = low_pass(train_set.iloc[:,bp2], threshold=100)
        lf_signal_3 = low_pass(train_set.iloc[:,bp3], threshold=100)
        
        plt.subplot(311)
        plt.plot(bhf_signal_1)
        plt.plot(lf_signal_1)
        plt.ylim(-40,40)
        plt.subplot(312)
        plt.plot(bhf_signal_2)
        plt.plot(lf_signal_2)
        plt.ylim(-40,40)
        plt.subplot(313)
        plt.plot(bhf_signal_3)
        plt.plot(lf_signal_3)
        plt.ylim(-40,40)
        plt.show()
        


# In[218]:


plot_n_bads(start=0, n=100)


# In[214]:


#378 379 380 do not look bad (these are idm 126)
list(train_meta.loc[train_meta['id_measurement']==126]['target'])


# In[219]:


plot_n_bads(start=100, n=100)


# In[ ]:


plot_n_bads(start=200, n=100)


# In[ ]:


plot_n_bads(start=300, n=100)


# In[ ]:


plot_n_bads(start=400, n=100)


# Why do HF components of phase 1 generally match phase 2, phase 2 generally weaker? (But in very strong PD cases is higher)

# In[177]:


#get a new phase set
bthree_phase_sig = train_meta_error_id_measurement[3]
bp1,bp2,bp3 = phase_indices(bthree_phase_sig)


# In[178]:


bhf_signal_1 = high_pass(train_set.iloc[:,bp1])
bhf_signal_2 = high_pass(train_set.iloc[:,bp2])
bhf_signal_3 = high_pass(train_set.iloc[:,bp3])


# In[179]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(bhf_signal_1, alpha=0.8)
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(bhf_signal_2, alpha=0.8)
plt.ylim(-40,40)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(bhf_signal_3, alpha=0.8)
plt.ylim(-40,40)
plt.show()


# In[175]:


x = sig
X = fftpack.fft(x,n=400)
freqs = fftpack.fftfreq(n=400,d=2e-2/x.size) 


# In[24]:


plt.plot(x)


# In[11]:


fig, ax = plt.subplots()
ax.set_title('Full Spectrum with Scipy')
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.stem(freqs[1:], np.abs(X)[1:])


# In[12]:


x = high_pass(train_set.iloc[:,p1])
X = fftpack.fft(x,n=400)
freqs = fftpack.fftfreq(n=400,d=2e-2/x.size) 


# ### Test signals
# 
# - want to know if, like in train data the phase 2 signal is different to phase1, and 3

# In[12]:


test_set=xt


# In[19]:


def plot_n_test(start=0, n=10):
    for i in range(n):
        id_m=test_idms[start+i]
        p1,p2,p3 = phase_indices(id_m)
        hf_signal_1 = high_pass(test_set.iloc[:,p1])
        hf_signal_2 = high_pass(test_set.iloc[:,p2])
        hf_signal_3 = high_pass(test_set.iloc[:,p3])
        
        lf_signal_1 = low_pass(test_set.iloc[:,p1], threshold=100)
        lf_signal_2 = low_pass(test_set.iloc[:,p2], threshold=100)
        lf_signal_3 = low_pass(test_set.iloc[:,p3], threshold=100)
        
        plt.subplot(311)
        plt.plot(hf_signal_1)
        plt.plot(lf_signal_1)
        plt.ylim(-40,40)
        plt.subplot(312)
        plt.plot(hf_signal_2)
        plt.plot(lf_signal_2)
        plt.ylim(-40,40)
        plt.subplot(313)
        plt.plot(hf_signal_3)
        plt.plot(lf_signal_3)
        plt.ylim(-40,40)
        plt.show()


# In[21]:


plot_n_test(start=0, n=100)


# List the training data, the second phase is very different to phase1 and 2. Is this due to phase rotation (out of plane)?

# In[117]:
'''

cols = list(xs)
len(cols), cols[:10]


# In[157]:


sig_ampls_0p_s = {}
sig_ampls_0n_s = {}
sig_ampls_1p_s = {}
sig_ampls_1n_s = {}
sig_ampls_2p_s = {}
sig_ampls_2n_s = {}
#anomalous where central phase total energy>phase0 and >phase 2
sig_ampls_0p_a = {}
sig_ampls_0n_a = {}
sig_ampls_1p_a = {}
sig_ampls_1n_a = {}
sig_ampls_2p_a = {}
sig_ampls_2n_a = {}
sig_ampls = {}
j=0
for sig_id in cols:
    row=train_meta.loc[train_meta['signal_id']==int(sig_id)]
    phase = row.iloc[0]['phase']
    assert 0<=phase<=2
    
    sig_amp=[]
    sig_amp0p=[]
    sig_amp1p=[]
    sig_amp2p=[]
    sig_amp0n=[]
    sig_amp1n=[]
    sig_amp2n=[]
    sig_hf = xs[sig_id]
    for i, v in sig_hf.iteritems():
        sig_amp.append(abs(v))
        if v>0:
            if phase==0:
                sig_amp0p.append(v)
            elif phase==1:
                sig_amp1p.append(v)
            elif phase==2:
                sig_amp2p.append(v)
            else:
                raise ValueError
        else:
            if phase==0:
                sig_amp0n.append(abs(v))
            elif phase==1:
                sig_amp1n.append(abs(v))
            elif phase==2:
                sig_amp2n.append(abs(v))
            else:
                raise ValueError
    sig_ampls[str(sig_id)] = sum(sig_amp)
    sum_sig_amp0p=sum(sig_amp0p)
    sum_sig_amp1p=sum(sig_amp1p)
    sum_sig_amp2p=sum(sig_amp2p)
    sum_sig_amp0n=sum(sig_amp0n)
    sum_sig_amp1n=sum(sig_amp1n)
    sum_sig_amp2n=sum(sig_amp2n)

    if (sum_sig_amp0p < sum_sig_amp1p) and (sum_sig_amp2p < sum_sig_amp1p):
        #anomalous
        print('anomalous p')
        sig_ampls_0p_a[str(sig_id)] = sum_sig_amp0p
        sig_ampls_1p_a[str(sig_id)] = sum_sig_amp1p
        sig_ampls_2p_a[str(sig_id)] = sum_sig_amp2p
        sig_ampls_0n_a[str(sig_id)] = sum_sig_amp0n
        sig_ampls_1n_a[str(sig_id)] = sum_sig_amp1n
        sig_ampls_2n_a[str(sig_id)] = sum_sig_amp2n
    elif (sum_sig_amp0n < sum_sig_amp1n) and (sum_sig_amp2n < sum_sig_amp1n):
        print('anomalous n')
        sig_ampls_0p_a[str(sig_id)] = sum_sig_amp0p
        sig_ampls_1p_a[str(sig_id)] = sum_sig_amp1p
        sig_ampls_2p_a[str(sig_id)] = sum_sig_amp2p
        sig_ampls_0n_a[str(sig_id)] = sum_sig_amp0n
        sig_ampls_1n_a[str(sig_id)] = sum_sig_amp1n
        sig_ampls_2n_a[str(sig_id)] = sum_sig_amp2n
    else:
        print('standard')
        sig_ampls_0p_s[str(sig_id)] = sum_sig_amp0p
        sig_ampls_1p_s[str(sig_id)] = sum_sig_amp1p
        sig_ampls_2p_s[str(sig_id)] = sum_sig_amp2p
        sig_ampls_0n_s[str(sig_id)] = sum_sig_amp0n
        sig_ampls_1n_s[str(sig_id)] = sum_sig_amp1n
        sig_ampls_2n_s[str(sig_id)] = sum_sig_amp2n
    #print(sig_ampls[str(sig_id)])
    #print(sig_ampls_2n[str(sig_id)])
    #print(shit_pos_sig_df.head(n=10))
    
    j+=1
    if j>50:
        break
sum_sig_df = pd.DataFrame([sig_ampls, sig_ampls_0p_s, sig_ampls_1p_s, sig_ampls_2p_s, sig_ampls_0n_s, sig_ampls_1n_s, sig_ampls_2n_s,
                          sig_ampls_0p_a, sig_ampls_1p_a, sig_ampls_2p_a, sig_ampls_0n_a, sig_ampls_1n_a, sig_ampls_2n_a])
sum_sig_df=sum_sig_df.T
sum_sig_df.columns = ['sig_ampl', 'phase_0_p_s', 'phase_1_p_s', 'phase_2_p_s', 'phase_0_n_s', 'phase_1_n_s', 'phase_2_n_s',
                     'phase_0_p_a', 'phase_1_p_a', 'phase_2_p_a', 'phase_0_n_a', 'phase_1_n_a', 'phase_2_n_a']
sum_sig_df.to_parquet(f'../input/bp_signals/sum_sig_df.parquet')


# In[151]:


sum_sig_df = pq.read_table(f'../input/bp_signals/sum_sig_df.parquet').to_pandas()


# In[152]:


#sum_sig_df.reset_index(inplace=True, drop=True)


# In[153]:


sum_sig_df.head(n=20)


# In[154]:


tot_sig_ampl=sum_sig_df['sig_ampl'].sum()
tot_phase_0_p_s=sum_sig_df['phase_0_p_s'].sum()
tot_phase_1_p_s=sum_sig_df['phase_1_p_s'].sum()
tot_phase_2_p_s=sum_sig_df['phase_2_p_s'].sum()
tot_phase_0_n_s=sum_sig_df['phase_0_n_s'].sum()
tot_phase_1_n_s=sum_sig_df['phase_1_n_s'].sum()
tot_phase_2_n_s=sum_sig_df['phase_2_n_s'].sum()
tot_phase_1_p_s, tot_phase_1_n_s


# In[155]:


plt.bar(list(range(0,6)),[tot_phase_0_p_s,tot_phase_1_p_s,tot_phase_2_p_s,tot_phase_0_n_s,tot_phase_1_n_s,tot_phase_2_n_s])
#plt.yscale('log')


# In[148]:


tot_phase_0_p_a=sum_sig_df['phase_0_p_a'].sum()
tot_phase_1_p_a=sum_sig_df['phase_1_p_a'].sum()
tot_phase_2_p_a=sum_sig_df['phase_2_p_a'].sum()
tot_phase_0_n_a=sum_sig_df['phase_0_n_a'].sum()
tot_phase_1_n_a=sum_sig_df['phase_1_n_a'].sum()
tot_phase_2_n_a=sum_sig_df['phase_2_n_a'].sum()


# In[149]:


plt.bar(list(range(0,6)),[tot_phase_0_p_a,tot_phase_1_p_a,tot_phase_2_p_a,tot_phase_0_n_a,tot_phase_1_n_a,tot_phase_2_n_a])

