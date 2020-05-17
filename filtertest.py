#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:57:13 2019

@author: burakgur
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 1
fs = 13      # sample rate, Hz
cutoff = 0.4  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


#%%
ir = signal.firwin(3, cutoff)
tuning_y = all_mean_data / np.max(all_mean_data)
tuning_x = epoch_freqs

filter_y = np.abs(h)
filter_x = 0.5*fs*w/np.pi

filtered_tuning = np.zeros((len(tuning_x),1))

for i in range(len(tuning_x)):
    curr_filter_response = filter_y[np.argmin(np.abs(filter_x - tuning_x[i]))]
    filtered_tuning[i] = tuning_y[i] / curr_filter_response

filtered_tuning = filtered_tuning / np.max(filtered_tuning)
recovered, remainder = signal.deconvolve(tuning_y, ir)

plt.plot(tuning_x, tuning_y, 'k',label = 'Original tuning')
plt.plot(tuning_x, filtered_tuning, 'r',label = 'GCaMP deconvolved')

plt.legend()

plt.xscale('log')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')
plt.show()