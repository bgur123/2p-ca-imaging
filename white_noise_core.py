#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:11:17 2020

@author: burakgur  - Based on Deniz Yuzak's white noise analysis scripts.
"""

from scipy.io import loadmat
import numpy as np
from scipy import interpolate

def interpolate_data(stimtimes, stimframes100hz, dsignal, imagetimes, freq):
    """Interpolates the stimulus frame numbers (*stimframes100hz*), signal
    traces (*dsignal*) by using the
    stimulus time (*stimtimes*)  and the image time stamps (*imagetimes*)
    recorded. Interpolation is done to a frequency (*freq*) defined by the
    user.
    recorded in

    Parameters
    ----------
    stimtimes : 1D array
        Stimulus time stamps obtained from stimulus_output file (with the
        rate of ~100Hz)
    stimframes100hz : 1D array
        Stimulus frame numbers through recording (with the rate of ~100Hz)
    dsignal : mxn 2D array
        Fluorescence responses of each ROI. Axis m is the number of ROIs while
        n is the time points of microscope recording with lower rate (10-15Hz)
    imagetimes : 1D array
        The time stamps of the image frames with the microscope recording rate
    freq : int
        The desired frequency to interpolate

    Returns
    -------
    newstimtimes : 1D array
        Stimulus time stamps with the rate of *freq*
    dsignal : mxn 2D array
        Fluorescence responses of each ROI with the rate of *freq*
    imagetimes : 1D array
        The time stamps of the image frames with the rate of *freq*
    """
    # Interpolation of stimulus frames and responses to freq

    # Creating time vectors of original 100 Hz(x) and freq Hz sampled(xi)
    # x = vector with 100Hz rate, xi = vector with user input rate (freq)
    x = np.linspace(0,len(stimtimes),len(stimtimes))
    xi = np.linspace(0,len(stimtimes),
                     np.round((np.max(stimtimes)-np.min(stimtimes))*freq)+1)

    # Get interpolated stimulus times for 20Hz
    # stimtimes and x has same rate (100Hz)
    # and newstimtimes is interpolated output of xi vector
    newstimtimes = np.interp(xi, x, stimtimes)
    newstimtimes =  np.array(newstimtimes,dtype='float32')

    # Get interpolated stimulus frame numbers for 20Hz
    # Below stimframes is a continuous function with stimtimes as x and
    # stimframes100Hz as y values
    stimframes = interpolate.interp1d(stimtimes,stimframes100hz,kind='nearest')
    # Below interpolated stimulus times are given as x values to the stimtimes
    # function to find interpolated stimulus frames (y value)
    stimframes = stimframes(newstimtimes)
    stimframes = stimframes.astype('int')

    #Get interpolated responses for 20Hz
    dsignal1 = np.empty(shape=(dsignal.shape[0],
                               len(newstimtimes)),dtype=dsignal.dtype)
    for i in range(dsignal.shape[0]):
        dsignal1[i]=np.interp(newstimtimes, imagetimes, dsignal[i])
    dsignal = dsignal1

    return (newstimtimes, dsignal, stimframes)