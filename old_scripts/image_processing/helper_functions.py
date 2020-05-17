#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:34:11 2018

@author: burakgur
"""
import os
from collections import OrderedDict
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt


from core_functions import extractRawSignal, readStimOut


# NOT WORKING FOR MARKPOINTS YET
def plotSaveROISignals(dataDir, saveFig = False, mode='Stimulus'):
    """ Extracts signals from a selected ROI set and creates plots for all ROI
    signals according to their tags along with the stimulus.

    Parameters
    ==========
    dataDir : str
        Path containing the ROIs and sima dataset structure
    
    saveFig : bool, optional
        Default: False
        
        Whether to save figures or not.
        
    mode : str, optional
        Default: Stimulus
        
        Determines what kind of plotting will be done according to the given data
        type. If stimulus is used then it will plot traces with stimulus, if 
        MarkPoints are used then it will plot with markpoints.
    
    
        
    Returns
    =======
    
    """
    
    # Get parent dir
    t_series_dir = os.path.dirname(dataDir)

    # Get signal file if exists if not extract
    try:
        signalFile_path = os.path.join(dataDir, '*.csv')
        signalFile = (glob.glob(signalFile_path))[0]    
        print('Signal file found...')
    except:
        print('Signal file not found proceeding with extraction from ROIs...')
        print('File: %s'% dataDir)
        (signalFile, chNames, usedChannel,
         roiKeys, usedRoiKey, usedExtLabel) = extractRawSignal(motCorrDir
                                                               =dataDir)
        
    if mode == 'MarkPoints':
        xmlPath = os.path.join(t_series_dir, '*_MarkPoints.xml')
        xmlFile = (glob.glob(xmlPath))[0]
                
        
    # Read the file and organize the data frame for plotting
    # Data comes from the bg subtracted traces and tags comes from the csv
    # file which has the no bg subtracted traces. 
    ROI_data = pd.read_csv(signalFile,sep='\t',header=2,dtype='float')
    ROI_data = ROI_data.drop(['Unnamed: 0', 'tags'], axis=1) 
    
    # Background subtraction by finding the 'bg' tag as background
    bg_data = ROI_data['bg']
    signal_data = ROI_data.subtract(bg_data, axis=0)
    signal_data = signal_data.drop(['bg'], axis=1) # Get rid of bg
    
    # dF/F by mean of traces
    mean_data = signal_data.mean(axis=0)
    signal_data = (signal_data - mean_data)/mean_data
    
    # Checking the tags of ROIs, while importing a pandas df the column names
    # which in our case are tags,that have the same name are altered with
    # a dot '.' and a number -> e.g. Layer1 & Layer1 -> Layer1 & Layer1.1
    # Here label the same type of ROIs the same again for convenient indexing
    signal_data = signal_data.T
    signal_data.index = [this.split(".")[0] for this in signal_data.index]
    signal_data = signal_data.T
    
    # Finding the unique tags and their occurences for plotting
    unique_columns, column_occurences = np.unique(signal_data.columns.values,
                                                  return_counts=True)
    
    if mode == 'Stimulus':
        # Finding stimulus
        stimOutPath = os.path.join(t_series_dir, '_stimulus_output_*')
        stimOutFile_path = (glob.glob(stimOutPath))[0]
        (stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile_path, 
                                              skipHeader=1)
        stim_name = stimType.split('\\')[-1] 
        
        stim_frames = rawStimData[:,7]  # Frame information
        stim_vals = rawStimData[:,3] # Stimulus value
        uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
        stim_vals = stim_vals[uniq_frame_id]
        stim_vals = stim_vals[:signal_data.shape[0]]
        stim_df = pd.DataFrame(stim_vals,columns=['Stimulus'],dtype='float')
        # Make normalized values of stimulus values for plotting
        stim_df = (stim_df/np.max(np.unique(stim_vals)))*5 
    elif mode == 'MarkPoints':
        a=5
        
    
    # Some color maps for plotting
    cmaps = OrderedDict()
    cmaps['Sequential'] = [
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    
    
    # Figure size etc.
    fig = plt.figure(1,figsize=(14, 12), facecolor='w', edgecolor='k')
    fig.suptitle('%s, stim: %s' % (os.path.basename(t_series_dir),stim_name),
                 fontsize=16)
    
    subPlotNumbers = unique_columns.shape[0]
    nrows = round(float(subPlotNumbers)/float(2))
    if nrows == 1:
        ncols = 1
    else:
        ncols = 2
    for iSubplot, column_name in enumerate(unique_columns):
        # Add linear values 1-2-3 to traces to shift them for visualization
        add_to_shift_traces = np.linspace(1,column_occurences[iSubplot],
                                          column_occurences[iSubplot])
        curr_plot_data = signal_data[[column_name]] + add_to_shift_traces
        iSubplot = iSubplot +1
        ax = fig.add_subplot(nrows, ncols, iSubplot)
        curr_plot_data.plot(ax=ax,legend=False,
                            colormap = cmaps['Sequential'][iSubplot],alpha=0.8)
        stim_df.plot(dashes=[6, 2],ax=ax, color='k')
        plt.title(column_name)
        plt.ylabel('dF/F')
    
    a = int(raw_input("How long you want to inspect this image?"))
    plt.pause(a)
    
    # Save the figure if desired
    if saveFig:
        # Saving figure
        exp_ID = os.path.split(os.path.split(t_series_dir)[0])[1]
        save_name = 'dF-%s-%s' % (exp_ID, os.path.basename(t_series_dir))
        os.chdir(dataDir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('Figure saved')
        plt.close(fig)
    return  None
 
def applyThreshold(nullResponse, realResponse, thresholdType = 'STD',
                   stdMultiplier = 3):
    """ Applys a threshold comparing the null and real responses and returning a 
    boolean if threshold is passed or not.

    Parameters
    ==========
    nullResponse : numpy.ndarray
    
        The signal which the statistics of it will be compared with the 
        real response signal. Usually a background adapted epoch.
        
    realResponse : numpy.ndarray
    
        The signal to be determined as different than null response or not.
        
    thresholdType : str
        Default: 'STD'
        
        Method of thresholding. 
            'STD'  ->   Determining the signal as "responding" if 
                        max(realResponse) > mean(nullResponse) * stdMultiplier *
                        std(nullResponse)
    stdMultiplier : float
        Default: 3
        
        see 'STD' method for where this number is used
    
    Returns
    =======
    thresholdPass : bool
        
        A boolean if the realResponse passes the threshold defined by the user.
  
    """
    if thresholdType == 'STD':
        threshold_value = nullResponse.mean() + (stdMultiplier * nullResponse.std())
        thresholdPass = (threshold_value < realResponse.max())
    
    return thresholdPass