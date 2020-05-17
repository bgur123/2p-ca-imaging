#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:19:33 2019

@author: burakgur
"""


import copy
import time
import sima
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.colors import LogNorm
from scipy.stats.stats import pearsonr
from itertools import permutations
from roipoly import RoiPoly
functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
import ROI_mod
from xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize
from core_functions import readStimOut, readStimInformation, getEpochCount, divide_all_epochs
from core_functions import divideEpochs


def get_stim_xml_params(t_series_path, stimInputDir):
    """ Gets the required stimulus and imaging parameters.
    Parameters
    ==========
    t_series_path : str
        Path to the T series folder for retrieving stimulus related information and
        xml file which contains imaging parameters.
    
    stimInputDir : str
        Path to the folder where stimulus input information is located.
        
    Returns
    =======
    stimulus_information : list 
        Stimulus related information is stored here.
    trialCoor : list
        Start, end coordinates for each trial for each epoch
    frameRate : float
        Image acquisiton rate.
    depth :
        Z axis value of the imaging dataset.

    """

    # Finding the xml file and retrieving relevant information
    
    xmlPath = os.path.join(t_series_path, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]
    
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]
    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)
    
    # Stimulus output information
    
    stimOutPath = os.path.join(t_series_path, '_stimulus_output_*')
    stimOutFile = (glob.glob(stimOutPath))[0]
    (stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile, 
                                          skipHeader=1)
    
    # Stimulus information
    (stimInputFile,stimInputData) = readStimInformation(stimType=stimType,
                                                      stimInputDir=stimInputDir)
    stimName = os.path.basename(stimInputFile)
    isRandom = int(stimInputData['Stimulus.randomize'][0])
    epochDur = stimInputData['Stimulus.duration']
    epochDur = [float(sec) for sec in epochDur]
        
    # Finding epoch coordinates and number of trials                                        
    if isRandom:
        epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
        (trialCoor, trialCount, isRandom) = divideEpochs(rawStimData=rawStimData,
                                                 epochCount=epochCount,
                                                 isRandom=isRandom,
                                                 framePeriod=framePeriod,
                                                 trialDiff=0.20,
                                                 overlappingFrames=0,
                                                 firstEpochIdx=0,
                                                 epochColumn=3,
                                                 imgFrameColumn=7,
                                                 incNextEpoch=True,
                                                 checkLastTrialLen=True)
    else:
        epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
        (trialCoor, trialCount) = divide_all_epochs(rawStimData, epochCount, 
                                                    framePeriod, trialDiff=0.20,
                                                    epochColumn=3, imgFrameColumn=7,
                                                    checkLastTrialLen=True)
     
    stimulus_information ={}
    stimulus_data = stimInputData
    stimulus_information['random'] = isRandom
    if not(isRandom):
        stimulus_information['baseline_epoch'] = 0  
        print('\n Stimulus non random, baseline epoch selected as 0th epoch\n')
    else:
        stimulus_information['baseline_epoch'] = 0 
        
    stimulus_information['epoch_dir'] = \
            np.asfarray(stimulus_data['Stimulus.stimrot.mean'])
    epoch_speeds = np.asfarray(stimulus_data['Stimulus.stimtrans.mean'])
    stimulus_information['epoch_frequency'] = \
        epoch_speeds/np.asfarray(stimulus_data['Stimulus.spacing'])
    stimulus_information['baseline_duration'] = \
        np.asfarray(stimulus_data['Stimulus.duration'][0])
    stimulus_information['epochs_duration'] =\
         np.asfarray(stimulus_data['Stimulus.duration'])
    stimulus_information['epoch_number'] =  \
        np.asfarray(stimulus_data['EPOCHS'][0])
    stimulus_information['stim_type'] =  \
        np.asfarray(stimulus_data['Stimulus.stimtype'])
        
    return stimulus_information, trialCoor, frameRate, depth, x_size, y_size, pixelArea,\
           stimType, stimOutFile, epochCount, stimName, layerPosition, stimInputFile,\
           xmlFile, trialCount, isRandom, stimInputData, rawStimData

def separate_trials_video(time_series,trialCoor,stimulus_information,
                    frameRate):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : list
        Start, end coordinates for each trial for each epoch
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    dff_baseline_dur_frame: int
        Duration of baseline before the stimulus for using in dF/F calculation.
        
    Returns
    =======
    wholeTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-

    """
    mov_xDim = time_series.shape[1]
    mov_yDim = time_series.shape[2]
    wholeTraces_allTrials = {}
    respTraces_allTrials = {}
    baselineTraces_allTrials = {}
    for iEpoch in trialCoor:
        currentEpoch = trialCoor[iEpoch]
        
        
        current_epoch_dur = stimulus_information['epochs_duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            trial_lens.append(current_trial_length)
            
            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            
            base_lens.append(baselineEnd - baselineStart) 
     
            
        trial_len =  min(trial_lens)-2
        resp_len = int(round(frameRate * current_epoch_dur))+1
#        resp_len = int(round(frameRate * 3))
        base_len = min(base_lens)
        
        if stimulus_information['random']:
            wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            respTraces_allTrials[iEpoch] = np.zeros(shape=(resp_len,mov_xDim,mov_yDim,
                               trial_numbers))
            baselineTraces_allTrials[iEpoch] = np.zeros(shape=(base_len,mov_xDim,mov_yDim,
                               trial_numbers))
        else:
            wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            respTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                               trial_numbers))
            base_len  = np.shape(wholeTraces_allTrials\
                                 [stimulus_information['baseline_epoch']])[0]
            baselineTraces_allTrials[iEpoch] = np.zeros(shape=(base_len,mov_xDim,mov_yDim,
                               trial_numbers))
                
        
        for trial_num , current_trial_coor in enumerate(currentEpoch):
            
            if stimulus_information['random']:
                trialStart = current_trial_coor[0][0]
                trialEnd = current_trial_coor[0][1]
                
                baselineStart = current_trial_coor[1][0]
                baselineEnd = current_trial_coor[1][1]
                
                respStart = current_trial_coor[1][1]
                epochEnd = current_trial_coor[0][1]
                
                raw_signal = time_series[trialStart:trialEnd, : , :]
            
                currentResp = time_series[respStart:epochEnd, : , :]
                   #        dffTraces_allTrials[iEpoch].append(dFF[:trial_len,:,:])
                wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                respTraces_allTrials[iEpoch][:,:,:,trial_num]= currentResp[:resp_len,:,:]
                baselineTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:base_len,:,:]
            else:
                
                # If the sequence is non random  the trials are just separated without any baseline
                trialStart = current_trial_coor[0][0]
                trialEnd = current_trial_coor[0][1]
                if iEpoch == stimulus_information['baseline_epoch']:
                    baseline_signal = time_series[trialStart:trialEnd, : , :]
                raw_signal = time_series[trialStart:trialEnd, : , :]
                
                wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                respTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
                baselineTraces_allTrials[iEpoch][:,:,:,trial_num]= baseline_signal[:base_len,:,:]
            
           
            
            
 
        
        print('Epoch %d completed \n' % iEpoch)
        
    return (wholeTraces_allTrials, respTraces_allTrials, baselineTraces_allTrials)

def calculate_pixel_SNR(baselineTraces_allTrials,respTraces_allTrials,
                  stimulus_information,frameRate,SNR_mode ='Estimate'):
    """ Calculates the pixel-wise signal-to-noise ratio (SNR). Equation taken from
    Kouvalainen et al. 1994 (see calculation of SNR true from SNR estimated). 
    
    Parameters
    ==========
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    SNR_mode : not implemented yet
    
        
        
    Returns
    =======
    
    SNR_max_matrix : np array
        An m x n array with pixel-wise SNR.

    """
    
    mov_xDim = np.shape(baselineTraces_allTrials[1])[1]
    mov_yDim = np.shape(baselineTraces_allTrials[1])[2]
    total_epoch_numbers = len(baselineTraces_allTrials)
    
    
#    total_background_dur = stimulus_information['epochs_duration'][0]
    SNR_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
    for iPlot, iEpoch in enumerate(baselineTraces_allTrials):
        
        trial_numbers = np.shape(baselineTraces_allTrials[iEpoch])[3]
        currentBaseTrace = baselineTraces_allTrials[iEpoch][:,:,:,:]
        currentRespTrace =  respTraces_allTrials[iEpoch][:,:,:,:]
        
        noise_std = currentBaseTrace.std(axis=0).mean(axis=2)
        resp_std = currentRespTrace.std(axis=0).mean(axis=2)
        
        signal_std = resp_std - noise_std
        # SNR calculation taken from
        curr_SNR_true = ((trial_numbers+1)/trial_numbers)*(signal_std/noise_std) - 1/trial_numbers
#        curr_SNR = (signal_std/noise_std) 
        SNR_matrix[:,:,iPlot] = curr_SNR_true
        
       
    SNR_matrix[np.isnan(SNR_matrix)] = np.nanmin(SNR_matrix) # change nan values with min values
    
    SNR_max_matrix = SNR_matrix.max(axis=2) # Take max SNR for every pixel for every epoch

    return SNR_max_matrix

def calculate_pixel_max(respTraces_allTrials,stimulus_information):
    
    """ Calculates the pixel-wise maximum responses for each epoch. Returns max 
    epoch indices as well but adjusting the indices considering that baseline is 0.
    
    Parameters
    ==========
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
        
    Returns
    =======
    
    MaxResp_matrix_without_edge : np array
        An array with pixel-wise maxiumum responses for epochs other than edges
        that are normally used as probe stimuli. Edge maximums are set to -100 so that
        they're never maximum.
    
    MaxResp_matrix_all_epochs : np array
        An array with pixel-wise maxiumum responses for every epoch.
        
    maxEpochIdx_matrix_without_edge : np array
        An array with pixel-wise maximum epoch indices. Adjusts the indices considering
        that baseline index is 0 and epochs start from 1.
        
    maxEpochIdx_matrix_all : np array
        An array with pixel-wise maximum epoch indices. Adjusts the indices considering
        that baseline index is 0 and epochs start from 1.
        
    """
    epoch_adjuster = stimulus_information['epoch_adjuster']
    
    mov_xDim = np.shape(respTraces_allTrials[1])[1]
    mov_yDim = np.shape(respTraces_allTrials[1])[2]
    total_epoch_numbers = len(respTraces_allTrials)
    
    # Create an epoch-wise maximum response list
    maxResp = {}
    meanResp ={}
    # Create an array with m x n x nEpochs
    MaxResp_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
    MeanResp_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
    
    for index, iEpoch in enumerate(respTraces_allTrials):
        
        # Find maximum of pixels after trial averaging
        curr_max =  np.nanmax(np.nanmean(respTraces_allTrials[iEpoch][:,:,:,:],axis=3),axis=0)
        curr_mean = np.nanmean(np.nanmean(respTraces_allTrials[iEpoch][:,:,:,:],axis=3),axis=0)
        
        maxResp[iEpoch] = curr_max
        MaxResp_matrix[:,:,index] = curr_max
        
        meanResp[iEpoch] = curr_mean
        MeanResp_matrix[:,:,index] = curr_mean
        
    
    # Make an additional one with edge and set edge maximum to 0 in the main array
    # This is to avoid assigning pixels to edge temporal frequency if they respond
    # max in the edge epoch but not in one of the grating epochs.
    MaxResp_matrix_all_epochs = copy.deepcopy(MaxResp_matrix)
    MeanResp_matrix_all_epochs = copy.deepcopy(MeanResp_matrix)
    MaxResp_matrix_without_edge = copy.deepcopy(MaxResp_matrix)
    if stimulus_information['edge_exists']:
        
        # Find edge epochs
        edge_epochs = np.where(stimulus_information['stim_type']==50)[0]
        edge_epochs = edge_epochs - epoch_adjuster
        
        for iEdgeEpoch in edge_epochs:
            MaxResp_matrix_without_edge[:,:,iEdgeEpoch] = -100
    
    # Finding pixel-wise max epochs
    maxEpochIdx_matrix_without_edge = np.argmax(MaxResp_matrix_without_edge,axis=2) 
    # To assign numbers like epoch numbers
    maxEpochIdx_matrix_without_edge = maxEpochIdx_matrix_without_edge + epoch_adjuster 
    
    # Finding pixel-wise max epochs
    maxEpochIdx_matrix_all = np.argmax(MaxResp_matrix_all_epochs,axis=2) 
    maxEpochIdx_matrix_all_mean = np.argmax(MeanResp_matrix_all_epochs,axis=2) 
    
    # To assign numbers like epoch numbers
    maxEpochIdx_matrix_all = maxEpochIdx_matrix_all + epoch_adjuster
    maxEpochIdx_matrix_all_mean = maxEpochIdx_matrix_all_mean + epoch_adjuster
    
    
    return MaxResp_matrix_all_epochs, MaxResp_matrix_without_edge, \
           maxEpochIdx_matrix_without_edge,maxEpochIdx_matrix_all, \
           MeanResp_matrix_all_epochs, maxEpochIdx_matrix_all_mean
            


def create_DSI_image(stimulus_information, maxEpochIdx_matrix_all,max_resp_matrix_all,
                     MaxResp_matrix_all_epochs):
    """ Makes pixel-wise plot of DSI

    Parameters
    ==========
   stimulus_information : list 
        Stimulus related information is stored here.
        
    maxEpochIdx_matrix_all : np array
        An array with pixel-wise maximum epoch indices. Adjusts the indices considering
        that baseline index is 0 and epochs start from 1.
        
    max_resp_matrix_all : np array
        An array with pixel-wise maxiumum responses for the experiment.
        
    MaxResp_matrix_all_epochs : np array
        An array with pixel-wise maxiumum responses for every epoch.
        
    

     
    Returns
    =======
    
    DSI_image : numpy.ndarray
        An image with CSI values ranging between -1 and 1. (-1 OFF 1 ON selective)
    
    """
    
    DSI_image = copy.deepcopy(max_resp_matrix_all) # copy it for keeping nan value
    for iEpoch, current_epoch_type in enumerate (stimulus_information['stim_type']):
        
        if (stimulus_information['random']) and (iEpoch ==0):
            continue
        
        current_pixels = (maxEpochIdx_matrix_all == iEpoch) & \
                            (~np.isnan(max_resp_matrix_all))
        current_freq = stimulus_information['epoch_frequency'][iEpoch]
        if ((current_epoch_type != 50) and (current_epoch_type != 61) and\
            (current_epoch_type != 46)) or (current_freq ==0):
            DSI_image[current_pixels] = 0
            continue
        current_dir = stimulus_information['epoch_dir'][iEpoch]
        if current_dir == 90:
            multiplier = -1
        else:
            multiplier = 1
        required_epoch_array = \
            (stimulus_information['epoch_dir'] == ((current_dir+180) % 360)) & \
            (stimulus_information['epoch_frequency'] == current_freq) & \
            (stimulus_information['stim_type'] == current_epoch_type)
            
        opposite_dir_epoch = [epoch_indx for epoch_indx, epoch in \
                              enumerate(required_epoch_array) if epoch][0]
        opposite_dir_epoch = opposite_dir_epoch # To find the real epoch number without the baseline
        # Since a matrix will be indexed which doesn't have any information about the baseline epoch
        
    
        opposite_response_trace = MaxResp_matrix_all_epochs\
            [:,:,opposite_dir_epoch-stimulus_information['epoch_adjuster']] 
            
        DSI_image[current_pixels] = multiplier*(np.abs(((max_resp_matrix_all[current_pixels] - \
                                      opposite_response_trace[current_pixels])\
                                        /(max_resp_matrix_all[current_pixels] + \
                                          opposite_response_trace[current_pixels]))))
#    DSI_image[(DSI_image>1)] = 0
#    DSI_image[(DSI_image<-1)] = 0
    
    return DSI_image

def create_CSI_image(stimulus_information, frameRate,respTraces_allTrials, 
                     DSI_image):
    """ Makes pixel-wise plot of CSI

    Parameters
    ==========
   stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    respTraces_allTrials : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
            
    DSI_image : numpy.ndarray
        An image with CSI values ranging between -1 and 1. (-1 OFF 1 ON selective)
    

     
    Returns
    =======
    
    CSI_image : numpy.ndarray
        An image with CSI values ranging between -1 and 1. (-1 OFF 1 ON selective)
    
    """
    # Image dimensions
    mov_xDim = np.shape(DSI_image)[0]
    mov_yDim = np.shape(DSI_image)[1]
    # Find edge epochs
    edge_epochs = np.where(stimulus_information['stim_type']==50)[0]
    epochDur= stimulus_information['epochs_duration']
    
    if len(edge_epochs) == 2: # 2 edges exist 
        
        ON_resp = np.zeros(shape=(mov_xDim,mov_yDim,2))
        OFF_resp = np.zeros(shape=(mov_xDim,mov_yDim,2))
        CSI_image = np.zeros(shape=(mov_xDim,mov_yDim))
        half_dur_frames = int((round(frameRate * epochDur[edge_epochs[0]]))/2)
        
        for index, epoch in enumerate(edge_epochs):
            
            
            OFF_resp[:,:,index] = np.nanmax(np.nanmean(\
                    respTraces_allTrials[epoch]\
                    [:half_dur_frames,:,:,:],axis=3),axis=0)
            ON_resp[:,:,index] = np.nanmax(np.nanmean(\
                   respTraces_allTrials[epoch]\
                   [half_dur_frames:,:,:,:],axis=3),axis=0)
        
        
        CSI_image[DSI_image>0] = (ON_resp[:,:,0][DSI_image>0] - OFF_resp[:,:,0][DSI_image>0])/(ON_resp[:,:,0][DSI_image>0] + OFF_resp[:,:,0][DSI_image>0])
        CSI_image[DSI_image<0] = (ON_resp[:,:,1][DSI_image<0] - OFF_resp[:,:,1][DSI_image<0])/(ON_resp[:,:,1][DSI_image<0] + OFF_resp[:,:,1][DSI_image<0])
        
    # It shouldn't be below -1 or above +1 if it is not noise
    CSI_image[(CSI_image>1)] = 0
    CSI_image[(CSI_image<-1)] = 0    
    
    return CSI_image


def plot_pixel_maps(im1, im2, im3, im4, exp_ID, depth, save_fig = False,
                    save_dir = None):
    """ Plots 4 images in a figure. Normally used with mean, max, DSI, CSI 
    images
    
    Parameters
    ==========
    mean_image : numpy array
        Mean image of the video.
    
    max_resp_matrix_all : numpy array
        Maximum responses
        
    DSI_image : numpy array
        Mean image of the video.
        
    CSI_image : numpy array
        Mean image of the video.
        
    exp_ID : str
    
    depth : int or float
        

    """
    plt.close('all')
    
    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True,
                            facecolor='k', edgecolor='w',figsize=(16, 10))
    
    
    depthstr = 'Z: %d' % depth
    figtitle = 'Summary ' + depthstr
    
    # Mean image
    fig1.suptitle(figtitle,fontsize=12)
    sns.heatmap(im1,ax=ax1[0][0],cbar_kws={'label': 'dF/F'},cmap='viridis')
    #    sns.heatmap(layer_masks,alpha=.2,cmap='Set1',ax=ax2[0],cbar=False)
    #    sns.heatmap(BG_mask,alpha=.1,ax=ax2[0],cbar=False)
    ax1[0][0].axis('off')
    ax1[0][0].set_title('Mean image')
    
    # Max responses
    sns.heatmap(im2,cbar_kws={'label': 'SNR'},ax=ax1[0][1])
    #sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'RdBu_r',ax=ax2[1])
    ax1[0][1].axis('off')
    ax1[0][1].set_title('SNR image')
    # DSI
    sns.heatmap(im3,cbar_kws={'label': 'DSI'},cmap = 'RdBu_r',ax=ax1[1][0])
    #sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'RdBu_r',ax=ax2[1])
    ax1[1][0].axis('off')
    ax1[1][0].set_title('DSI (Blue: --> Red: <--)')
    
    #CSI
    sns.heatmap(im4,cbar_kws={'label': 'CSI'},cmap = 'inferno',ax=ax1[1][1],vmax=1,vmin=-1)
    
    ax1[1][1].axis('off')    
    ax1[1][1].set_title('CSI (Dark:OFF, Red:ON)')
    
    if save_fig:
        # Saving figure
        save_name = 'summary_%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('Pixel maps saved')

def generate_cluster_movie(dataDir, stimulus_information, 
                           wholeTraces_allTrials_video):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces.
    
    Parameters
    ==========
    dataDir: str
        Path into the directory where the motion corrected dataset with selected
        masks is present.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    wholeTraces_allTrials_video : list containing np arrays
        Epoch list of time traces including all trials in the form of:
        baseline epoch - stimulus epoch - baseline epoch
            
    
        
    Returns
    =======
    
    cluster_dataset : sima.imaging.ImagingDataset 
        Sima dataset to be used for segmentation.
    """
    print('Generating cluster movie...\n')
    # Directory for where to save the cluster movie
    selected_movie_dir = os.path.join(dataDir,'cluster.sima')
    mov_xDim = np.shape(wholeTraces_allTrials_video[1])[1]
    mov_yDim = np.shape(wholeTraces_allTrials_video[1])[2]
    epoch_freqs = stimulus_information['epoch_frequency']
    
    # Taking frequencies < 15
    epochs_to_use = np.where((epoch_freqs<15))[0]
    if stimulus_information['random']:
        epochs_to_use=np.delete(epochs_to_use,0)
    epoch_frames = np.zeros(shape=np.shape(epochs_to_use))
    
    # Generating and filling the movie array movie array
    for index, epoch in enumerate(epochs_to_use):
        epoch_frames[index] = np.shape(wholeTraces_allTrials_video[epoch])[0]
        
    cluster_movie = np.zeros(shape=(int(epoch_frames.sum()),1,mov_xDim,mov_yDim,1))
    
    startFrame = 0
    for index, epoch in enumerate(epochs_to_use):
        if index>0:
            startFrame =  endFrame 
        endFrame = startFrame + epoch_frames[index]
        cluster_movie[int(startFrame):int(endFrame),0,:,:,0] = \
        wholeTraces_allTrials_video[epoch].mean(axis=3)
            
    
    # Create a sima dataset and export the cluster movie
    b = sima.Sequence.create('ndarray',cluster_movie)
    cluster_dataset = sima.ImagingDataset([b],None)
    cluster_dataset.export_frames([[[os.path.join(selected_movie_dir,'cluster_vid.tif')]]],
                                      fill_gaps=True,scale_values=True)
    print('Cluster movie generated...\n')
    
    return cluster_dataset

def find_clusters_STICA(cluster_dataset, area_min, area_max):
    """ Makes pixel-wise plot of DSI

    Parameters
    ==========
    cluster_dataset : sima.imaging.ImagingDataset 
        Sima dataset to be used for segmentation.
        
    area_min : int
        Minimum area of a cluster in pixels
        
    area_max : int
        Maximum area of a cluster in pixels
        
    

     
    Returns
    =======
    
    clusters : sima.ROI.ROIList
        A list of ROIs.
        
    all_clusters_image: numpy array
        A numpy array that contains the masks.
    """    
    print('\n-->Segmentation running...')
    segmentation_approach = sima.segment.STICA(channel = 0,components=45,mu=0.1)
    segmentation_approach.append(sima.segment.SparseROIsFromMasks(
            min_size=area_min,smooth_size=3))
    #segmentation_approach.append(sima.segment.MergeOverlapping(threshold=0.90))
    #segmentation_approach.append(sima.segment.SmoothROIBoundaries(tolerance=0.1,n_processes=(nCpu - 1)))
    size_filter = sima.segment.ROIFilter(lambda roi: roi.size >= area_min and \
                                         roi.size <= area_max)
#    circ_filter = sima.segment.CircularityFilter(circularity_threhold=0.7)
    segmentation_approach.append(size_filter)
#    segmentation_approach.append(circ_filter)
    start1 = time.time()
    clusters = cluster_dataset.segment(segmentation_approach, 'auto_ROIs')
    initial_cluster_num = len(clusters)
    end1 = time.time()
    time_passed = end1-start1
    print('Clusters found in %d minutes\n' % \
          round(time_passed/60) )
    print('Number of initial clusters: %d\n' % initial_cluster_num)
    
    
    # Generating an image with all clusters
    data_xDim = cluster_dataset.frame_shape[1]
    data_yDim = cluster_dataset.frame_shape[2]
    all_clusters_image = np.zeros(shape=(data_xDim,data_yDim))
    all_clusters_image[:] = np.nan
    for index, roi in enumerate(clusters):
        curr_mask = np.array(roi)[0,:,:]
        all_clusters_image[curr_mask] = index+1        
        
    return clusters, all_clusters_image

def get_layers_bg_mask(dataDir):
    """ Gets the masks of pre-selected (with roibuddy) layers.

    Parameters
    ==========
    dataDir: str
        Path into the directory where the motion corrected dataset with selected
        masks is present.
    
  
    Returns
    =======
    
    layer_masks_bool: np array
        A boolean image of where the masks are located
        
    BG_mask: np array
        A boolean image of background mask
    
    """                       
                                    
    dataset = sima.ImagingDataset.load(dataDir)
    roiKeys = dataset.ROIs.keys()
    roiKeyNo = 0
    rois_layer = dataset.ROIs[roiKeys[roiKeyNo]]
    layer_masks = np.zeros(shape=(np.shape(np.array(rois_layer[0]))[1],np.shape(np.array(rois_layer[0]))[2]))
    layer_masks_bool = np.zeros(shape=(np.shape(np.array(rois_layer[0]))[1],np.shape(np.array(rois_layer[0]))[2]))
    layer_masks[:] = np.nan
    
    BG_mask = np.zeros(shape=(np.shape(np.array(rois_layer[0]))[1],np.shape(np.array(rois_layer[0]))[2]))
    BG_mask[:] = np.nan
    for index, roi in enumerate(rois_layer):
        curr_mask = np.array(roi)[0,:,:]
        roi_label = roi.label
        
        if roi_label == 'Layer1':
            
            L1_mask = curr_mask
            layer_masks[curr_mask] = 1
            layer_masks_bool[curr_mask] = 1
            print("Layer 1 mask found\n")
        elif roi_label == 'Layer2':
            L2_mask = curr_mask
            layer_masks[curr_mask] = 2
            layer_masks_bool[curr_mask] = 1
            print("Layer 2 mask found\n")
        elif roi_label == 'Layer3':
             
            L3_mask = curr_mask
            layer_masks[curr_mask] = 3
            layer_masks_bool[curr_mask] = 1
            print("Layer 3 mask found\n")
        elif roi_label == 'Layer4':
            L4_mask = curr_mask
            layer_masks[curr_mask] = 4
            layer_masks_bool[curr_mask] = 1
            print("Layer 4 mask found\n")
        elif roi_label == 'LobDen':
            LD_mask = curr_mask
    #        layer_masks[curr_mask] = 3
    #        layer_masks_bool[curr_mask] = 1
        elif roi_label == 'MedDen':
            MD_mask = curr_mask
    #        layer_masks[curr_mask] = 4
    #        layer_masks_bool[curr_mask] = 1
        elif roi_label == 'BG':
            BG_mask = curr_mask
            print("BG mask found\n")
        else:
            print("ROI doesn't match to criteria: ")
            print(roi_label)      

    return layer_masks_bool, BG_mask                              
                                
                        
def select_regions(image_to_select_from, image_cmap ="gray",pause_t=7,
                   ask_name=True):
    """ Enables user to select rois from a given image using roipoly module.

    Parameters
    ==========
    image_to_select_from : numpy.ndarray
        An image to select ROIs from
    
    Returns
    =======
    
    """
    plt.close('all')
    stopsignal = 0
    roi_number = 0
    roi_masks = []
    mask_names = []
    
    im_xDim = np.shape(image_to_select_from)[0]
    im_yDim = np.shape(image_to_select_from)[1]
    mask_agg = np.zeros(shape=(im_xDim,im_yDim))
    iROI = 0
    plt.style.use("dark_background")
    while (stopsignal==0):

        
        # Show the image
        fig = plt.figure()
        plt.imshow(image_to_select_from, interpolation='nearest', cmap=image_cmap)
        plt.colorbar()
        plt.imshow(mask_agg, alpha=0.3,cmap = 'tab20b')
        plt.title("Select ROI: ROI%d" % roi_number)
        plt.show(block=False)
       
        
        # Draw ROI
        curr_roi = RoiPoly(color='r', fig=fig)
        iROI = iROI + 1
        plt.waitforbuttonpress()
        plt.pause(pause_t)
        if ask_name:
            mask_name = raw_input("\nEnter the ROI name:\n>> ")
        else:
            mask_name = iROI
        mask_names.append(mask_name)
        
        curr_mask = curr_roi.get_mask(image_to_select_from)
        roi_masks.append(curr_mask)
        
        mask_agg[curr_mask] += 1
        
        
        
        roi_number += 1
        signal = raw_input("\nPress k for exiting program, otherwise press enter")
        if (signal == 'k'):
            stopsignal = 1
        
    
    return roi_masks, mask_names

def clusters_restrict_size_regions(rois, cluster_region_bool,
                                   cluster_1d_max_size_pixel,
                                   cluster_1d_min_size_pixel):
    """ Makes pixel-wise plot of DSI

    Parameters
    ==========
    clusters : sima.ROI.ROIList
        A list of ROIs.
        
    cluster_region_bool : Boolean array
        A boolean array that depicts the delimiting regions of clusters. 
        
    cluster_1d_max_size_pixel : int
        Maximum 1d size of a cluster in pixels
        
    cluster_1d_min_size_pixel : int
        Minimum 1d size of a cluster in pixels
        
    cluster_dataset : sima.imaging.ImagingDataset 
        Sima dataset to be used for segmentation.
        
    

     
    Returns
    =======
    
    separated_masks : list
        A list of ROIs.
        
    sep_masks_image: numpy array
        A numpy array that contains the masks.
    """    
    
    # Getting rid of clusters based on pre-defined regions and size
    passed_rois  = []
    ROI_mod.calcualte_mask_1d_size(rois)
    for roi in rois:
        # Check if mask is within the pre-defined regions
        mask_inclusion_points =  np.where(roi.mask * cluster_region_bool)[0]
        if mask_inclusion_points.size == np.where(roi.mask)[0].size: 
            # Check if mask is within the size restrictions
            if ((roi.x_size < cluster_1d_max_size_pixel) & (roi.y_size < cluster_1d_max_size_pixel) & \
                   (roi.x_size > cluster_1d_min_size_pixel) & (roi.y_size > cluster_1d_min_size_pixel)):
                
                passed_rois.append(roi)
                
            
    # Generating an image with masks
    data_xDim = np.shape(rois[0].mask)[0]
    data_yDim = np.shape(rois[0].mask)[1]
    
    passed_rois_image = np.zeros(shape=(data_xDim,data_yDim))
    passed_rois_image[:] = np.nan
    for index, roi in enumerate(passed_rois):
        passed_rois_image[roi.mask] = index+1
    
    print('Clusters excluded based on layers...')
    
    all_pre_selected_mask = np.zeros(shape=(data_xDim,data_yDim))

    pre_selected_roi_indices = np.arange(len(passed_rois))
    pre_selected_roi_indices_copy = np.arange(len(passed_rois))
    
    for index, roi in enumerate(passed_rois):
        all_pre_selected_mask[roi.mask] += 1
        
    # Getting rid of overlapping clusters
    while len(np.where(all_pre_selected_mask>1)[0]) != 0:
        
        for index, roi_idx in enumerate(pre_selected_roi_indices):
            
            if pre_selected_roi_indices[index] != -1:
                curr_mask = passed_rois[roi_idx].mask
                non_intersection_matrix = \
                    (all_pre_selected_mask[curr_mask] == 1)
                
                if len(np.where(non_intersection_matrix)[0]) == 0: 
                    # get rid of cluster if it doesn't have any non overlapping part
                    pre_selected_roi_indices[index] = -1
                    all_pre_selected_mask[curr_mask] -= 1
                    
                elif (len(np.where(non_intersection_matrix)[0]) != len(all_pre_selected_mask[curr_mask])): 
                    # get rid of cluster if it has any overlapping part
                    pre_selected_roi_indices[index] = -1
                    all_pre_selected_mask[curr_mask] -= 1
            else:
               continue
    
    # To retrieve some clusters if there are no overlaps 
    for iRep in range(100):
        
        for index, roi in enumerate(pre_selected_roi_indices):
            if pre_selected_roi_indices[index] == -1:
        #        print(index)
                curr_mask = passed_rois[pre_selected_roi_indices_copy[index]].mask
                non_intersection_matrix = (all_pre_selected_mask[curr_mask] == 0)
                if (len(np.where(non_intersection_matrix)[0]) == len(all_pre_selected_mask[curr_mask])):
                    # If there's no cluster here add the cluster back
                    print('cluster added back')
                    pre_selected_roi_indices[index] = pre_selected_roi_indices_copy[index]
                    all_pre_selected_mask[curr_mask] += 1
    
    separated_roi_indices = pre_selected_roi_indices[pre_selected_roi_indices != -1]
    sep_masks_image = np.zeros(shape=(data_xDim,data_yDim))
    sep_masks_image[:] = np.nan
    separated_rois = []
    
    for index, sep_clus_idx in enumerate(separated_roi_indices):
        sep_masks_image[passed_rois[sep_clus_idx].mask] = index+1
        
        separated_rois.append(passed_rois[sep_clus_idx])
    
    print('Clusters separated...')
    print('Cluster pass ratio: %.2f' % (float(len(separated_rois))/\
                                        float(len(rois))))
    print('Total clusters: %d'% len(separated_rois))
    
    return separated_rois, sep_masks_image
        


def separate_trials_ROI_v3(time_series,trialCoor,rois,stimulus_information,
                           frameRate, df_method, df_use = True, plotting=False,
                           max_resp_trial_len = 'max'):
    """ Separates trials epoch-wise into big lists of whole traces, response traces
    and baseline traces. Adds responses and whole traces into the ROI_bg 
    instances.
    
    Parameters
    ==========
    time_series : numpy array
        Time series in the form of: frames x m x n (m & n are pixel dimensions)
    
    trialCoor : list
        Start, end coordinates for each trial for each epoch
    
    rois : list
        A list of ROI_bg instances.
        
    stimulus_information : list 
        Stimulus related information is stored here.
        
    frameRate : float
        Image acquisiton rate.
        
    df_method : str
        Method for calculating dF/F defined in the ROI_bg class.
        
    plotting: bool
        If the user wants to visualize the masks and the traces for clusters.
        
    Returns
    =======
    wholeTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            baseline epoch - stimulus epoch - baseline epoch
            
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    """
    wholeTraces_allTrials_ROIs = {}
    respTraces_allTrials_ROIs = {}
    baselineTraces_allTrials_ROIs = {}
    all_clusters_dF_whole_trace = np.zeros(shape=(len(rois),
                                                  np.shape(time_series)[0]))
    
    for iROI, roi in enumerate(rois):
        
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
        roi.calculateDf(method=df_method,moving_avg = True, bins = 3)
        all_clusters_dF_whole_trace[iROI,:] = roi.df_trace
        
        if df_use:
            roi.base_dur = [] # Initialize baseline duration here (not good practice...)
        if plotting:
            plt.figure(figsize=(8, 7))
            grid = plt.GridSpec(8, 1, wspace=0.4, hspace=0.3)
            plt.subplot(grid[:7,0])
            roi.showRoiMask()
            plt.subplot(grid[7:8,0])
            plt.plot(roi.df_trace)
            plt.title('Cluster %d %s:' % (iROI, roi))
            
            
            plt.waitforbuttonpress()
            plt.close('all')
            
        
    for iEpoch in trialCoor:
        currentEpoch = trialCoor[iEpoch]
        current_epoch_dur = stimulus_information['epochs_duration'][iEpoch]
        trial_numbers = len(currentEpoch)
        trial_lens = []
        resp_lens = []
        base_lens = []
        for curr_trial_coor in currentEpoch:
            current_trial_length = curr_trial_coor[0][1]-curr_trial_coor[0][0]
            trial_lens.append(current_trial_length)
            
            baselineStart = curr_trial_coor[1][0]
            baselineEnd = curr_trial_coor[1][1]
            
            base_lens.append(baselineEnd - baselineStart) 
        
        
        trial_len =  min(trial_lens) -2
        
        if (max_resp_trial_len == 'max') or (current_epoch_dur < max_resp_trial_len):
            resp_len = int(round(frameRate * current_epoch_dur))+1
        else:
            resp_len = int(round(frameRate * max_resp_trial_len))+1
            
        
        base_len = min(base_lens)
        wholeTraces_allTrials_ROIs[iEpoch] = {}
        respTraces_allTrials_ROIs[iEpoch] = {}
        baselineTraces_allTrials_ROIs[iEpoch] = {}
   
        for iCluster, roi in enumerate(rois):
            
            if stimulus_information['random']:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                         trial_numbers))
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                         trial_numbers))
            else:
                wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                         trial_numbers))
                base_len  = np.shape(wholeTraces_allTrials_ROIs\
                                     [stimulus_information['baseline_epoch']]\
                                     [iCluster])[0]
                baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(int(frameRate*1.5),
                                   trial_numbers))
            
            for trial_num , current_trial_coor in enumerate(currentEpoch):
                
                if stimulus_information['random']:
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    baselineStart = current_trial_coor[1][0]
                    baselineEnd = current_trial_coor[1][1]
                    
                    respStart = current_trial_coor[1][1]
                    epochEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                        roi_resp = roi.df_trace[respStart:epochEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        roi_resp = roi.raw_trace[respStart:epochEnd]
                    
                            
                    wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_resp[:resp_len]
                    baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:base_len]
                else:
                    
                    # If the sequence is non random  the trials are just separated without any baseline
                    trialStart = current_trial_coor[0][0]
                    trialEnd = current_trial_coor[0][1]
                    
                    if df_use:
                        roi_whole_trace = roi.df_trace[trialStart:trialEnd]
                    else:
                        roi_whole_trace = roi.raw_trace[trialStart:trialEnd]
                        
                    
                    if iEpoch == stimulus_information['baseline_epoch']:
                        baseline_trace = roi_whole_trace[:base_len]
                        baseline_trace = baseline_trace[-int(frameRate*1.5):]
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= baseline_trace
                    else:
                        baselineTraces_allTrials_ROIs[iEpoch][iCluster]\
                            [:,trial_num]= baselineTraces_allTrials_ROIs\
                            [stimulus_information['baseline_epoch']][iCluster]\
                            [:,trial_num]
                    
                    wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= roi_whole_trace[:trial_len]
                    
                    
    for iEpoch in trialCoor:
        for iCluster, roi in enumerate(rois):
            
            # Appending trial averaged responses to roi instances
            if df_use:
                if not stimulus_information['random']:
                    if iEpoch > 0 and iEpoch < len(trialCoor)-1:
                        
                        wt = np.concatenate((wholeTraces_allTrials_ROIs[iEpoch-1][iCluster].mean(axis=1),
                                            wholeTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1),
                                            wholeTraces_allTrials_ROIs[iEpoch+1][iCluster].mean(axis=1)),
                                            axis =0)
                        roi.base_dur.append(len(wholeTraces_allTrials_ROIs[iEpoch-1][iCluster].mean(axis=1))) 
                    else:
                        wt = wholeTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1)
                        roi.base_dur.append(0) 
                else:
                    wt = wholeTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1)
                    base_dur = frameRate * stimulus_information['baseline_duration']
                    roi.base_dur.append(int(round(base_dur)))
                    
                roi.appendTrace(wt,iEpoch, trace_type = 'whole')
                roi.appendTrace(respTraces_allTrials_ROIs[iEpoch][iCluster].mean(axis=1),
                                  iEpoch, trace_type = 'response' )
                    
                    
                
        
    if df_use:
        print('Trial separation for ROIs completed')
    else:
        print('Trial separation not done (df not calculated)')
    return (wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs, 
            baselineTraces_allTrials_ROIs, all_clusters_dF_whole_trace) 
                               
        
def plot_roi_properties(images, properties, colormaps,underlying_image,vminmax,
                        exp_ID,depth,save_fig = False, save_dir = None,
                        figsize=(10, 6)):
    """ 
    Parameters
    ==========
    
        
    Returns
    =======

    """
    plt.close('all')
    plt.style.use("dark_background")
    
    total_n_images = len(images)
    col_row_n = math.ceil(math.sqrt(total_n_images))
    
    fig1, ax1 = plt.subplots(ncols=col_row_n, nrows=col_row_n, sharex=True, 
                             sharey=True,
                            facecolor='k', edgecolor='w',figsize=figsize)
    depthstr = 'Z: %d' % depth
    figtitle = 'ROIs summary: ' + depthstr
    fig1.suptitle(figtitle,fontsize=12)
    
    for idx, ax in enumerate(ax1.reshape(-1)): 
        
        sns.heatmap(underlying_image,cmap='gray',ax=ax,cbar=False)
        
        sns.heatmap(images[idx],alpha=0.8,cmap = colormaps[idx],ax=ax,
                    cbar=True,cbar_kws={'label': properties[idx]},
                    vmin = vminmax[idx][0], vmax=vminmax[idx][1])
        ax.axis('off')
    
    if save_fig:
        # Saving figure
        save_name = 'ROI_props_%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('ROI property images saved')
        

def plot_roi_masks(first_roi_image, second_roi_image,
                       underlying_image,n_roi1, n_roi2,exp_ID,
                       save_fig = False, save_dir = None):
    """ Plots two different cluster images underlying an another common image.
    Parameters
    ==========
    first_clusters_image : numpy array
        An image array where clusters (all from segmentation) have different 
        values.
    
    second_cluster_image : numpy array
        An image array where clusters (the final ones) have different values.
        
    underlying_image : numpy array
        An image which will be underlying the clusters.
        
    Returns
    =======

    """

    plt.close('all')
    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True,
                            facecolor='k', edgecolor='w',figsize=(10, 15))
    
    # All clusters
    sns.heatmap(underlying_image,cmap='gray',ax=ax1[0],cbar=False)
    sns.heatmap(first_roi_image,alpha=0.3,cmap = 'tab20b',ax=ax1[0],
                cbar=False)
    
    ax1[0].axis('off')
    ax1[0].set_title('All ROIs n=%d' % n_roi1)
    
    sns.heatmap(underlying_image,cmap='gray',ax=ax1[1],cbar=False)
    sns.heatmap(second_roi_image,alpha=0.5,cmap = 'tab20b',ax=ax1[1],
                cbar=False) 
    
    ax1[1].axis('off')
    ax1[1].set_title('Final ROIs n=%d' % n_roi2)
    
    if save_fig:
        # Saving figure
        save_name = 'ROIs_%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('ROI images saved')


def plot_raw_responses_stim(responses, rawStimData, exp_ID, save_fig =False, 
                            save_dir = None, ax_to_plot =None):
    """ Gets the required stimulus and imaging parameters.
    Parameters
    ==========
    responses : n x m numpy array
        Response traces along the row dimension. (n ROIs, m time points)
    
    rawStimData : numpy array
        Raw stimulus output data where the frames and stim values are stored.
        
    Returns
    =======
    

    """
    
    adder = np.linspace(0, np.shape(responses)[0]*4, 
                        np.shape(responses)[0])[:,None]
    scaled_responses = responses + adder
    
    # Finding stimulus
    
    stim_frames = rawStimData[:,7]  # Frame information
    stim_vals = rawStimData[:,3] # Stimulus value
    uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
    stim_vals = stim_vals[uniq_frame_id]
    # Make normalized values of stimulus values for plotting
    
    stim_vals = (stim_vals/np.max(np.unique(stim_vals))) \
        *np.max(scaled_responses)/3
    stim_df = pd.DataFrame(stim_vals,columns=['Stimulus'],dtype='float')
    
    resp_df = pd.DataFrame(np.transpose(scaled_responses),dtype='float')
    
    if ax_to_plot is None:
        ax = resp_df.plot(legend=False,alpha=0.8,lw=0.5)
    else:
        ax = ax_to_plot
        resp_df.plot(legend=False,alpha=0.8,lw=0.5,ax=ax_to_plot)
        
    stim_df.plot(dashes=[2, 1],ax=ax,color='w',alpha=.8,lw=2)
    plt.title('Responses (N:%d)' % np.shape(responses)[0])
    
    if save_fig:
        # Saving figure
        save_name = 'ROI_traces%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('All traces figure saved')
    

def calculate_SNR_Corr(base_traces_all_roi, resp_traces_all_roi,
                       rois, epoch_to_exclude = None):
    """ Calculates the signal-to-noise ratio (SNR). Equation taken from
    Kouvalainen et al. 1994 (see calculation of SNR true from SNR estimated).
    Also calculates the correlation between the first and the last trial to 
    estimate the reliability of responses.
    
    
    Parameters
    ==========
    respTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -stimulus epoch-
        
    baselineTraces_allTrials_ROIs : list containing np arrays
        Epoch list of time traces including all trials in the form of:
            -baseline epoch-
            
    rois : list
        A list of ROI_bg instances.
        
    epoch_to_exclude : int 
        Default: None
        Epoch number to exclude when calculating corr and SNR
        
        
    Returns
    =======
    
    SNR_max_matrix : np array
        SNR values for all ROIs.
        
    Corr_matrix : np array
        SNR values for all ROIs.
        
    """
    total_epoch_numbers = len(base_traces_all_roi)
    
    SNR_matrix = np.zeros(shape=(len(rois),total_epoch_numbers))
    Corr_matrix = np.zeros(shape=(len(rois),total_epoch_numbers))
    
    for iROI, roi in enumerate(rois):
        
        for iEpoch, iEpoch_index in enumerate(base_traces_all_roi):
            
            if iEpoch_index == epoch_to_exclude:
                SNR_matrix[iROI,iEpoch] = 0
                Corr_matrix[iROI,iEpoch] = 0
                continue
            
            trial_numbers = np.shape(resp_traces_all_roi[iEpoch_index][iROI])[1]
            
            
            currentBaseTrace = base_traces_all_roi[iEpoch_index][iROI][:,:]
            currentRespTrace =  resp_traces_all_roi[iEpoch_index][iROI][:,:]
            
            # Reliability between all possible combinations of trials
            perm = permutations(range(trial_numbers), 2) 
            coeff =[]
            for iPerm, pair in enumerate(perm):
                curr_coeff, pval = pearsonr(currentRespTrace[:,pair[0]],
                                            currentRespTrace[:,pair[1]])
                coeff.append(curr_coeff)
                
            coeff = np.array(coeff).mean()
            
            noise_std = currentBaseTrace.std(axis=0).mean(axis=0)
            resp_std = currentRespTrace.std(axis=0).mean(axis=0)
            signal_std = resp_std - noise_std
            # SNR calculation taken from
            curr_SNR_true = ((trial_numbers+1)/trial_numbers)*(signal_std/noise_std) - 1/trial_numbers
        #        curr_SNR = (signal_std/noise_std) 
            SNR_matrix[iROI,iEpoch] = curr_SNR_true
            Corr_matrix[iROI,iEpoch] = coeff
        
        roi.SNR = np.nanmax(SNR_matrix[iROI,:])
        roi.reliability = np.nanmax(Corr_matrix[iROI,:])
    
     
    SNR_max_matrix = np.nanmax(SNR_matrix,axis=1) 
    Corr_matrix = np.nanmax(Corr_matrix,axis=1)
    
    return SNR_max_matrix, Corr_matrix

def plot_df_dataset(df, plot_x, properties, save_name = 'ROI_plots_%s', 
                    exp_ID=None, save_fig = False, save_dir=None):
    """ Plots a variable against 3 other variables

    Parameters
    ==========
   
     
    Returns
    =======
    
    
    """
    plt.close('all')
    plt.figure(figsize=(10,5))
    
    
    plt.subplot(221)
    sns.scatterplot(x=plot_x, y=properties[0],alpha=.8,color='grey',
                    data =df)
    plt.xlim(0, df[plot_x].max()+0.1)
    plt.ylim(0, df[properties[0]].max()+0.1)
    
    plt.subplot(222)
    sns.scatterplot(x=plot_x, y=properties[1],alpha=.8,color='grey',
                    data =df)
    plt.xlim(0, df[plot_x].max()+0.1)
    plt.ylim(0, df[properties[1]].max()+0.1)
    
    plt.subplot(223)
    sns.scatterplot(x=plot_x, y=properties[2],alpha=.8,color='grey',
                    data =df) 
    plt.xlim(0, df[plot_x].max()+0.1)
    plt.ylim(0, df[properties[2]].max()+0.1)
    
    plt.subplot(224)
    sns.countplot(properties[3],data =df)
    if save_fig:
            # Saving figure
            save_name = 'ROI_plots_%s' % (exp_ID)
            os.chdir(save_dir)
            plt.savefig('%s.png'% save_name, bbox_inches='tight')
            print('ROI properties saved')
    return None


def make_exp_summary(figtitle,extraction_type,mean_image,roi_image,roi_traces,
                     rawStimData,bf_image,
                     rois_df,rois,stimulus_information,save_fig,current_movie_ID,
                     summary_save_dir):
    """ Plots a summary of experiment with relevant information.
    
    """
    
    
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    colors = [plt.cm.Dark2(4), plt.cm.Dark2(3)]
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(figtitle,fontsize=12)
    
    grid = plt.GridSpec(3, 6, wspace=1, hspace=0.3)
    
    ## ROIs
    ax=plt.subplot(grid[0,:2])
    
    sns.heatmap(mean_image,cmap='gray',ax=ax,cbar=False)
    sns.heatmap(roi_image,alpha=0.3,cmap = 'Set1',ax=ax,
                cbar_kws={'fraction':0.1,
                          'shrink' : 0,
                          'ticks': []})
    ax.axis('off')
    ax.set_title('ROIs')   
    
    ## Raw traces
    ax=plt.subplot(grid[0,2:])
    adder = np.linspace(0, np.shape(roi_traces)[0]*4, 
                            np.shape(roi_traces)[0])[:,None]
    scaled_responses = roi_traces + adder
    # Finding stimulus
    stim_frames = rawStimData[:,7]  # Frame information
    stim_vals = rawStimData[:,3] # Stimulus value
    uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
    stim_vals = stim_vals[uniq_frame_id]
    # Make normalized values of stimulus values for plotting
    stim_vals = (stim_vals/np.max(np.unique(stim_vals))) \
        *np.max(scaled_responses)/3
    stim_df = pd.DataFrame(stim_vals,columns=['Stimulus'],dtype='float')
    resp_df = pd.DataFrame(np.transpose(scaled_responses),dtype='float')
    resp_df.plot(legend=False,alpha=0.8,lw=0.5,ax=ax,cmap='Set1')   
    stim_df.plot(dashes=[1, 1],ax=ax,color='w',alpha=.8,lw=1)
    ax.get_legend().remove()
    ax.axis('off')
    ax.set_title('Raw traces')
    
    ## BF masks
    ax=plt.subplot(grid[1,:2])
    sns.heatmap(mean_image,cmap='gray',ax=ax,cbar=False)
    sns.heatmap(bf_image,cmap='plasma',
                cbar_kws={'ticks': np.unique(bf_image[~np.isnan(bf_image)]),
                          'fraction':0.1,
                          'shrink' : 1,
                          'label': 'Hz',},alpha=0.5,vmin=0.1,vmax=3)
    
#     sns.heatmap(bf_image,cmap='plasma',
#                cbar_kws={'ticks': np.unique(bf_image[~np.isnan(bf_image)]),
#                          'fraction':0.1,
#                          'shrink' : 1,
#                          'label': 'Hz',},alpha=0.5,vmin=0.1,vmax=3,
#                          norm=LogNorm(vmin=0.1,vmax=3))
    ax.axis('off')
    ax.set_title('BF map')   
    
    ## Histogram
    ax=plt.subplot(grid[1,2])
    chart = sns.countplot('BF',data =rois_df,palette='plasma')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, 
                          fontweight='light')
    leg = chart.legend()
    leg.remove()
    
    
    ## Tuning curve
    ax=plt.subplot(grid[1,3:])
    # Plot tuning curves
    tunings = np.squeeze(list(map(lambda roi : roi.TF_curve_resp, rois)))
    mean_t = np.mean(tunings,axis=0)
    std_t = np.std(tunings,axis=0)
    ub = mean_t + std_t
    lb = mean_t - std_t
    # Tuning curve
    
    TF_stim = rois[0].TF_curve_stim
    ax.fill_between(TF_stim, ub, lb,
                     color=colors[1], alpha=.2)
    
    ax.plot(TF_stim,mean_t,'-o',lw=4,color=colors[1],
            markersize=10)
    #ax.plot(TF_stim,tunings.T,alpha=0.3,lw=1)
    
    ax.set_xscale('log') 
    ax.set_title('Frequency tuning curve')  
    ax.set_xlabel('Hz')
    ax.set_ylabel('dF')
    ax.set_xlim((ax.get_xlim()[0],10)) 
    
    ## Plotting all tuning curves
    if len(rois) > 100:
        ax=plt.subplot(grid[2,:])
    elif len(rois) > 75:
        ax=plt.subplot(grid[2,:4])
    elif len(rois) > 50:
        ax=plt.subplot(grid[2,:3])
    elif len(rois) > 25:
        ax=plt.subplot(grid[2,:2])
    else:
        ax=plt.subplot(grid[2,1])
        
        
    non_edge_stims = (stimulus_information['stim_type'] != 50)
    uniq_freq_nums = len(np.where(np.unique(stimulus_information['epoch_frequency'][non_edge_stims])>0)[0])
    bfs = np.squeeze(list(map(lambda roi : roi.BF, rois)))
    sorted_indices = np.argsort(bfs)
    tf_tunings = tunings[sorted_indices,:]
    plot_tf_array = np.zeros(shape=(np.shape(tf_tunings)[0],np.shape(tf_tunings)[0]*np.shape(tf_tunings)[1]))
    plot_tf_array[:] = np.nan
    for i in range(np.shape(tf_tunings)[0]):
        
        curr_data = tf_tunings[i,:]
        curr_data = curr_data + np.mod(i,9)
        curve_start = i*np.shape(tf_tunings)[1] - (i*(uniq_freq_nums-1))
        plot_tf_array[i,curve_start:curve_start+uniq_freq_nums] = curr_data
        
    ax.plot(np.transpose(plot_tf_array),'-o',linewidth=2.0, alpha=.8,
            color=colors[1],markersize=0.4)
    ax.axis('off')
    ax.set_title('ROI tuning curves N: %s' % len(rois))
    
    if save_fig:
        # Saving figure 
        save_name = 'Summary_%s_%s' % (current_movie_ID, extraction_type)
        os.chdir(summary_save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
                                
        
def batchPixelAnalysis_Plots(alignedDataDir, pixel_analysis_params):
    """ Makes pixel-wise plots of DSI, CSI, TF

    Parameters
    ==========
    alignedDataDir: str
        Path into the directory where the motion corrected data will be plotted
    
    pixel_analysis_params : list
        A list of parameters to use in pixel analysis.
    

     
    Returns
    =======
    
    """
    print('Running pixel analysis and saving figures...\n')
    


   
    all_folders = os.listdir(alignedDataDir)
    for folder_name in all_folders:
        # Enter if it is an experiment folder
        if 'fly' in folder_name.lower(): 
            experiment_path = os.path.join(alignedDataDir,folder_name)
            current_exp_ID = folder_name.lower()
            
            print("-> Processing %s...\n" % (current_exp_ID))
            
            # Finding T-series for aligning
            t_series_names = [file_n for file_n in os.listdir(experiment_path)\
                                 if 'tseries' in file_n.lower() or \
                                 't-series' in file_n.lower()]
            for iTseries, t_name in enumerate(t_series_names):
                t_series_path = os.path.join(experiment_path,t_name)
                
                stimOutPath = os.path.join(t_series_path, '_stimulus_output_*')
                stimOutFile_path = (glob.glob(stimOutPath))[0]
                (stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile_path, 
                                                      skipHeader=1)
                if (('DriftingSquare' in stimType) or \
                    ('DriftingSine' in stimType)) and ('Hz' in stimType):
                    print("--> TF stimulus found: %s...\n" % (t_name))
                    dataDir = os.path.join(t_series_path,'motCorr.sima')
                    
                    current_movie_ID = current_exp_ID + '-'+t_name
                    run_pixel_analysis(dataDir,current_movie_ID,pixel_analysis_params)
                else:
                    print("--> No appropriate stim found: %s...\n" % (t_name))
                    continue
                    
                print("--> Figures successfully saved for: %s...\n" % (t_name))
                
                    
                    
    print("---Figures are saved, check the log for errors---")  
    return        




