#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:13:34 2019

@author: burakgur
"""

import os
import copy
import sima
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import filters, io
functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
from cluster_analysis_functions import get_stim_xml_params, separate_trials
from cluster_analysis_functions import calculate_pixel_SNR, calculate_pixel_max



#%% Setting the directories
#from xmlUtilities import getFramePeriod, getLayerPosition, getMicRelativeTime
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,'motion_corrected/chosen_alignment')
stimInputDir = os.path.join(initialDirectory,'stimulus_types')
saveOutputDir = os.path.join(initialDirectory,'analyzed_data')
figureSaveDir = os.path.join(initialDirectory,'results')
# Database directory
dataBaseDir = os.path.join(initialDirectory,'database')
metaDataBaseFile = os.path.join(dataBaseDir, 'metaDataBase.txt')
# Directory where the processed data will be stored for further analysis
processedDataStoreDir= \
os.path.join(dataBaseDir,'processed_data')

#%%  Setting the parameters
cluster_analysis_params = {}
cluster_analysis_params['use_otsu'] = False
cluster_analysis_params['cropping'] = False 
cluster_analysis_params['use_smoothing'] = False
cluster_analysis_params['sigma'] = 0.75
cluster_analysis_params['save_figures'] = True
cluster_analysis_params['stim_input_directory'] = stimInputDir
cluster_analysis_params['save_fig_dir'] = figureSaveDir
cluster_analysis_params['save_cluster_movie'] = True
cluster_analysis_params['dff_baseline_dur'] = 1.5
cluster_analysis_params['cluster_max_1d_size_micron'] = 5 # in microns
cluster_analysis_params['cluster_min_1d_size_micron'] = 1 # in microns
cluster_analysis_params['SNR_threshold'] = 0.5
cluster_analysis_params['DSI_threshold'] = 0
cluster_analysis_params['interpolation_rate'] = 10 # in Hz
cluster_analysis_params['save_output_dir'] = saveOutputDir
cluster_analysis_params['function_dir'] = functionPath

#%%
dataDir = os.path.join(alignedDataDir,'190917bg_fly3/TSeries-09172019-1039-001/motCorr.sima')

imageID='Image01';

current_exp_ID ='asd'

#run_cluster_analysis_v3(dataDir,imageID,current_exp_ID,cluster_analysis_params)


#%%

#dff_baseline_type = cluster_analysis_params['dff_baseline_type']
dff_baseline_dur = cluster_analysis_params['dff_baseline_dur'] # in seconds, for calculating dF/F
cluster_max_1d_size_micron = cluster_analysis_params['cluster_max_1d_size_micron'] # in microns
cluster_min_1d_size_micron = cluster_analysis_params['cluster_min_1d_size_micron'] # in microns
DSI_threshold = cluster_analysis_params['DSI_threshold']
SNR_threshold = cluster_analysis_params['SNR_threshold']
sigma = cluster_analysis_params['sigma']
use_otsu = cluster_analysis_params['use_otsu']
crop_movie = cluster_analysis_params['cropping']
save_figures = cluster_analysis_params['save_figures']
stimInputDir = cluster_analysis_params['stim_input_directory']
save_example_movie = cluster_analysis_params['save_cluster_movie']
    
    
#%%

movie_path = os.path.join(dataDir,'motCorr.tif')
save_path = os.path.join(dataDir)

#%% Load the image and get the parameters
raw_time_series = io.imread(movie_path)
time_series = copy.deepcopy(raw_time_series)
time_series = time_series - np.mean(time_series,axis=0) # Do dF/F
# time_series = np.transpose(np.transpose(time_series) - time_series[:,BG_mask].mean(axis=1))
#    time_series = time_series + time_series.min()
#    print('Background subtraction done...\n')
mean_image = time_series.mean(0)
frame_num = time_series.shape[0]

# Get stimulus and xml information
t_series_path = os.path.dirname(dataDir)
 
(stimulus_information, trialCoor, frameRate, depth, x_size, y_size, pixelArea,
        stimType, stimOutFile, epochCount, stimName, layerPosition, stimInputFile,
        xmlFile,trialCount, isRandom, stimInputData) = get_stim_xml_params(t_series_path, stimInputDir)

if 50 in stimulus_information['stim_type']:
    edge_exists = True
    print('---> Edge epochs found')
else:
    edge_exists = False
    print('---> No edge epoch exists')

frame_num = time_series.shape[0]
mov_xDim = time_series.shape[1]
mov_yDim = time_series.shape[2]


dff_baseline_dur_frame = int(round(frameRate * dff_baseline_dur))
# Trial separation and dF/F
(wholeTraces_allTrials, respTraces_allTrials, baselineTraces_allTrials) =\
    separate_trials(time_series,trialCoor,stimulus_information,
                    frameRate,dff_baseline_dur_frame)    
    
#%% Pixel wise analysis
smooth_time_series = filters.gaussian(time_series, sigma=sigma)
(wholeTraces_allTrials_smooth, respTraces_allTrials_smooth, baselineTraces_allTrials_smooth) =\
    separate_trials(smooth_time_series,trialCoor,stimulus_information,
                    frameRate,dff_baseline_dur_frame)
# Signal-to-noise ratio calculations
SNR_image = calculate_pixel_SNR(baselineTraces_allTrials_smooth,respTraces_allTrials_smooth,
                  stimulus_information,frameRate,SNR_mode ='Estimate')

 
# Calculate maximum response
MaxResp_matrix_all_epochs, MaxResp_matrix_without_edge_epochs, \
maxEpochIdx_matrix_without_edge,maxEpochIdx_matrix_all = \
    calculate_pixel_max(respTraces_allTrials_smooth,stimulus_information,edge_exists)

# Getting the max responses for each pixel over epochs
max_resp_matrix_wo_edge = np.nanmax(MaxResp_matrix_without_edge_epochs,axis=2) 
max_resp_matrix_all = np.nanmax(MaxResp_matrix_all_epochs,axis=2)



# DSI and TF
DSI_image = copy.deepcopy(max_resp_matrix_all) # copy it for keeping nan values

for iEpoch in (trialCoor):
    current_epoch_type = stimulus_information['stim_type'][iEpoch]
    current_pixels = (maxEpochIdx_matrix_all == iEpoch) & \
                        (~np.isnan(max_resp_matrix_all))
    current_freq = stimulus_information['epoch_frequency'][iEpoch]
    if ((current_epoch_type != 50) and (current_epoch_type != 61)) or (current_freq ==0):
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
        
    opposite_dir_epoch = [epoch_indx for epoch_indx, epoch in enumerate(required_epoch_array) if epoch][0]
    opposite_dir_epoch = opposite_dir_epoch # To find the real epoch number without the baseline
    # Since a matrix will be indexed which doesn't have any information about the baseline epoch
    

    opposite_response_trace = MaxResp_matrix_all_epochs[:,:,opposite_dir_epoch]
    DSI_image[current_pixels] = multiplier*(np.abs(((max_resp_matrix_all[current_pixels] - \
                                  opposite_response_trace[current_pixels])\
                                    /(max_resp_matrix_all[current_pixels] + \
                                      opposite_response_trace[current_pixels]))))
#%%
plt.style.use("dark_background")
fig2, ax2 = plt.subplots(ncols=3,sharex=True, sharey=True,
                        facecolor='k', edgecolor='w',figsize=(16, 4))


depthstr = 'Z: %d' % depth

figtitle = 'Summary ' + depthstr

fig2.suptitle(figtitle,fontsize=12)
sns.heatmap(mean_image,ax=ax2[0],cbar_kws={'label': 'AU'},cmap='viridis')
#    sns.heatmap(layer_masks,alpha=.2,cmap='Set1',ax=ax2[0],cbar=False)
#    sns.heatmap(BG_mask,alpha=.1,ax=ax2[0],cbar=False)
ax2[0].axis('off')
ax2[0].set_title('Mean image with designated layers')
sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'RdBu_r',ax=ax2[1])
ax2[1].axis('off')


sns.heatmap(SNR_image,cbar_kws={'label': 'SNR'},ax=ax2[2],vmax=6)

ax2[2].axis('off')    



#%%
use_smooth_for_cluster = 1
selected_movie_dir = os.path.join(dataDir,'cluster.sima')
print('Cluster analysis started...\n')
epoch_freqs = stimulus_information['epoch_frequency']
epoch_dirs = stimulus_information['epoch_dir']

# Just taking frequencies < 10
epochs_to_use= np.where((epoch_freqs<10))[0]
epoch_frames = np.zeros(shape=np.shape(epochs_to_use))

# Figuring out how many frames there are
for index, epoch in enumerate(epochs_to_use):
    epoch_frames[index] = np.shape(wholeTraces_allTrials[epoch])[0]
    
cluster_movie = np.zeros(shape=(int(epoch_frames.sum()),1,mov_xDim,mov_yDim,1))

startFrame = 0
for index, epoch in enumerate(epochs_to_use):
    if index>0:
        startFrame =  endFrame 
    endFrame = startFrame + epoch_frames[index]
    if use_smooth_for_cluster:
        cluster_movie[int(startFrame):int(endFrame),0,:,:,0] = \
        wholeTraces_allTrials_smooth[epoch].mean(axis=3)
    else:
        cluster_movie[int(startFrame):int(endFrame),0,:,:,0] = \
        wholeTraces_allTrials[epoch].mean(axis=3)
        

# Create a sima dataset and export the cluster movie
b = sima.Sequence.create('ndarray',cluster_movie)

cluster_dataset = sima.ImagingDataset([b],None)
if use_smooth_for_cluster:
    cluster_dataset.export_frames([[[os.path.join(selected_movie_dir,'cluster_vid_smooth.tif')]]],
                                  fill_gaps=True,scale_values=True)
else:
        
    cluster_dataset.export_frames([[[os.path.join(selected_movie_dir,'cluster_vid.tif')]]],
                                  fill_gaps=True,scale_values=True)