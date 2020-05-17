#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:40:39 2019

@author: burakgur
"""
import os
import time
functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
from cluster_analysis_functions import get_stim_xml_params
from cluster_analysis_functions import separate_trials, calculate_pixel_max
from cluster_analysis_functions import calculate_pixel_SNR
import sima
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd 
import math
from scipy.stats.stats import pearsonr
from skimage import filters, io
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
tID = '181011bg_fly2/TSeries-003'
dataDir = os.path.join(alignedDataDir,tID,'motCorr.sima') 
imageID = 'asd'

movie_path = os.path.join(dataDir,'motCorr.tif')
save_path = os.path.join(dataDir)

#%% Layer information
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
        

#Load the image and get the parameters
raw_time_series = io.imread(movie_path)
time_series = copy.deepcopy(raw_time_series)
#    time_series = np.transpose(np.transpose(time_series) - time_series[:,BG_mask].mean(axis=1))
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
MaxResp_matrix_all_epochs, MaxResp_matrix_without_edge_epochs, maxEpochIdx_matrix_without_edge = \
    calculate_pixel_max(respTraces_allTrials_smooth,stimulus_information,edge_exists)

# Getting the max responses for each pixel over epochs
max_resp_matrix_wo_edge = np.nanmax(MaxResp_matrix_without_edge_epochs,axis=2) 
max_resp_matrix_all = np.nanmax(MaxResp_matrix_all_epochs,axis=2)



# DSI and TF
TF_tuning_image = copy.deepcopy(max_resp_matrix_wo_edge) # copy it for keeping nan values
DSI_image = copy.deepcopy(max_resp_matrix_wo_edge) # copy it for keeping nan values

for iEpoch in (trialCoor):
    current_epoch_type = stimulus_information['stim_type'][iEpoch]
    current_pixels = (maxEpochIdx_matrix_without_edge == iEpoch) & \
                        (~np.isnan(max_resp_matrix_wo_edge))
    current_freq = stimulus_information['epoch_frequency'][iEpoch]
    current_dir = stimulus_information['epoch_dir'][iEpoch]
    
    required_epoch_array = \
        (stimulus_information['epoch_dir'] == ((current_dir+180) % 360)) & \
        (stimulus_information['epoch_frequency'] == current_freq) & \
        (stimulus_information['stim_type'] == current_epoch_type)
        
    opposite_dir_epoch = [epoch_indx for epoch_indx, epoch in enumerate(required_epoch_array) if epoch][0]
    opposite_dir_epoch = opposite_dir_epoch- 1 # To find the real epoch number without the baseline
    # Since a matrix will be indexed which doesn't have any information about the baseline epoch
    
    TF_tuning_image[current_pixels] = current_freq

    opposite_response_trace = MaxResp_matrix_without_edge_epochs[:,:,opposite_dir_epoch]
    DSI_image[current_pixels] = np.abs(((max_resp_matrix_wo_edge[current_pixels] - \
                                  opposite_response_trace[current_pixels])\
                                    /(max_resp_matrix_wo_edge[current_pixels] + \
                                      opposite_response_trace[current_pixels])))
                                


#%%
# 
plt.style.use("dark_background")
fig2, ax2 = plt.subplots(ncols=3,sharex=True, sharey=True,
                        facecolor='k', edgecolor='w',figsize=(16, 4))


depthstr = 'Z: %d' % depth

figtitle = 'Summary ' + depthstr

fig2.suptitle(figtitle,fontsize=12)
sns.heatmap(mean_image,ax=ax2[0],cbar_kws={'label': 'AU'},cmap='viridis')
sns.heatmap(layer_masks,alpha=.2,cmap='Set1',ax=ax2[0],cbar=False)
sns.heatmap(BG_mask,alpha=.1,ax=ax2[0],cbar=False)
ax2[0].axis('off')
ax2[0].set_title('Mean image with designated layers')
sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'magma',ax=ax2[1])
ax2[1].axis('off')


sns.heatmap(SNR_image,cbar_kws={'label': 'SNR'},ax=ax2[2])

ax2[2].axis('off')    

if save_figures:
    # Saving figure
    save_name = 'Summary_TF_DSI_%s' % (imageID)
    os.chdir(save_path)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
    print('Figure saved')
    
#%% Clustering
selected_movie_dir = os.path.join(dataDir,'cluster.sima')
print('Cluster analysis started...\n')
epoch_freqs = stimulus_information['epoch_frequency'][1:]
epoch_dirs = stimulus_information['epoch_dir'][1:]

# Just taking frequencies < 10
epochs_to_use= np.where((epoch_freqs<10))[0]+1
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
    cluster_movie[int(startFrame):int(endFrame),0,:,:,0] = wholeTraces_allTrials[epoch].mean(axis=3)

# Create a sima dataset and export the cluster movie
b = sima.Sequence.create('ndarray',cluster_movie)

cluster_dataset = sima.ImagingDataset([b],None)
cluster_dataset.export_frames([[[os.path.join(selected_movie_dir,'cluster_vid.tif')]]],
                              fill_gaps=True,scale_values=True)

cluster_1d_max_size_pixel = cluster_max_1d_size_micron/x_size
cluster_1d_min_size_pixel = cluster_min_1d_size_micron/x_size
area_max = int(math.pow(2/x_size, 2))  # 4um square max area
#area_min = int(math.pow(cluster_1d_min_size_pixel, 2)) # 1um square min area
area_min = int(math.pow(1.1/x_size, 2)) # 1um square min area


segmentation_approach = sima.segment.STICA(channel = 0,components=45,mu=0.1)
segmentation_approach.append(sima.segment.SparseROIsFromMasks(min_size=area_min,smooth_size=3))
#segmentation_approach.append(sima.segment.MergeOverlapping(threshold=0.90))
#segmentation_approach.append(sima.segment.SmoothROIBoundaries(tolerance=0.1,n_processes=(nCpu - 1)))
size_filter = sima.segment.ROIFilter(lambda roi: roi.size >= area_min and roi.size <= area_max)
segmentation_approach.append(size_filter)
start1 = time.time()
rois = cluster_dataset.segment(segmentation_approach, 'auto_ROIs')
initial_cluster_num = len(rois)
end1 = time.time()
time_passed = end1-start1
print('Clusters found in %d minutes\n' % \
      round(time_passed/60) )
print('Number of initial clusters: %d\n' % initial_cluster_num)

#%%
all_masks_aggregate = np.zeros(shape=(mov_xDim,mov_yDim))
all_masks_aggregate[:] = np.nan
for index, roi in enumerate(rois):
    curr_mask = np.array(roi)[0,:,:]
    
    # Layer filtering
    all_masks_aggregate[curr_mask] = index+1
    
scale_bar_size = int(10/x_size)

all_masks_aggregate[5:7,15:15+scale_bar_size] = index+3

#%% Cluster size and delimited layer based exclusion

passed_clusters  = {}
cluster_number = 0
for iCluster, cluster in enumerate(rois):
    curr_cluster = np.array(cluster)[0,:,:]
    cluster_x_size = np.max(cluster.coords[0][:,0])-np.min(cluster.coords[0][:,0])
    cluster_y_size = np.max(cluster.coords[0][:,1])-np.min(cluster.coords[0][:,1])
    
    mask_inclusion_points =  np.where(curr_cluster * layer_masks_bool)[0]
    if mask_inclusion_points.size == np.where(curr_cluster)[0].size: 
    
        if ((cluster_x_size < cluster_1d_max_size_pixel) & (cluster_y_size < cluster_1d_max_size_pixel) & \
               (cluster_x_size > cluster_1d_min_size_pixel) & (cluster_y_size > cluster_1d_min_size_pixel)):
            
            passed_clusters[cluster_number] = curr_cluster
            cluster_number =  cluster_number + 1
        
passed_cluster_num = len(passed_clusters)
print('Cluster pass ratio: %.2f' % (float(passed_cluster_num)/float(initial_cluster_num)))
print('Total clusters: %d'% passed_cluster_num)

size_excluded_masks = np.zeros(shape=(mov_xDim,mov_yDim))
size_excluded_masks[:] = np.nan
for index, cluster_num in enumerate(passed_clusters):
    size_excluded_masks[passed_clusters[cluster_num]] = index+1

size_excluded_masks[5:7,15:15+scale_bar_size] = index+3


#%% Get rid of overlapping clusters
#Get rid of ROIs that overlap with most ROIs and iterate
all_pre_selected_mask = np.zeros(shape=(mov_xDim,mov_yDim))

pre_selected_clusters_indices = np.arange(len(passed_clusters))
pre_selected_clusters_indices_copy = np.arange(len(passed_clusters))
for index, roi in enumerate(passed_clusters):
    curr_mask = passed_clusters[roi]
    all_pre_selected_mask[curr_mask] += 1
while len(np.where(all_pre_selected_mask>1)[0]) != 0:
    
    for index, roi in enumerate(pre_selected_clusters_indices):
        if pre_selected_clusters_indices[index] != -1:
            curr_mask = passed_clusters[roi]
            non_intersection_matrix = (all_pre_selected_mask[curr_mask] == 1)
            
            if len(np.where(non_intersection_matrix)[0]) == 0: 
                # get rid of cluster if it doesn't have any non overlapping part
                pre_selected_clusters_indices[index] = -1
                all_pre_selected_mask[curr_mask] -= 1
                
            elif (len(np.where(non_intersection_matrix)[0]) != len(all_pre_selected_mask[curr_mask])): 
                # get rid of cluster if it has any overlapping part
                pre_selected_clusters_indices[index] = -1
                all_pre_selected_mask[curr_mask] -= 1
        else:
           continue

# To retrieve some clusters if there are no overlaps 
for iRep in range(100):
    
    for index, roi in enumerate(pre_selected_clusters_indices):
        if pre_selected_clusters_indices[index] == -1:
    #        print(index)
            curr_mask = passed_clusters[pre_selected_clusters_indices_copy[index]]
            non_intersection_matrix = (all_pre_selected_mask[curr_mask] == 0)
            if (len(np.where(non_intersection_matrix)[0]) == len(all_pre_selected_mask[curr_mask])):
                # If there's no cluster here add the cluster back
                print(index)
                pre_selected_clusters_indices[index] = pre_selected_clusters_indices_copy[index]
                all_pre_selected_mask[curr_mask] += 1
                        
#%%
separated_cluster_indices = pre_selected_clusters_indices[pre_selected_clusters_indices != -1]
sep_masks = np.zeros(shape=(mov_xDim,mov_yDim))
sep_masks[:] = np.nan
separated_masks = {}

for index, sep_clus_idx in enumerate(separated_cluster_indices):
    sep_masks[passed_clusters[sep_clus_idx]] = index+1
    
    separated_masks[index] = passed_clusters[sep_clus_idx]

sep_masks[5:7,15:15+scale_bar_size] = index+3


    
#%%
plt.close('all')
plt.figure(1,figsize=(12, 12))
plt.subplot(221)
plt.imshow(mean_image,cmap='gray')
plt.imshow(layer_masks,cmap='Paired',alpha=.4)
plt.text
plt.axis('off')
plt.title('Avg image')

plt.subplot(222)


plt.imshow(mean_image,cmap='gray')
plt.imshow(all_masks_aggregate,cmap='viridis',alpha=.6)
#plt.imshow(layer_masks,cmap='Set1',alpha=.4)

plt.axis('off')
plt.title('All clusters N: %d'% initial_cluster_num)

plt.subplot(223)

plt.imshow(mean_image,cmap='gray')
plt.imshow(size_excluded_masks,cmap='viridis',alpha=.6)
plt.axis('off')
plt.title('After size and layer exclusion N: %d'% passed_cluster_num)

plt.subplot(224)

plt.imshow(mean_image,cmap='gray',alpha=.8)
plt.imshow(sep_masks,cmap='viridis',alpha=.6)

plt.axis('off')
plt.title('After overlap removal N: %d'% len(separated_cluster_indices))

if save_figures:
    # Saving figure
    save_name = 'Summary_clusters_%s' % (imageID)
    os.chdir(save_path)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
    print('Figure saved')
    
#%% Trial separation and dF/F
wholeTraces_allTrials_ROIs = {}
respTraces_allTrials_ROIs = {}
baselineTraces_allTrials_ROIs = {}
background_signal_allTrials = {}
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
    resp_len = int(round(frameRate * current_epoch_dur))+1
    base_len = min(base_lens)
    wholeTraces_allTrials_ROIs[iEpoch] = {}
    respTraces_allTrials_ROIs[iEpoch] = {}
    baselineTraces_allTrials_ROIs[iEpoch] = {}
    for iCluster, cluster_num in enumerate(separated_masks):
        wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                                 trial_numbers))
        respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                             trial_numbers))
        baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                                    trial_numbers))
        
        curr_mask = passed_clusters[cluster_num]
        for trial_num , current_trial_coor in enumerate(currentEpoch):
            
            trialStart = current_trial_coor[0][0]
            trialEnd = current_trial_coor[0][1]
            
            baselineStart = current_trial_coor[1][0]
            baselineEnd = current_trial_coor[1][1]
            
            respStart = current_trial_coor[1][1]
            epochEnd = current_trial_coor[0][1]
            
            baselineResp_cluster = time_series[baselineEnd-dff_baseline_dur_frame:baselineEnd,curr_mask].mean(axis=1)
            baselineMean = baselineResp_cluster.mean(axis=0)
            
            
            dFF_whole_trace = (time_series[trialStart:trialEnd, curr_mask].mean(axis=1) - baselineMean) / baselineMean # Not used
            dFF_resp = (time_series[respStart:epochEnd, curr_mask].mean(axis=1) - baselineMean) / baselineMean # Not used

            
            
    #        dffTraces_allTrials[iEpoch].append(dFF[:trial_len,:,:])
            wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= dFF_whole_trace[:trial_len]
            respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= dFF_resp[:resp_len]
            baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= dFF_whole_trace[:base_len]
        
    print('Epoch %d completed \n' % iEpoch)

#%% SNR and trial average of Clusters

total_epoch_numbers = len(baselineTraces_allTrials_ROIs)

trialAvgAllRoi ={}

#    total_background_dur = stimulus_information['epochs_duration'][0]
SNR_matrix = np.zeros(shape=(len(separated_masks),total_epoch_numbers))
Corr_matrix = np.zeros(shape=(len(separated_masks),total_epoch_numbers))

for iEpoch, iEpoch_index in enumerate(baselineTraces_allTrials_ROIs):
    trialAvgAllRoi[iEpoch_index] = {}
    for iCluster in range(len(separated_masks)):
        trial_numbers = np.shape(baselineTraces_allTrials_ROIs[iEpoch_index][iCluster])[1]
    
        currentBaseTrace = baselineTraces_allTrials_ROIs[iEpoch_index][iCluster][:,:]
        currentRespTrace =  respTraces_allTrials_ROIs[iEpoch_index][iCluster][:,:]
        
        trialAvgAllRoi[iEpoch_index][iCluster] = wholeTraces_allTrials_ROIs[iEpoch_index][iCluster].mean(axis=1)
       
        coeff, pval = pearsonr(currentRespTrace[:,0],currentRespTrace[:,-1])
        noise_std = currentBaseTrace.std(axis=0).mean(axis=0)
        resp_std = currentRespTrace.std(axis=0).mean(axis=0)
        signal_std = resp_std - noise_std
        # SNR calculation taken from
        curr_SNR_true = ((trial_numbers+1)/trial_numbers)*(signal_std/noise_std) - 1/trial_numbers
    #        curr_SNR = (signal_std/noise_std) 
        SNR_matrix[iCluster,iEpoch] = curr_SNR_true
        Corr_matrix[iCluster,iEpoch] = coeff
   
SNR_matrix[np.isnan(SNR_matrix)] = np.nanmin(SNR_matrix) # change nan values with min values

SNR_max_matrix = SNR_matrix.max(axis=1) # Take max SNR for every pixel for every epoch
Corr_matrix = Corr_matrix.max(axis=1)

#%% max resp

# Create an array with m x n x nEpochs
MaxResp_matrix = np.zeros(shape=(len(separated_masks),total_epoch_numbers))

for iEpoch, iEpoch_index in enumerate(respTraces_allTrials):
    for iCluster in range(len(separated_masks)):
        # Find maximum of pixels after trial averaging
        curr_max =  np.max(np.nanmean(respTraces_allTrials_ROIs[iEpoch_index][iCluster][:,:],axis=1),axis=0)
        MaxResp_matrix[iCluster,iEpoch] = curr_max


# Make an additional one with edge and set edge maximum to 0 in the main array
# This is to avoid assigning pixels to edge temporal frequency if they respond
# max in the edge epoch but not in one of the grating epochs.
MaxResp_matrix_all = copy.deepcopy(MaxResp_matrix)
MaxResp_matrix_without_edge = copy.deepcopy(MaxResp_matrix)
if edge_exists:
    
    # Find edge epochs
    edge_epochs = np.where(stimulus_information['stim_type']==50)[0]
    edge_epochs = edge_epochs - 1
    
    for iEdgeEpoch in edge_epochs:
        MaxResp_matrix_without_edge[:,iEdgeEpoch] = -100

# Finding pixel-wise max epochs
maxEpochIdx_matrix_without_edge = np.argmax(MaxResp_matrix_without_edge,axis=1) 
# To assign numbers like epoch numbers
maxEpochIdx_matrix_without_edge = maxEpochIdx_matrix_without_edge + 1 


# Finding pixel-wise max epochs
maxEpochIdx_matrix_all = np.argmax(MaxResp_matrix_all,axis=1) 
# To assign numbers like epoch numbers
maxEpochIdx_matrix_all = maxEpochIdx_matrix_all + 1 

#%%
# DSI and TF

maxofMax_resp_matrix_wo_edge = np.nanmax(MaxResp_matrix_without_edge,axis=1) 
maxofMax_resp_matrix_all = np.nanmax(MaxResp_matrix_all,axis=1)

TF_tuning_clusters = copy.deepcopy(maxofMax_resp_matrix_wo_edge) # copy it for keeping nan values
DSI_clusters = copy.deepcopy(maxofMax_resp_matrix_wo_edge) # copy it for keeping nan values


for iEpoch in (trialCoor):
    current_epoch_type = stimulus_information['stim_type'][iEpoch]
    current_clusters = (maxEpochIdx_matrix_without_edge == iEpoch)
    current_freq = stimulus_information['epoch_frequency'][iEpoch]
    current_dir = stimulus_information['epoch_dir'][iEpoch]
    
        
    required_epoch_array = \
        (stimulus_information['epoch_dir'] == ((current_dir+180) % 360)) & \
        (stimulus_information['epoch_frequency'] == current_freq) & \
        (stimulus_information['stim_type'] == current_epoch_type)
        
    opposite_dir_epoch = [epoch_indx for epoch_indx, epoch in enumerate(required_epoch_array) if epoch][0]
    opposite_dir_epoch = opposite_dir_epoch- 1 # To find the real epoch number without the baseline
    
    TF_tuning_clusters[current_clusters] = current_freq

    opposite_response_trace = MaxResp_matrix_without_edge[:,opposite_dir_epoch]
    DSI_clusters[current_clusters] = np.abs(((maxofMax_resp_matrix_wo_edge[current_clusters] - \
                                  opposite_response_trace[current_clusters])\
                                    /(maxofMax_resp_matrix_wo_edge[current_clusters] + \
                                      opposite_response_trace[current_clusters])))
            
#%%
# CSI, DSI based on Probe stimulus (edges)
if edge_exists:
    
    # Find edge epochs
    edge_epochs = np.where(stimulus_information['stim_type']==50)[0]
    edge_epochs = edge_epochs-1 # To match the array indices from list indices
    epochDur= stimulus_information['epochs_duration']
    if len(edge_epochs) == 2: # 2 edges exist 
        ON_resp = np.zeros(shape=(len(separated_masks),2))
        OFF_resp = np.zeros(shape=(len(separated_masks),2))
        CSI_matrix = np.zeros(len(separated_masks))
        
        bool_edge_max = (maxEpochIdx_matrix_all == edge_epochs[0]+1) | (maxEpochIdx_matrix_all == edge_epochs[1]+1)
        first_dir_trace = MaxResp_matrix_all[bool_edge_max,edge_epochs[0]]
        opposite_dir_trace = MaxResp_matrix_all[bool_edge_max,edge_epochs[1]]
        DSI_edge = ((first_dir_trace - opposite_dir_trace)\
                                    /(first_dir_trace + opposite_dir_trace))
        
        DSI_clusters[bool_edge_max] = DSI_edge
        half_dur_frames = int((round(frameRate * epochDur[edge_epochs[0]+1]))/2)
        for index, epoch in enumerate(edge_epochs):
           
            for iCluster in range(len(separated_masks)):
                
                
                OFF_resp[iCluster,index] = np.nanmax(np.nanmean(
                        respTraces_allTrials_ROIs[epoch+1][iCluster][:half_dur_frames,:],axis=1),axis=0)
                ON_resp[iCluster,index] = np.nanmax(np.nanmean(
                        respTraces_allTrials_ROIs[epoch+1][iCluster][half_dur_frames:,:],axis=1),axis=0)
        
        CSI_matrix[DSI_clusters>0] = (ON_resp[:,0][DSI_clusters>0] - OFF_resp[:,0][DSI_clusters>0])/(ON_resp[:,0][DSI_clusters>0] + OFF_resp[:,0][DSI_clusters>0])
        CSI_matrix[DSI_clusters<0] = (ON_resp[:,1][DSI_clusters<0] - OFF_resp[:,1][DSI_clusters<0])/(ON_resp[:,1][DSI_clusters<0] + OFF_resp[:,1][DSI_clusters<0])
        
#%%

if edge_exists:
    df = pd.DataFrame({'Reliability':Corr_matrix,'DSI':DSI_clusters, 
                    'SNR':SNR_max_matrix,'TF':TF_tuning_clusters,'CSI':CSI_matrix})
#    ax1 = sns.scatterplot(x=df["CSI"], y=df["DSI"],hue = df["Reliability"])
   
    
#    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
   
else:
     df = pd.DataFrame({'Reliability':Corr_matrix,'DSI':DSI_clusters, 
                        'SNR':SNR_max_matrix,'TF':TF_tuning_clusters})
   

#%%
plt.close('all')
plt.style.use("default")
ax1 = sns.scatterplot(x=df["DSI"], y=df["SNR"],hue=df["Reliability"])
#ax1.plot([0.4, 0.4], [0, df["SNR"].max()+0.1], 'k--', color = 'k')
ax1.plot([0, 1], [0.5, 0.5], 'k--', color = 'k')
plt.text(-0.9, 0.6, 'SNR Threshold',fontweight='bold', color='maroon')
#ax1.plot([-0.4, -0.4], [0, df["SNR"].max()+0.1], 'k--', color = 'k')
plt.xlim(0, 1)
plt.ylim(0, df["SNR"].max()+0.1)
#sns.jointplot(x=df["DSI"], y=df["SNR"], kind='scatter',color = 'skyblue')
#sns.jointplot(x=df["CSI"], y=df["SNR"], kind='scatter',color = 'skyblue')
if save_figures:
    # Saving figure
    save_name = 'DSI_SNR_reliability_%s' % (imageID)
    os.chdir(save_path)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
    print('Figure saved')

#%% Selecting clusters for end of analysis
corr_t = 0.5
dsi_t = DSI_threshold
SNR_T = SNR_threshold
#bool_pass = SNR_max_matrix>SNR_threshold
bool_pass = (np.abs(DSI_clusters)>dsi_t) & (Corr_matrix>corr_t) & (SNR_max_matrix>SNR_T)
#bool_pass = (np.abs(DSI_image)>dsi_t) & (Corr_matrix>corr_t) 


selected_cluster_indices = np.where(bool_pass)[0]
selected_clusters_SNR = SNR_max_matrix[bool_pass]
selected_clusters_TF = TF_tuning_clusters[bool_pass]
selected_clusters_DSI = DSI_clusters[bool_pass]
selected_clusters_Corr = Corr_matrix[bool_pass]

if edge_exists:
    selected_clusters_CSI = CSI_matrix[bool_pass]

selected_masks = {}
for index, selected_clus_idx in enumerate(selected_cluster_indices):
    selected_masks[index] = separated_masks[selected_clus_idx]

#%%
selected_df = pd.DataFrame({'DSI':selected_clusters_DSI, 'Reliability':selected_clusters_Corr,
                        'SNR':selected_clusters_SNR,'TF':selected_clusters_TF})

df = pd.DataFrame({'Reliability':Corr_matrix,'DSI':DSI_clusters, 
                        'SNR':SNR_max_matrix,'TF':TF_tuning_clusters})

#%%

ax1 = sns.scatterplot(x=df["DSI"], y=df["SNR"],alpha=.5,color='grey')
ax1.plot([dsi_t, dsi_t], [0, df["SNR"].max()+0.1], 'k--', color = 'k',linewidth=2,alpha=.6)
ax1.plot([0, 1], [0.5, 0.5], 'k--', color = 'k',linewidth=2,alpha=.6)
ax1.plot([-dsi_t, -dsi_t], [0, df["SNR"].max()+0.1], 'k--', color = 'k',linewidth=2,alpha=.6)
plt.xlim(0, 1)
plt.ylim(0, df["SNR"].max()+0.1)

#ax1 = sns.scatterplot(x=selected_df["DSI"], y=selected_df["Reliability"],size=selected_df["SNR"],color = 'maroon',alpha=.6)
ax1 = sns.scatterplot(x=selected_df["DSI"], y=selected_df["SNR"],color = 'darkgreen')
plt.xlim(0, 1)
plt.ylim(0, selected_df["SNR"].max()+0.1)

if save_figures:
    # Saving figure
    save_name = 'Summary_selected_DSI_SNR_%s' % (imageID)
    os.chdir(save_path)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
    print('Figure saved')
    
selected_masks_all = np.zeros(shape=(mov_xDim,mov_yDim))
selected_masks_all[:] = np.nan
for index, roi in enumerate(selected_masks):
    curr_mask = selected_masks[roi]
    selected_masks_all[curr_mask] = index+1
    
selected_masks_all[5:7,5:5+scale_bar_size]=index+2  


#%%
plt.close('all')
plt.imshow(mean_image,cmap='Greys_r',alpha=.9)
plt.imshow(selected_masks_all,cmap='viridis',alpha=.9)
plt.text(9,11,'10 um',color='yellow')

plt.axis('off')
plt.title('After DSI-SNR threshold N: %d' % (index+1))
if save_figures:
    # Saving figure
    save_name = 'Summary_image_selected_clusters_%s' % (imageID)
    os.chdir(save_path)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
    print('Figure saved')
#%%
plt.close('all')
tf_to_plot = []
selected_masks_TF = np.zeros(shape=(mov_xDim,mov_yDim))
for index, roi in enumerate(selected_masks):
    curr_mask = selected_masks[roi]
    if selected_clusters_TF[index] <= 1.5:
        selected_masks_TF[curr_mask] = selected_clusters_TF[index]
        tf_to_plot.append(selected_clusters_TF[index])
#        plt.imshow(curr_mask,alpha=.3)
#        plt.pause(0.3)
    else:
        selected_masks_TF[curr_mask] = 1.5
        tf_to_plot.append(0)
        
selected_masks_TF[selected_masks_TF==0] = np.nan
#for index, roi in enumerate(selected_cluster_indices):
#    curr_mask = passed_clusters[roi]
#    
#    selected_masks_TF[curr_mask] = selected_clusters_TF[index]
#    tf_to_plot.append(selected_clusters_TF[index])

plt.figure(2,figsize=(13, 5))
grid = plt.GridSpec(3, 8, wspace=0.4, hspace=0.3)
plt.subplot(grid[:,:6])
sns.heatmap(selected_masks_TF,cmap='magma',cbar_kws={'label': 'TF (Hz)'},vmin=0.1,
                                                       vmax=1.5,alpha=.8)  

plt.imshow(mean_image,alpha=.8)
#sns.heatmap(mean_image,alpha=.3)
plt.axis('off')
plt.title('Clusters')
plt.subplot(grid[:,6:])
sns.countplot(selected_clusters_TF,palette='magma')
plt.xlabel('TF (Hz)')
plt.ylabel('Cluster count')
if save_figures:
    # Saving figure
    save_name = 'Summary_image_TF_selected_clusters_%s' % (imageID)
    os.chdir(save_path)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
    print('Figure saved')

#%% 
all_epoch_ind = np.arange(np.shape(MaxResp_matrix_without_edge)[1])
non_edge_epoch_ind = np.zeros(np.shape(MaxResp_matrix_without_edge)[1],dtype=bool)
if edge_exists:
    for i, iEpoch in enumerate(edge_epochs):
        non_edge_epoch_ind = (all_epoch_ind == iEpoch) | non_edge_epoch_ind
non_edge_epoch_ind = np.invert(non_edge_epoch_ind)

selected_cluster_TF_tuning_all = MaxResp_matrix_without_edge[selected_cluster_indices,:]
selected_cluster_TF_tuning_no_edge = selected_cluster_TF_tuning_all[:,non_edge_epoch_ind]


#%%
uniq_freq_nums = len(np.where(np.unique(stimulus_information['epoch_frequency'])>0)[0])
selected_cluster_TF_tuning_no_edge = np.zeros(shape=(len(selected_cluster_indices),uniq_freq_nums))
for ind,iCluster in enumerate(selected_cluster_indices):
    
    curr_max_epoch = maxEpochIdx_matrix_all[iCluster]
    current_dir = stimulus_information['epoch_dir'][curr_max_epoch]
    current_epoch_type = stimulus_information['stim_type'][curr_max_epoch]
    
    required_epoch_array = \
        (stimulus_information['epoch_dir'] == ((current_dir+180) % 360)) & \
        (stimulus_information['stim_type'] == current_epoch_type)
        
    selected_cluster_TF_tuning_no_edge[ind,:] = MaxResp_matrix_without_edge[iCluster,required_epoch_array[1:]]

#%%
cluster_layer_information = {}

l1_exist = np.where(np.unique(layer_masks)==1)[0].size == 1
l2_exist = np.where(np.unique(layer_masks)==2)[0].size == 1
l3_exist = np.where(np.unique(layer_masks)==3)[0].size == 1
l4_exist = np.where(np.unique(layer_masks)==4)[0].size == 1


for index, roi in enumerate(selected_masks):
    curr_mask = selected_masks[roi]
    if (l1_exist) & (np.where(curr_mask * L1_mask)[0].size != 0):
        cluster_layer_information[roi] = 'Layer 1'
#            plt.imshow(curr_mask,alpha=.3)
#            plt.pause(0.3)
    elif (l2_exist) & (np.where(curr_mask * L2_mask)[0].size != 0):
        cluster_layer_information[roi] = 'Layer 2'
    elif (l3_exist) & (np.where(curr_mask * L3_mask)[0].size != 0):
        cluster_layer_information[roi] = 'Layer 3'
        
    elif (l4_exist) & (np.where(curr_mask * L4_mask)[0].size != 0):
        cluster_layer_information[roi] = 'Layer 4'
    else:
        cluster_layer_information[roi] = 'NA'
            
cluster_numbers = np.arange(len(selected_masks))
#%% Plotting traces

for ind,iCluster in enumerate(selected_cluster_indices):
    curr_max_epoch = maxEpochIdx_matrix_all[iCluster]
    
    current_epoch_type = stimulus_information['stim_type'][curr_max_epoch]
    current_freq = stimulus_information['epoch_frequency'][curr_max_epoch]
    current_dir = stimulus_information['epoch_dir'][curr_max_epoch]
    
        
    required_epoch_array = \
        (stimulus_information['epoch_dir'] == ((current_dir+180) % 360)) & \
        (stimulus_information['epoch_frequency'] == current_freq) & \
        (stimulus_information['stim_type'] == current_epoch_type)
        
    opposite_dir_epoch = [epoch_indx for epoch_indx, epoch in enumerate(required_epoch_array) if epoch][0]
    
    plt.close('all')
    PD = '%d$^\circ$' % current_dir
    ND = '%d$^\circ$' % ((current_dir+180) % 360)
    plt.plot(trialAvgAllRoi[curr_max_epoch][iCluster],label=PD)
    plt.plot(trialAvgAllRoi[opposite_dir_epoch][iCluster],label=ND)
    plt.title('Cluster %d - %s - %.2f Hz'  % (iCluster,cluster_layer_information[ind],current_freq))
    plt.legend()
    plt.pause(2)
#%% Sorting according to TF
sorted_indices = np.argsort(selected_clusters_TF)
sorted_TF_tuning = selected_cluster_TF_tuning_no_edge[sorted_indices,:]
sorted_DSI = selected_clusters_DSI[sorted_indices]


#%%

tf_tunings = copy.deepcopy(sorted_TF_tuning)
plt.close('all')
plot_tf_array = np.zeros(shape=(np.shape(tf_tunings)[0],np.shape(tf_tunings)[0]*np.shape(tf_tunings)[1]))
plot_tf_array[:] = np.nan
for i in range(np.shape(tf_tunings)[0]):
    
    curr_data = tf_tunings[i,:]
    curr_data = curr_data + np.mod(i,9)
    curve_start = i*np.shape(tf_tunings)[1] - i*8
    plot_tf_array[i,curve_start:curve_start+uniq_freq_nums] = curr_data
    
plt.plot(np.transpose(plot_tf_array),linewidth=2.0, alpha=.8)
plt.axis('off')
plt.title('Cluster TF tuning curves')
if save_figures:
    # Saving figure
    save_name = 'Summary_cluster_TF_tunings_%s' % (imageID)
    os.chdir(save_path)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
    print('Figure saved')
    

#%%


        
#%%
intRate = cluster_analysis_params['interpolation_rate']
# Interpolation of responses to a certain frequency
print('Interpolating to %d Hz' % intRate)
interpolationRate = intRate; # Interpolation rate in Hz
interpolatedAllRoi = interpolateTrialAvgROIs(trialAvgAllRoi=trialAvgAllRoi, 
                                             framePeriod=1/frameRate, 
                                             intRate=interpolationRate)
        
framePeriod=1/frameRate
# locals() needs to be called within the script that
# generates the variables to be saved
varDict = locals()
os.chdir(cluster_analysis_params['function_dir'])
savePath = saveWorkspace(outDir=cluster_analysis_params['save_output_dir'], baseName=imageID,
                     varDict=varDict, varFile='variablesToSave_cluster.txt',
                     extension='.pickle')