#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:17:51 2019

@author: burakgur
"""

import os
import cv2
import numpy as np
import glob
import copy
import seaborn as sns
from skimage import io
from skimage import filters
from skimage import exposure
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
import matplotlib.colors as colors

functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
from core_functions import readStimOut, readStimInformation, getEpochCount
from core_functions import divideEpochs
from xmlUtilities import getFramePeriod, getLayerPosition


#%%
#from xmlUtilities import getFramePeriod, getLayerPosition, getMicRelativeTime
alignedDataDir = '/Users/burakgur/2p/Python_data/motion_corrected/chosen_alignment'
stimInputDir = "/Users/burakgur/2p/Python_data/stimulus_types"


use_otsu = False
crop_movie = False
#%%
tSeriesName = '190201bg_fly2/TSeries--004'
flyPath = os.path.join(alignedDataDir,tSeriesName) 
movie_path = os.path.join(flyPath,'motCorr.tif')

#%% Load the image and get the parameters
raw_movie = io.imread(movie_path)
mean_image = raw_movie.mean(0)

frame_num = raw_movie.shape[0]

scaled_mean_image = mean_image/mean_image.max(); #Scale the image 0-1
#%% Plotting the mean image
plt.imshow(mean_image)
plt.axis('off')


#%% Crop image, press R to confirm crop
if crop_movie:
    cropping = False
     
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    
    image = img_as_ubyte(scaled_mean_image)
    
    oriImage = copy.deepcopy(image)
     
     
    def mouse_crop(event, x, y, flags, param):
        # grab references to the global variables
        global x_start, y_start, x_end, y_end, cropping
     
        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True
     
        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y
     
        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            cropping = False # cropping is finished
     
            refPoint = [(x_start, y_start), (x_end, y_end)]
     
            if len(refPoint) == 2: #when two points were found
                roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.imshow("Cropped", roi)
                
        return x_start, y_start, x_end, y_end
     
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)
     
    while True:
     
        i = image.copy()
     
        if not cropping:
            cv2.imshow("image", image)
     
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)
        
        key = cv2.waitKey(1)
        
        if key == ord("r"):
            break
     
        
     
    # close all open windows
    cv2.destroyAllWindows()
    
    cropped_image =  mean_image[y_start:y_end , x_start:x_end]
    background_trace =  raw_movie[:,y_start:y_end , x_start:x_end].mean(axis=(1,2))
    cropped_movie =  raw_movie[:,y_start:y_end , x_start:x_end]


#%%
if crop_movie:
    plt.imshow(cropped_image)
    plt.axis('off')
    scaled_mean_image = cropped_image/cropped_image.max();

#%% Gaussian filter
filtered_image = filters.gaussian(scaled_mean_image, sigma=1.5)
plt.imshow(filtered_image)

#%% Otsu thresholding
otsu_threshold_Value = filters.threshold_otsu(filtered_image)



hist, bins_center = exposure.histogram(filtered_image)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(filtered_image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.imshow(filtered_image > otsu_threshold_Value, 
           cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.plot(bins_center, hist, lw=2)
plt.axvline(otsu_threshold_Value, color='k', ls='--')

plt.tight_layout()
plt.show()
#%%
if use_otsu == 1:
    if crop_movie:
        cropped_movie[:,filtered_image < otsu_threshold_Value] = 0
    else:
        raw_movie[:,filtered_image < otsu_threshold_Value] = 0
    
        




#%% Stimulus and xml related
# Finding the xml file and retrieving relevant information
xmlPath = os.path.join(flyPath, '*-???.xml')
xmlFile = (glob.glob(xmlPath))[0]

#  Finding the frame period (1/FPS) and layer position
framePeriod = getFramePeriod(xmlFile=xmlFile)
frameRate = 1/framePeriod
layerPosition = getLayerPosition(xmlFile=xmlFile)
depth = layerPosition[2]

# Stimulus output information

stimOutPath = os.path.join(flyPath, '_stimulus_output_*')
stimOutFile_path = (glob.glob(stimOutPath))[0]
(stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile_path, 
                                      skipHeader=1)
stim_name = stimType.split('\\')[-1] 

stim_frames = rawStimData[:,7]  # Frame information
stim_vals = rawStimData[:,3] # Stimulus value
uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
stim_vals = stim_vals[uniq_frame_id]
stim_vals = stim_vals[:frame_num]

# Stimulus information

(stimInputFile,stimInputData) = readStimInformation(stimType=stimType,
                                                  stimInputDir=stimInputDir)

isRandom = int(stimInputData['Stimulus.randomize'][0])
epochDur = stimInputData['Stimulus.duration']
epochDur = [float(sec) for sec in epochDur]
    
# Finding epoch coordinates and number of trials                                        
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
 
stimulus_information ={}
stimulus_data = stimInputData
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

     

#%% Define movie and other parameters
    



selected_movie = raw_movie
frame_num = selected_movie.shape[0]
mov_xDim = selected_movie.shape[1]
mov_yDim = selected_movie.shape[2]



base_dur = 1.5 # in seconds, for calculating dF/F
base_dur_frame = int(round(frameRate * base_dur))
#%% Trial separation and dF/F
wholeTraces_allTrials = {}
respTraces_allTrials = {}
baselineTraces_allTrials = {}
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
 
        
    trial_len =  min(trial_lens)
    resp_len = int(round(frameRate * current_epoch_dur))+1
    base_len = min(base_lens)
    
    
    wholeTraces_allTrials[iEpoch] = np.zeros(shape=(trial_len,mov_xDim,mov_yDim,
                       trial_numbers))
    respTraces_allTrials[iEpoch] = np.zeros(shape=(resp_len,mov_xDim,mov_yDim,
                       trial_numbers))
    baselineTraces_allTrials[iEpoch] = np.zeros(shape=(base_len,mov_xDim,mov_yDim,
                       trial_numbers))

    for trial_num , current_trial_coor in enumerate(currentEpoch):
        
        trialStart = current_trial_coor[0][0]
        trialEnd = current_trial_coor[0][1]
        
        baselineStart = current_trial_coor[1][0]
        baselineEnd = current_trial_coor[1][1]
        
        respStart = current_trial_coor[1][1]
        epochEnd = current_trial_coor[0][1]
        
        baselineResp = selected_movie[baselineEnd-base_dur_frame:baselineEnd, : , :]
        baselineMean = baselineResp.mean(axis=0)
        
        dFF = (selected_movie[trialStart:trialEnd, : , :] - baselineMean) / baselineMean # Not used
        raw_signal = selected_movie[trialStart:trialEnd, : , :]
        
        currentResp = selected_movie[respStart:epochEnd, : , :]
        currentResp_dFF = (selected_movie[respStart:epochEnd, : , :] - baselineMean) / baselineMean
        
        
#        dffTraces_allTrials[iEpoch].append(dFF[:trial_len,:,:])
        wholeTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:trial_len,:,:]
        respTraces_allTrials[iEpoch][:,:,:,trial_num]= currentResp[:resp_len,:,:]
        baselineTraces_allTrials[iEpoch][:,:,:,trial_num]= raw_signal[:base_len,:,:]
    
    print('Epoch %d completed \n' % iEpoch)
               
#%%Dark plots
#sns.set(style="ticks", context="talk",rc={"font.size":14,"axes.titlesize":14,
#                                          "axes.labelsize":14,'xtick.labelsize':12,
#                                          'ytick.labelsize' : 12,
#                                          'legend.fontsize':12})
#    
#current_palette = sns.color_palette('hls',8)

#%% SNR calculation
# Taken from Kouvalainen et al 1994


total_epoch_numbers = len(wholeTraces_allTrials)
background_dur = stimulus_information['epochs_duration'][0]

SNR_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
for iPlot, iEpoch in enumerate(trialCoor):
    
    currentEpoch = trialCoor[iEpoch]
    
    trial_numbers = len(currentEpoch)
    current_epoch_dur = stimulus_information['epochs_duration'][iEpoch]
    
    if current_epoch_dur < background_dur:
        used_frames = int(round(frameRate * current_epoch_dur))-1
    else:
        used_frames = int(round(frameRate * background_dur))-1
                                              # std
    currentBaseTrace = baselineTraces_allTrials[iEpoch][-used_frames:,:,:,:]
    currentRespTrace =  respTraces_allTrials[iEpoch][:used_frames,:,:,:]
    
    noise_std = currentBaseTrace.std(axis=0).mean(axis=2)
    resp_std = currentRespTrace.std(axis=0).mean(axis=2)
    
    signal_std = resp_std - noise_std
    curr_SNR = ((trial_numbers+1)/trial_numbers)*(signal_std/noise_std) - 1/trial_numbers
    
    SNR_matrix[:,:,iPlot] = curr_SNR
    
    epoch_freq = stimulus_information['epoch_frequency'][iEpoch]
    epoch_dir = stimulus_information['epoch_dir'][iEpoch]
    
#    plt.figure(iPlot)
#    plt.title('%d %s Hz' % (epoch_dir, str(epoch_freq)))
#    sns.heatmap(curr_SNR,cbar_kws={'label': 'SNR'},cmap="magma")
#    
#    plt.axis('off')
    
plt.figure(1)
plt.title('SNR')
SNR_matrix[np.isnan(SNR_matrix)] = np.nanmin(SNR_matrix)
sns.heatmap(SNR_matrix.max(axis=2),cbar_kws={'label': 'SNR'})

plt.axis('off')
    
    
#%% Max resp and DSI
maxResp = {}
MaxResp_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
for iPlot, iEpoch in enumerate(trialCoor):
    
    currentEpoch = trialCoor[iEpoch]
    
    trial_numbers = len(currentEpoch)
  
    curr_max =  np.nanmax(np.nanmean(respTraces_allTrials[iEpoch][:,:,:,:],axis=3),axis=0)
    
    maxResp[iEpoch] = curr_max
    MaxResp_matrix[:,:,iPlot] = curr_max
#%% TF MAP

epoch_directions = stimulus_information['epoch_dir']
maxEpochIdx_matrix = np.argmax(MaxResp_matrix,axis=2) # Finding pixel-wise max epochs
maxEpochIdx_matrix = maxEpochIdx_matrix + 1 # To assign numbers like epoch numbers




max_resp_matrix = np.nanmax(MaxResp_matrix,axis=2) # getting the max responses
TF_tuning_image = copy.deepcopy(max_resp_matrix)
DSI_image = copy.deepcopy(max_resp_matrix)[:]

#%% DSI and TF

for iEpoch in (trialCoor):
    current_pixels = (maxEpochIdx_matrix == iEpoch) & (~np.isnan(max_resp_matrix))
    current_freq = stimulus_information['epoch_frequency'][iEpoch]
    current_dir = stimulus_information['epoch_dir'][iEpoch]
    if current_dir == 270:
        current_multiplier = -1
    else:
        current_multiplier = 1
    required_epoch_array = (stimulus_information['epoch_dir'] == ((current_dir+180) % 360)) & \
        (stimulus_information['epoch_frequency'] == current_freq)
    opposite_dir_epoch = [epoch_indx for epoch_indx, epoch in enumerate(required_epoch_array) if epoch][0]
    opposite_dir_epoch =- 1 # To find the real epoch number without the baseline
    
    TF_tuning_image[current_pixels] = current_freq
  
    
    
    opposite_response_trace = MaxResp_matrix[:,:,opposite_dir_epoch]
    DSI_image[current_pixels] = current_multiplier* ((max_resp_matrix[current_pixels] - opposite_response_trace[current_pixels])\
                                    /(max_resp_matrix[current_pixels] + opposite_response_trace[current_pixels]))
    

TF_tuning_image_noNaN = copy.deepcopy(TF_tuning_image)
TF_tuning_image_noNaN[np.isnan(TF_tuning_image)] = -1

#sns.heatmap(TF_tuning_image,cbar_kws={'label': 'TF'})
#plt.axis('off')
#%%
#TF_tuning_SNR = np.zeros((mov_xDim,mov_yDim,2))
#TF_tuning_SNR[:,:,0] = TF_tuning_image
#TF_tuning_SNR[:,:,1] = SNR_matrix.max(axis=2)
#sns.heatmap(TF_tuning_image,vmin=0.1,vmax=8)

#%%
#plt.style.use("dark_background")
#TF_tuning_image[max_resp_matrix < 3] = 0.01
#hm1 = sns.heatmap(TF_tuning_image,norm=colors.LogNorm( vmin=0.1, vmax=30),cmap='viridis')
#hm1.axis('off')
#sns.heatmap(TF_tuning_image,cmap='Set2')


#%%
plt.style.use("dark_background")
fig2, ax2 = plt.subplots(ncols=2,sharex=True, sharey=True,
                        facecolor='k', edgecolor='w')
figtitle = 'Summary' 
fig2.suptitle(figtitle,fontsize=12)
sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'RdBu',ax=ax2[0])
ax2[0].axis('off')
sns.heatmap(max_resp_matrix,cbar_kws={'label': 'Max signal'},ax=ax2[1])
ax2[1].axis('off')
#%% Visualizing TF tuning
# use an RGB image strategy?
unique_vals = np.unique(TF_tuning_image_noNaN)

TF_tuning_R = np.zeros((mov_xDim,mov_yDim),'uint8')
TF_tuning_G = np.zeros((mov_xDim,mov_yDim),'uint8')
TF_tuning_B = np.zeros((mov_xDim,mov_yDim),'uint8')
TF_tuning_RGB_SNR = np.zeros((mov_xDim,mov_yDim,4), 'uint8')
TF_tuning_RGB_MaxResp = np.zeros((mov_xDim,mov_yDim,4), 'uint8')
TF_tuning_RGB_DSI = np.zeros((mov_xDim,mov_yDim,4), 'uint8')


# Self defined colors 
color_list = np.array([[228,26,28],\
                       [55,126,184],\
                       [77,175,74],\
                       [152,78,163],\
                       [255,127,0],\
                       [255,255,51],\
                       [166,86,40],\
                       [247,129,191],\
                       [0,0,0],\
                       [0,0,0],\
                       [0,0,0],\
                       [0,0,0],\
                       [0,0,0],\
                       [0,0,0],\
                       [0,0,0],\
                       [0,0,0]],dtype='uint8')
for iColor, value in enumerate(np.unique(TF_tuning_image_noNaN)):
    if value == -1:
        curr_color = [0,0,0]
    else:
        curr_color = color_list[iColor-1,:]
         
    TF_tuning_R[TF_tuning_image_noNaN==value] = curr_color[0]
    TF_tuning_G[TF_tuning_image_noNaN==value] = curr_color[1]
    TF_tuning_B[TF_tuning_image_noNaN==value] = curr_color[2]

TF_tuning_RGB_SNR[:,:,0] = TF_tuning_R
TF_tuning_RGB_SNR[:,:,1] = TF_tuning_G
TF_tuning_RGB_SNR[:,:,2] = TF_tuning_B

SNR_pixels = SNR_matrix.max(axis=2)
TF_tuning_RGB_SNR[:,:,3] = 255*(SNR_pixels - np.min(SNR_pixels))/np.ptp(SNR_pixels)


TF_tuning_RGB_MaxResp[:,:,0] = TF_tuning_R
TF_tuning_RGB_MaxResp[:,:,1] = TF_tuning_G
TF_tuning_RGB_MaxResp[:,:,2] = TF_tuning_B

max_resp_matrix[np.isnan(max_resp_matrix)] = np.nanmin(max_resp_matrix)
TF_tuning_RGB_MaxResp[:,:,3] = 255*(max_resp_matrix - np.min(max_resp_matrix))/np.ptp(max_resp_matrix)

TF_tuning_RGB_DSI[:,:,0] = TF_tuning_R
TF_tuning_RGB_DSI[:,:,1] = TF_tuning_G
TF_tuning_RGB_DSI[:,:,2] = TF_tuning_B

DSI_im_copy = copy.deepcopy(DSI_image)
DSI_im_copy[np.isnan(DSI_im_copy)] = 0
DSI_im_copy = np.abs(DSI_im_copy)
TF_tuning_RGB_DSI[:,:,3] = 255*(DSI_im_copy - np.min(DSI_im_copy))/np.ptp(DSI_im_copy)

#%% Cv2 style
alpha_max_resp = 255*(max_resp_matrix - np.min(max_resp_matrix))/np.ptp(max_resp_matrix)
img_BGRA = cv2.merge((TF_tuning_B, TF_tuning_G, TF_tuning_R))

#%%
plt.style.use("dark_background")
fig1, ax = plt.subplots(ncols=3,sharex=True, sharey=True,
                        facecolor='k', edgecolor='w')
figtitle = 'Summary' 
fig1.suptitle(figtitle,fontsize=12)

ax[0].imshow(mean_image)
ax[0].axis('off')
im1 = ax[1].imshow(DSI_image,cmap='RdBu')
ax[1].axis('off')
im2 = ax[2].imshow(TF_tuning_RGB_DSI)
ax[2].axis('off')
#%%
plt.imshow(TF_tuning_RGB_MaxResp)
plt.axis('off')

#%%
plt.imshow(TF_tuning_RGB_DSI)
plt.axis('off')
    
    
#%% BF maps