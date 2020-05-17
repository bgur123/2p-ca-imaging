#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:40:02 2019

@author: burakgur
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:07:41 2019

@author: burakgur
"""

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
import sima
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr

functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
from core_functions import readStimOut, readStimInformation, getEpochCount
from core_functions import divideEpochs
from xmlUtilities import getFramePeriod, getLayerPosition,getPixelSize

#%% Cluster analysis variables
cluster_max_1d_size = 5 # in microns
cluster_min_1d_size = 2
DSI_threshold = 0.2

#%%
#from xmlUtilities import getFramePeriod, getLayerPosition, getMicRelativeTime
alignedDataDir = '/Users/burakgur/2p/Python_data/motion_corrected/chosen_alignment'
stimInputDir = "/Users/burakgur/2p/Python_data/stimulus_types"


use_otsu = False
crop_movie = False
smooth_movie = True
sigma = 0.75 # For smoothing
#%%
tSeriesName = '190215bg_fly1/TSeries-02152019-1231-001'
flyPath = os.path.join(alignedDataDir,tSeriesName,'motCorr.sima') 
movie_path = os.path.join(flyPath,'motCorr.tif')

#%% Load the image and get the parameters
raw_movie = io.imread(movie_path)
mean_image = raw_movie.mean(0)

frame_num = raw_movie.shape[0]

scaled_mean_image = mean_image/mean_image.max(); #Scale the image 0-1

#%% dF/F
#def exponenial_func(x, a, b, c):
#    return a*np.exp(-b*x)+c
#
#frame_num = np.shape(filtered_video)[0]
#series_x = np.shape(filtered_video)[1]
#series_y = np.shape(filtered_video)[2]
#
#dF_time_series = np.zeros(shape=(frame_num,series_x,series_y))
#
#for iX in range(np.shape(filtered_video)[1]):
#    for iY in range(np.shape(filtered_video)[2]):
#        curr_pixel_trace = filtered_video[:,iX,iY]
#        time_trace = range(frame_num)
#        
##        popt, pcov = curve_fit(exponenial_func,time_trace, curr_pixel_trace, 
##                               p0=(1, 0.0001, 1),maxfev=4000)
##        
##        xx = np.arange(0, frame_num)
##        yy = exponenial_func(xx, *popt)
#        mean_val = np.nanmean(curr_pixel_trace)
#        dF_pixel_trace = (curr_pixel_trace - mean_val)/mean_val
#        dF_time_series[:,iX,iY] =  dF_pixel_trace
        
#%%

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
filtered_image = filters.gaussian(scaled_mean_image, sigma=sigma)
plt.imshow(filtered_image)
#%%
filtered_video = filters.gaussian(raw_movie, sigma=sigma)
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
t_series_path = os.path.dirname(flyPath)
xmlPath = os.path.join(t_series_path, '*-???.xml')
xmlFile = (glob.glob(xmlPath))[0]

#  Finding the frame period (1/FPS) and layer position
framePeriod = getFramePeriod(xmlFile=xmlFile)
frameRate = 1/framePeriod
layerPosition = getLayerPosition(xmlFile=xmlFile)
depth = layerPosition[2]

# Stimulus output information

stimOutPath = os.path.join(t_series_path, '_stimulus_output_*')
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
stimulus_information['stim_type'] =  \
    np.asfarray(stimulus_data['Stimulus.stimtype'])
if 50 in stimulus_information['stim_type']:
    edge_exists = True
    print('Edge epochs found')
else:
    edge_exists = False
     

#%% Define movie and other parameters
    


if smooth_movie:
    selected_movie = raw_movie
else:
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
               


#%%
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
    
#plt.figure(1)
#plt.title('SNR')
SNR_matrix[np.isnan(SNR_matrix)] = np.nanmin(SNR_matrix)
#sns.heatmap(SNR_matrix.max(axis=2),cbar_kws={'label': 'SNR'})
SNR_max_matrix = SNR_matrix.max(axis=2)
#plt.axis('off')

#%%
#%% Max resp and DSI
maxResp = {}
MaxResp_matrix = np.zeros(shape=(mov_xDim,mov_yDim,total_epoch_numbers))
for index, iEpoch in enumerate(trialCoor):
    
    currentEpoch = trialCoor[iEpoch]
    
    trial_numbers = len(currentEpoch)
  
    curr_max =  np.nanmax(np.nanmean(respTraces_allTrials[iEpoch][:,:,:,:],axis=3),axis=0)
    
    maxResp[iEpoch] = curr_max
    MaxResp_matrix[:,:,index] = curr_max
    

# Edge excluded 
if edge_exists:
    MaxResp_matrix_with_edge = copy.deepcopy(MaxResp_matrix)
    # Find edge epochs
    edge_epochs = np.where(stimulus_information['stim_type']==50)[0]
    edge_epochs = edge_epochs - 1
    
    for iEdgeEpoch in edge_epochs:
        MaxResp_matrix[:,:,iEdgeEpoch] = 0
#%% TF MAP

epoch_directions = stimulus_information['epoch_dir']
maxEpochIdx_matrix = np.argmax(MaxResp_matrix,axis=2) # Finding pixel-wise max epochs
maxEpochIdx_matrix = maxEpochIdx_matrix + 1 # To assign numbers like epoch numbers




max_resp_matrix = np.nanmax(MaxResp_matrix,axis=2) # getting the max responses
TF_tuning_image = copy.deepcopy(max_resp_matrix)
DSI_image = copy.deepcopy(max_resp_matrix)[:]

#%% DSI and TF

for iEpoch in (trialCoor):
    current_epoch_type = stimulus_information['stim_type'][iEpoch]
    
        
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

#%%
if edge_exists:
    
    # Find edge epochs
    edge_epochs = np.where(stimulus_information['stim_type']==50)[0]
    edge_epochs = edge_epochs-1
    if len(edge_epochs) == 2: # 2 edges exist 
        ON_resp = np.zeros(shape=(mov_xDim,mov_yDim,2))
        OFF_resp = np.zeros(shape=(mov_xDim,mov_yDim,2))
        CSI_image = np.zeros(shape=(mov_xDim,mov_yDim))
        first_dir_trace = MaxResp_matrix_with_edge[:,:,edge_epochs[0]]
        opposite_dir_trace = MaxResp_matrix_with_edge[:,:,edge_epochs[1]]
        DSI_image = ((first_dir_trace - opposite_dir_trace)\
                                    /(first_dir_trace + opposite_dir_trace))
        half_dur_frames = int((round(frameRate * epochDur[edge_epochs[0]+1]))/2)
        for index, epoch in enumerate(edge_epochs):
            
            
            OFF_resp[:,:,index] = np.nanmax(np.nanmean(respTraces_allTrials[epoch+1][:half_dur_frames,:,:,:],axis=3),axis=0)
            ON_resp[:,:,index] = np.nanmax(np.nanmean(respTraces_allTrials[epoch+1][half_dur_frames:,:,:,:],axis=3),axis=0)
        
        
        CSI_image[DSI_image>0] = (ON_resp[:,:,0][DSI_image>0] - OFF_resp[:,:,0][DSI_image>0])/(ON_resp[:,:,0][DSI_image>0] + OFF_resp[:,:,0][DSI_image>0])
        CSI_image[DSI_image<0] = (ON_resp[:,:,1][DSI_image<0] - OFF_resp[:,:,1][DSI_image<0])/(ON_resp[:,:,1][DSI_image<0] + OFF_resp[:,:,1][DSI_image<0])
        
    
#%%

plt.style.use("dark_background")
fig2, ax2 = plt.subplots(ncols=3,sharex=True, sharey=True,
                        facecolor='k', edgecolor='w',figsize=(12, 5))
figtitle = 'Summary' 
fig2.suptitle(figtitle,fontsize=12)
sns.heatmap(DSI_image,cbar_kws={'label': 'DSI'},cmap = 'RdBu',ax=ax2[0],center=0)
ax2[0].axis('off')
sns.heatmap(SNR_max_matrix,cbar_kws={'label': 'SNR'},ax=ax2[1])
ax2[1].axis('off')
sns.heatmap(CSI_image,cbar_kws={'label': 'CSI'},cmap = 'inferno',ax=ax2[2],center=0)
ax2[2].axis('off')


#%% Visualizing TF tuning
# use an RGB image strategy?
unique_vals = np.unique(TF_tuning_image_noNaN)

TF_tuning_R = np.zeros((mov_xDim,mov_yDim),'uint8')
TF_tuning_G = np.zeros((mov_xDim,mov_yDim),'uint8')
TF_tuning_B = np.zeros((mov_xDim,mov_yDim),'uint8')
TF_tuning_RGB_SNR = np.zeros((mov_xDim,mov_yDim,4), 'uint8')
TF_tuning_RGB_CSI = np.zeros((mov_xDim,mov_yDim,4), 'uint8')
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
    
color_map_R = np.zeros((len(unique_vals),2),'uint8')
color_map_G = np.zeros((len(unique_vals),2),'uint8')
color_map_B = np.zeros((len(unique_vals),2),'uint8')

color_map_RGB =np.zeros((len(unique_vals),2,3),'uint8')
color_counter = 0
for idx , value in enumerate(np.unique(TF_tuning_image_noNaN)):
    if value == -1:
        curr_color = [0,0,0]
    else:
        curr_color = color_list[color_counter,:]
        color_counter = color_counter+ 1
        
    color_map_R[idx,:] = curr_color[0]
    color_map_G[idx,:] = curr_color[1]
    color_map_B[idx,:] = curr_color[2]
         
    TF_tuning_R[TF_tuning_image_noNaN==value] = curr_color[0]
    TF_tuning_G[TF_tuning_image_noNaN==value] = curr_color[1]
    TF_tuning_B[TF_tuning_image_noNaN==value] = curr_color[2]

color_map_RGB[:,:,0] = color_map_R
color_map_RGB[:,:,1] = color_map_G
color_map_RGB[:,:,2] = color_map_B

TF_tuning_RGB_SNR[:,:,0] = TF_tuning_R
TF_tuning_RGB_SNR[:,:,1] = TF_tuning_G
TF_tuning_RGB_SNR[:,:,2] = TF_tuning_B

SNR_pixels = SNR_matrix.max(axis=2)
TF_tuning_RGB_SNR[:,:,3] = 255*(SNR_pixels - np.min(SNR_pixels))/np.ptp(SNR_pixels)


TF_tuning_RGB_CSI[:,:,0] = TF_tuning_R
TF_tuning_RGB_CSI[:,:,1] = TF_tuning_G
TF_tuning_RGB_CSI[:,:,2] = TF_tuning_B

CSI_im_copy = copy.deepcopy(CSI_image)
CSI_im_copy[np.isnan(CSI_im_copy)] = 0
CSI_im_copy = np.abs(CSI_im_copy)
TF_tuning_RGB_CSI[:,:,3] = 255*(CSI_im_copy - np.min(CSI_im_copy))/np.ptp(CSI_im_copy)


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

plt.figure(figsize=(5, 5))
grid = plt.GridSpec(3, 8, wspace=0.4, hspace=0.3)
plt.subplot(grid[:,0])
colormap_Hz = plt.imshow(color_map_RGB)
plt.yticks(np.arange(len(unique_vals)),unique_vals)
plt.xticks([])
plt.ylabel('Temporal Frequency (Hz)')
plt.subplot(grid[:,1:])
tuning_plot = plt.imshow(TF_tuning_RGB_DSI)
plt.title('DSI based')
plt.axis('off')

#%%
plt.style.use("dark_background")
plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.imshow(mean_image)  
plt.title('Mean image')
plt.axis('off')

plt.subplot(222)
plt.imshow(TF_tuning_RGB_DSI)  
plt.title('DSI based')
plt.axis('off')
plt.subplot(223)
plt.imshow(TF_tuning_RGB_CSI)  
plt.title('CSI based')
plt.axis('off')
plt.subplot(224)
plt.title('SNR based')
plt.imshow(TF_tuning_RGB_SNR, alpha=.9)  
plt.axis('off')


#%% Cluster analysis
selected_movie_dir = os.path.join(flyPath,'cluster.sima')

#%% Clusters by sima

epochs_to_use = [1,2,3,4,5,6,7,13,14,15,16,17,18,19,25,26]
#epochs_to_use = [25,26]

epoch_frames = np.zeros(shape=np.shape(epochs_to_use))

# Figuring out how many frames there are
for index, epoch in enumerate(epochs_to_use):
    epoch_frames[index] = np.shape(wholeTraces_allTrials[epoch])[0]
    
    
cluster_movie = np.zeros(shape=(int(epoch_frames.sum()),1,mov_xDim,mov_yDim,1))
whole_movie = np.zeros(shape=(frame_num,1,mov_xDim,mov_yDim,1))
whole_movie[:,0,:,:,0]= copy.deepcopy(filtered_video)
startFrame = 0
for index, epoch in enumerate(epochs_to_use):
    if index>0:
        startFrame =  endFrame 
    endFrame = startFrame + epoch_frames[index]
    cluster_movie[int(startFrame):int(endFrame),0,:,:,0] = wholeTraces_allTrials[epoch].mean(axis=3)


movie_for_cluster = whole_movie
b = sima.Sequence.create('ndarray',movie_for_cluster)

cluster_dataset = sima.ImagingDataset([b],None)
cluster_dataset.export_frames([[[os.path.join(selected_movie_dir,'cluster_vid.tif')]]],
                              fill_gaps=True,scale_values=True)
#%%
import math
x_size, y_size, pixelArea = getPixelSize(xmlFile)
x_max_size = cluster_max_1d_size/x_size
y_max_size = cluster_max_1d_size/y_size
area_max = int(math.pow(x_max_size/2, 2))
area_min = int(cluster_min_1d_size/x_size)

segmentation_approach = sima.segment.STICA(channel = 0,components=45,mu=0.1)
segmentation_approach.append(sima.segment.SparseROIsFromMasks(min_size=area_min,
                                                              smooth_size=8))
#segmentation_approach.append(sima.segment.MergeOverlapping(threshold=0.90))
#segmentation_approach.append(sima.segment.SmoothROIBoundaries())
size_filter = sima.segment.ROIFilter(lambda roi: roi.size >= area_min and roi.size <= area_max)
segmentation_approach.append(size_filter)

rois = cluster_dataset.segment(segmentation_approach, 'auto_ROIs')
initial_cluster_num = len(rois)
print('Number of initial clusters: %d' % initial_cluster_num)
#%%
all_masks_aggregate = np.zeros(shape=(mov_xDim,mov_yDim))
for index, roi in enumerate(rois):
    curr_mask = np.array(roi)[0,:,:]
    
    all_masks_aggregate[curr_mask] = index+1
#    if SNR_max_matrix[index] > 3:
#        plt.imshow(mean_image)
#        plt.imshow(curr_mask, alpha=.5)
#        plt.pause(0.5)

#%%

#%% Trial separation and dF/F
wholeTraces_allTrials_ROIs = {}
respTraces_allTrials_ROIs = {}
baselineTraces_allTrials_ROIs = {}
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
    wholeTraces_allTrials_ROIs[iEpoch] = {}
    respTraces_allTrials_ROIs[iEpoch] = {}
    baselineTraces_allTrials_ROIs[iEpoch] = {}
    for iCluster, cluster in enumerate(rois):
        wholeTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(trial_len,
                                                                 trial_numbers))
        respTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(resp_len,
                                                             trial_numbers))
        baselineTraces_allTrials_ROIs[iEpoch][iCluster] = np.zeros(shape=(base_len,
                                                                    trial_numbers))
        
        curr_mask = np.array(cluster)[0,:,:]
        for trial_num , current_trial_coor in enumerate(currentEpoch):
            
            trialStart = current_trial_coor[0][0]
            trialEnd = current_trial_coor[0][1]
            
            baselineStart = current_trial_coor[1][0]
            baselineEnd = current_trial_coor[1][1]
            
            respStart = current_trial_coor[1][1]
            epochEnd = current_trial_coor[0][1]
            
            baselineResp_cluster = selected_movie[baselineEnd-base_dur_frame:baselineEnd,curr_mask].mean(axis=1)
            baselineMean = baselineResp_cluster.mean(axis=0)
            
            
            dFF_whole_trace = (selected_movie[trialStart:trialEnd, curr_mask].mean(axis=1) - baselineMean) / baselineMean # Not used
            dFF_resp = (selected_movie[respStart:epochEnd, curr_mask].mean(axis=1) - baselineMean) / baselineMean # Not used

            
            
    #        dffTraces_allTrials[iEpoch].append(dFF[:trial_len,:,:])
            wholeTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= dFF_whole_trace[:trial_len]
            respTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= dFF_resp[:resp_len]
            baselineTraces_allTrials_ROIs[iEpoch][iCluster][:,trial_num]= dFF_whole_trace[:base_len]
        
    print('Epoch %d completed \n' % iEpoch)
               
#%% SNR Clusters

total_epoch_numbers = len(baselineTraces_allTrials_ROIs)


#    total_background_dur = stimulus_information['epochs_duration'][0]
SNR_matrix = np.zeros(shape=(initial_cluster_num,total_epoch_numbers))
Corr_matrix = np.zeros(shape=(initial_cluster_num,total_epoch_numbers))

for iEpoch, iEpoch_index in enumerate(baselineTraces_allTrials_ROIs):
    
    for iCluster in range(initial_cluster_num):
        trial_numbers = np.shape(baselineTraces_allTrials_ROIs[iEpoch_index][iCluster])[1]
    
        currentBaseTrace = baselineTraces_allTrials_ROIs[iEpoch_index][iCluster][:,:]
        currentRespTrace =  respTraces_allTrials_ROIs[iEpoch_index][iCluster][:,:]
        
        coeff, pval = pearsonr(currentRespTrace[:,0],currentRespTrace[:,1])
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
MaxResp_matrix = np.zeros(shape=(initial_cluster_num,total_epoch_numbers))

for iEpoch, iEpoch_index in enumerate(respTraces_allTrials):
    
    for iCluster in range(initial_cluster_num):
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
plt.close('all')
sns.countplot(maxEpochIdx_matrix_without_edge[SNR_max_matrix>2])

#%% Plotting TF responses
#number_to_plot = 1000
#add_to_shift_traces = np.linspace(1,initial_cluster_num,
#                                          initial_cluster_num)
#
#shifted = MaxResp_matrix_without_edge[:,:]+ add_to_shift_traces[:,None]
#plt.plot(np.transpose(shifted[:number_to_plot,:10]))
#
#%%
bool_cluster = (maxEpochIdx_matrix_without_edge==4) 
SNR_clusters =  MaxResp_matrix_without_edge[bool_cluster,:24]
SNR_cluster_num = np.shape(SNR_clusters)[0]
add_to_shift_traces = np.linspace(1,SNR_cluster_num,
                                          SNR_cluster_num)

shifted = SNR_clusters[:,:]+ add_to_shift_traces[:,None]
plt.plot(np.transpose(shifted[:,:]))
#%%
#responding_clusters = 
#%%
# DSI and TF

maxofMax_resp_matrix_wo_edge = np.nanmax(MaxResp_matrix_without_edge,axis=1) 
maxofMax_resp_matrix_all = np.nanmax(MaxResp_matrix_all,axis=1)

TF_tuning_image = copy.deepcopy(maxofMax_resp_matrix_wo_edge) # copy it for keeping nan values
DSI_image = copy.deepcopy(maxofMax_resp_matrix_wo_edge) # copy it for keeping nan values


for iEpoch in (trialCoor):
    current_epoch_type = stimulus_information['stim_type'][iEpoch]
    current_clusters = (maxEpochIdx_matrix_without_edge == iEpoch)
    current_freq = stimulus_information['epoch_frequency'][iEpoch]
    current_dir = stimulus_information['epoch_dir'][iEpoch]
    
    if current_dir == 270:
        current_multiplier = -1
    elif current_dir == 90:
        current_multiplier = 1
    else:
        continue
        
    required_epoch_array = \
        (stimulus_information['epoch_dir'] == ((current_dir+180) % 360)) & \
        (stimulus_information['epoch_frequency'] == current_freq) & \
        (stimulus_information['stim_type'] == current_epoch_type)
        
    opposite_dir_epoch = [epoch_indx for epoch_indx, epoch in enumerate(required_epoch_array) if epoch][0]
    opposite_dir_epoch = opposite_dir_epoch- 1 # To find the real epoch number without the baseline
    
    TF_tuning_image[current_clusters] = current_freq

    opposite_response_trace = MaxResp_matrix_without_edge[:,opposite_dir_epoch]
    DSI_image[current_clusters] = current_multiplier * \
                                ((maxofMax_resp_matrix_wo_edge[current_clusters] - \
                                  opposite_response_trace[current_clusters])\
                                    /(maxofMax_resp_matrix_wo_edge[current_clusters] + \
                                      opposite_response_trace[current_clusters]))



#%%
# CSI, DSI based on Probe stimulus (edges)
if edge_exists:
    
    # Find edge epochs
    edge_epochs = np.where(stimulus_information['stim_type']==50)[0]
    edge_epochs = edge_epochs-1 # To match the array indices from list indices
    epochDur= stimulus_information['epochs_duration']
    if len(edge_epochs) == 2: # 2 edges exist 
        ON_resp = np.zeros(shape=(initial_cluster_num,2))
        OFF_resp = np.zeros(shape=(initial_cluster_num,2))
        CSI_matrix = np.zeros(initial_cluster_num)
        
        bool_edge_max = (maxEpochIdx_matrix_all == edge_epochs[0]+1) | (maxEpochIdx_matrix_all == edge_epochs[1]+1)
        first_dir_trace = MaxResp_matrix_all[bool_edge_max,edge_epochs[0]]
        opposite_dir_trace = MaxResp_matrix_all[bool_edge_max,edge_epochs[1]]
        DSI_edge = ((first_dir_trace - opposite_dir_trace)\
                                    /(first_dir_trace + opposite_dir_trace))
        
        DSI_image[bool_edge_max] = DSI_edge
        half_dur_frames = int((round(frameRate * epochDur[edge_epochs[0]+1]))/2)
        for index, epoch in enumerate(edge_epochs):
           
            for iCluster in range(initial_cluster_num):
                
                
                OFF_resp[iCluster,index] = np.nanmax(np.nanmean(
                        respTraces_allTrials_ROIs[epoch+1][iCluster][:half_dur_frames,:],axis=1),axis=0)
                ON_resp[iCluster,index] = np.nanmax(np.nanmean(
                        respTraces_allTrials_ROIs[epoch+1][iCluster][half_dur_frames:,:],axis=1),axis=0)
        
        CSI_matrix[DSI_image>0] = (ON_resp[:,0][DSI_image>0] - OFF_resp[:,0][DSI_image>0])/(ON_resp[:,0][DSI_image>0] + OFF_resp[:,0][DSI_image>0])
        CSI_matrix[DSI_image<0] = (ON_resp[:,1][DSI_image<0] - OFF_resp[:,1][DSI_image<0])/(ON_resp[:,1][DSI_image<0] + OFF_resp[:,1][DSI_image<0])
        
#%%
      
SNR_masks_aggregate = np.zeros(shape=(mov_xDim,mov_yDim))
for index, roi in enumerate(rois):
    curr_mask = np.array(roi)[0,:,:]
    
    
    if SNR_max_matrix[index] > 2:
        SNR_masks_aggregate[curr_mask] = index+1

DSI_masks_aggregate = np.zeros(shape=(mov_xDim,mov_yDim))
for index, roi in enumerate(rois):
    curr_mask = np.array(roi)[0,:,:]
    
    
    if np.abs(DSI_image[index]) > 0.3:
        DSI_masks_aggregate[curr_mask] = index+1
        
DSI_SNR_masks_aggregate = np.zeros(shape=(mov_xDim,mov_yDim))
for index, roi in enumerate(rois):
    curr_mask = np.array(roi)[0,:,:]
    
    
    if (np.abs(DSI_image[index]) > 0.4) & (SNR_max_matrix[index] > 2)\
    & (np.abs(CSI_matrix)[index] > 0.4):
        DSI_SNR_masks_aggregate[curr_mask] = index+1

COR_masks_agg = np.zeros(shape=(mov_xDim,mov_yDim))
for index, roi in enumerate(rois):
    curr_mask = np.array(roi)[0,:,:]
    
    
    if Corr_matrix[index] > 0.8:
        COR_masks_agg[curr_mask] = index+1
        
CSI_masks_agg = np.zeros(shape=(mov_xDim,mov_yDim))
for index, roi in enumerate(rois):
    curr_mask = np.array(roi)[0,:,:]
    
    
    if np.abs(CSI_matrix)[index] > 0.5:
        CSI_masks_agg[curr_mask] = index+1
        
TF_high_masks_agg = np.zeros(shape=(mov_xDim,mov_yDim))
for index, roi in enumerate(rois):
    curr_mask = np.array(roi)[0,:,:]
    
    
    if (maxEpochIdx_matrix_without_edge[index] == 5) & (SNR_max_matrix[index] > 2):
        TF_high_masks_agg[curr_mask] = index+1
        
TF_low_masks_agg = np.zeros(shape=(mov_xDim,mov_yDim))
for index, roi in enumerate(rois):
    curr_mask = np.array(roi)[0,:,:]
    
    
    if (maxEpochIdx_matrix_without_edge[index] == 1) & (SNR_max_matrix[index] > 2):
        TF_low_masks_agg[curr_mask] = index+1
        
    
#%%
import pandas as pd
#%%
df = pd.DataFrame({'Corr':Corr_matrix,'DSI':DSI_image, 
                        'SNR':SNR_max_matrix,'CSI':np.abs(CSI_matrix),'TF':TF_tuning_image})

#%%
sns.jointplot(x=df["DSI"], y=df["SNR"], kind='scatter',color = 'skyblue')
#%%
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(mean_image) 
plt.imshow(all_masks_aggregate, alpha=.6)  
plt.axis('off')
plt.subplot(222)
plt.imshow(mean_image) 
plt.imshow(SNR_masks_aggregate, alpha=.6)  
plt.axis('off')
plt.subplot(223)
plt.imshow(mean_image) 
plt.imshow(TF_high_masks_agg, alpha=.6,cmap='magma')  
plt.axis('off')
plt.subplot(224)
plt.imshow(mean_image) 
plt.imshow(TF_low_masks_agg, alpha=.6,cmap='magma')  
plt.axis('off')





#%% Determining cluster properties

# Pixel constraints on space
x_max_size = cluster_max_1d_size/x_size
y_max_size = cluster_max_1d_size/y_size
clusters_to_exclude = []
cluster_DSIs = []
cluster_SNRs = []
cluster_CSIs = []
passed_clusters  = []
for iCluster, cluster in enumerate(rois):
    curr_cluster = np.array(cluster)[0,:,:]
    cluster_x_size = np.max(cluster.coords[0][:,0])-np.min(cluster.coords[0][:,0])
    cluster_y_size = np.max(cluster.coords[0][:,1])-np.min(cluster.coords[0][:,1])
    
    cluster_DSI = DSI_image[iCluster]
    cluster_DSIs.append(cluster_DSI)
    
    cluster_SNR = SNR_max_matrix[iCluster]
    cluster_SNRs.append(cluster_SNR)
    
    cluster_CSI = CSI_matrix[iCluster]
    cluster_CSIs.append(cluster_CSI)
    
    if not((cluster_SNR > 2) & (np.abs(cluster_DSI) > 0.2) & \
           (cluster_x_size < x_max_size) & (cluster_y_size < y_max_size)):
        clusters_to_exclude.append(iCluster)
    else:
        passed_clusters.append(curr_cluster)
    

excluded_cluster_num = len(clusters_to_exclude)
passed_cluster_num = initial_cluster_num-excluded_cluster_num
print('Cluster pass ratio: %.2f' % (float(initial_cluster_num-excluded_cluster_num)/float(initial_cluster_num)))
print('Total clusters: %d'% passed_cluster_num)

mask_aggregate = np.zeros(shape=(mov_xDim,mov_yDim))
for index, curr_mask in enumerate(passed_clusters):
    mask_aggregate[curr_mask] = index+1
#    plt.imshow(cropped_image)
#    plt.imshow(curr_mask, alpha=.5)
#    plt.pause(0.5)


dataset = pd.DataFrame({'CSI':cluster_CSIs,'DSI':cluster_DSIs, 'SNR':cluster_SNRs})
#%%
#plt.subplot(121)
sns.jointplot('SNR','DSI',data=dataset)
#plt.subplot(121)
#%%
sns.pairplot(data=dataset)
#%% Get rid of overlapping clusters
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 
   
for iCluster, cluster in enumerate(passed_clusters):
    cluster_pixels_x = np.unique(np.where(cluster)[0]) 
    cluster_pixels_y = np.unique(np.where(cluster)[1])
    
    for cluster_to_compare in 
        a = intersection(cluster_pixels_x,np.unique(np.where(passed_clusters[3])[1]) )
    
    

#%%
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(mean_image) 
plt.imshow(all_masks_aggregate, alpha=.6)  
plt.axis('off')
plt.subplot(222)
plt.imshow(mean_image) 
plt.imshow(mask_aggregate, alpha=.6)  
plt.axis('off')
plt.subplot(223)
plt.imshow(DSI_image,cmap='RdBu') 
plt.imshow(mask_aggregate, alpha=.6)  
plt.axis('off')
plt.subplot(224)
plt.imshow(CSI_image,cmap='viridis') 
plt.imshow(mask_aggregate, alpha=.6)  
plt.axis('off')
