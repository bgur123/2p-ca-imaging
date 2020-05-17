#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:11:23 2019

@author: burakgur
"""
# %% Importing packages
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from skimage import io
import ROI_mod
from core_functions import saveWorkspace
import cluster_analysis_functions_v2

# %% Setting the directories

initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/current_selected')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '191210_T4T5_4th')
summary_save_dir = os.path.join(initialDirectory,
                                'selected_experiments/current_results')

# %% Parameters to adjust
plt.close('all')

current_exp_ID = '191206bg_fly1'
current_t_series = 'TSeries-12062019-0951-001'
Genotype = 'R64G09-Recomb-Homo'
#Genotype = 'R64G09-Recomb-Hete'
#Genotype = 'R42F06-x2'
Age = '--days'

analysis_type = '2D_edges_find_rois_delay_profile_save'
# '8D_edges_find_rois_save' 
# '2D_edges_find_rois_delay_profile_save'
# 'stripes_delay_profile'
# 'gratings_transfer_rois_save'

#%%

dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)
figure_save_dir = os.path.join(dataDir, 'Results')
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)
current_movie_ID = current_exp_ID + '-' + current_t_series



# %% PART 1: Load movie, get stimulus and imaging information
cluster_analysis_params = {'save_output_dir': saveOutputDir, 'deltaF_method': 'mean', 'save_figs': True, 'corr_t': 0.4,
                           'snr_t': 0.75, 'dsi_t': 0, 'csi_t': 0, 'area_max_micron': 4, 'area_min_micron': 0.5,
                           'cluster_max_1d_size_micron': 4, 'cluster_min_1d_size_micron': 0.6}
save_fig = cluster_analysis_params['save_figs']
movie_path = os.path.join(dataDir, '{t_name}_motCorr.tif'.format(t_name=current_t_series))
raw_time_series = io.imread(movie_path)
time_series = copy.deepcopy(raw_time_series)
mean_image = time_series.mean(0)
# delta F/F used just for plotting
# time_series = (time_series-time_series.mean())/time_series.mean()
frame_num = time_series.shape[0]
mov_xDim = time_series.shape[1]
mov_yDim = time_series.shape[2]

## Get stimulus and xml information
(stimulus_information, trialCoor, frameRate, depth, x_size, y_size, pixelArea,
 stimType, stimOutFile, epochCount, stimName, layerPosition,
 stimInputFile, xmlFile, trialCount, isRandom, stimInputData,
 rawStimData) = \
    cluster_analysis_functions_v2.get_stim_xml_params(dataDir, stimInputDir)
stimulus_information['stim_name'] = stimType.split('\\')[-1]

if isRandom:
    stimulus_information['epoch_adjuster'] = 1
else:
    stimulus_information['epoch_adjuster'] = 0

imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                         'depth' : depth}
experiment_conditions = {'Genotype' : Genotype, 'Age': Age, 
                         'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}



# %% PART 2: Select layers for regions
# Select layers and background
plt.close('all')
plt.style.use("default")
print('\n\nSelect layers for regions')
[layer_masks, layer_names] = cluster_analysis_functions_v2.select_regions(mean_image, image_cmap="viridis",
                                                                          pause_t=10)
# [layer_masks, layer_names] = select_regions(DSI_image, image_cmap ="RdBu_r",
#                                            pause_t=10)

cluster_region_bool = np.zeros(shape=(mov_xDim, mov_yDim))
cluster_region_bool[layer_masks[layer_names.index('Layer1')]] = 1
cluster_region_bool[layer_masks[layer_names.index('Layer2')]] = 1

# %% Run region selection
manual_selection = 0
# Defining cluster properties

if manual_selection:
    [roi_masks, layer_names] = cluster_analysis_functions_v2.select_regions(mean_image, image_cmap="viridis",
                                                                            pause_t=3)
    # Generating an image with all clusters

    all_rois_image = np.zeros(shape=(mov_xDim, mov_yDim))
    all_rois_image[:] = np.nan
    for index, roi in enumerate(roi_masks):
        curr_mask = roi
        all_rois_image[curr_mask] = index + 1
    extraction_type = 'manual'
else:
    # Trial separation and dF/F for cluster video
    (wholeTraces_allTrials_video, respTraces_allTrials, baselineTraces_allTrials) = \
        cluster_analysis_functions_v2.separate_trials_video(time_series, trialCoor, stimulus_information,
                                                            frameRate)
    cluster_dataset = cluster_analysis_functions_v2.generate_cluster_movie(dataDir, stimulus_information,
                                                                           wholeTraces_allTrials_video)
    area_max_micron = cluster_analysis_params['area_max_micron']
    area_min_micron = cluster_analysis_params['area_min_micron']
    area_max = int(math.pow(math.sqrt(area_max_micron) / x_size, 2))
    area_min = int(math.pow(math.sqrt(area_min_micron) / x_size, 2))

    [roi_masks, all_rois_image] = cluster_analysis_functions_v2.find_clusters_STICA(cluster_dataset,
                                                                                    area_min, area_max)
    extraction_type = 'sima_STICA'
#%%
figtitle = 'Summary: %s Gen: %s | Age: %s | Z: %d | Extraction: %s' % \
           (current_exp_ID,
            Genotype, Age,
            depth, extraction_type)
           
# %% Part 3 Generating ROI_bg classes and processing ROI data
## Generate ROI classes
rois = ROI_mod.generate_ROI_instances(roi_masks, layer_masks, layer_names,
                                      mean_image, experiment_info = experiment_conditions, 
                                      imaging_info =imaging_information)

# ROI Signals
(wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,
 baselineTraces_allTrials_ROIs,
 all_clusters_dF_whole_trace) = cluster_analysis_functions_v2.separate_trials_ROI_v3(
    raw_time_series, trialCoor, rois, stimulus_information,
    frameRate, cluster_analysis_params['deltaF_method'],
    max_resp_trial_len='max')

# No dF/F for SNR
(_, respTraces_SNR, baseTraces_SNR, _) = cluster_analysis_functions_v2.separate_trials_ROI_v3(
    raw_time_series, trialCoor, rois, stimulus_information,
    frameRate, cluster_analysis_params['deltaF_method'],
    df_use=False)

# SNR and reliability
if not (stimulus_information['random']):
    epoch_to_exclude = stimulus_information['baseline_epoch']
else:
    epoch_to_exclude = None

[SNR_rois, corr_rois] = cluster_analysis_functions_v2.calculate_SNR_Corr(baseTraces_SNR,
                                                                         respTraces_SNR, rois,
                                                                         epoch_to_exclude=epoch_to_exclude)

# Calculating max responses, DSI, CSI, BF, TF tuning
map(lambda roi: roi.findMaxResponse_all_epochs(), rois)
map(lambda roi: roi.appendStimInfo(stimulus_information,
                                   raw_stim_info=rawStimData), rois)
map(lambda roi: roi.calculate_DSI_PD(method='vector'), rois)
map(lambda roi: roi.calculate_CSI(frameRate=frameRate), rois)
map(lambda roi: roi.set_extraction_type(extraction_type), rois)



# Thresholding
dsi_t = cluster_analysis_params['dsi_t']
corr_t = cluster_analysis_params['corr_t']
snr_t = cluster_analysis_params['snr_t']
threshold_dict = {'DSI': dsi_t,
                  'SNR': snr_t,
                  'reliability': corr_t}

thresholded_rois = ROI_mod.threshold_ROIs(rois, threshold_dict)

# Exclude and separate overlapping clusters
scale_bar_size = int(5 / x_size)
if manual_selection:
    separated_rois = thresholded_rois
    final_roi_image = ROI_mod.get_masks_image(separated_rois)
else:
    cluster_max_1d_size_micron = cluster_analysis_params['cluster_max_1d_size_micron']
    cluster_min_1d_size_micron = cluster_analysis_params['cluster_min_1d_size_micron']
    cluster_1d_max_size_pixel = cluster_max_1d_size_micron / x_size
    cluster_1d_min_size_pixel = cluster_min_1d_size_micron / x_size
    [separated_rois, separated_roi_image] = cluster_analysis_functions_v2.clusters_restrict_size_regions \
        (thresholded_rois, cluster_region_bool,
         cluster_1d_max_size_pixel,
         cluster_1d_min_size_pixel)

    final_roi_image = separated_roi_image

cluster_analysis_functions_v2.plot_roi_masks(all_rois_image, final_roi_image,
                                             mean_image, len(roi_masks), len(separated_rois),
                                             current_movie_ID, save_fig=save_fig, save_dir=figure_save_dir)

# Plot RF
separated_rois = ROI_mod.map_RF(separated_rois,edges=True)
ROI_mod.plot_RFs(separated_rois[140:160])
rfs, screen = ROI_mod.plot_RF_centers_on_screen(separated_rois,prop='PD')

final_rois = separated_rois


# Plot RF map and screen
plt.subplot(121)
PD_image,alpha_image = ROI_mod.generate_colorMasks_properties(final_rois, 'PD')
plt.imshow(PD_image,cmap='hsv',alpha=.5)
plt.colorbar()
plt.imshow(mean_image,cmap='gist_gray',alpha=.4)
plt.title('PD map')
plt.axis('off')
plt.subplot(122)
plt.imshow(rfs,cmap='hsv',alpha=.5)
plt.colorbar()
plt.imshow(screen, cmap='binary', alpha=.3)
plt.title('RF center on screen')
ax = plt.gca()
ax.axis('off')


# %% PART 4: Save data
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
varDict = locals()
pckl_save_name = ('%s_%s' % (current_movie_ID, extraction_type))
saveWorkspace(outDir=cluster_analysis_params['save_output_dir'],
              baseName=pckl_save_name, varDict=varDict, varFile='varSave_cluster_v2.txt',
              extension='.pickle')

print('%s saved...' % pckl_save_name)

