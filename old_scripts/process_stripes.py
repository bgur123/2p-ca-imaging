#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:49:53 2019

@author: burakgur
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:21:26 2019

@author: burakgur
"""

# %% Importing packages
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from skimage import filters, io

import ROI_mod
from core_functions import saveWorkspace
import cluster_analysis_functions_v2

# %% Setting the directories

initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/miriam_test')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data')
                             # '191113_T4T5_all_re_analyzed_1')
summary_save_dir = os.path.join(initialDirectory,
                                'selected_experiments/results')

# %% Parameters to adjust
current_exp_ID = '191127_miriam_fly1'
current_t_series = 'TSeries1'
#Genotype = 'R64G09-Recomb-Homo'
# Genotype = 'R64G09-Recomb-Hete'
#Genotype = 'R42F06-x2'
Genotype = 'Miriam_test'
Age = '--days'

# %% PART 1: Process movie and generate cluster movie

dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series
                       , 'motCorr.sima')
figure_save_dir = os.path.join(dataDir, 'Results')
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)

current_movie_ID = current_exp_ID + '-' + current_t_series
cluster_analysis_params = {'save_output_dir': saveOutputDir, 'deltaF_method': 'mean', 'save_figs': True, 'corr_t': 0.4,
                           'snr_t': 0.75, 'dsi_t': 0, 'csi_t': 0, 'area_max_micron': 4, 'area_min_micron': 0.5,
                           'cluster_max_1d_size_micron': 4, 'cluster_min_1d_size_micron': 0.6}

plt.close('all')

save_fig = cluster_analysis_params['save_figs']
movie_path = os.path.join(dataDir, 'motCorr.tif')
raw_time_series = io.imread(movie_path)
time_series = copy.deepcopy(raw_time_series)
# delta F/F used just for plotting
# time_series = (time_series-time_series.mean())/time_series.mean()
frame_num = time_series.shape[0]
mov_xDim = time_series.shape[1]
mov_yDim = time_series.shape[2]

## Get stimulus and xml information
t_series_path = os.path.dirname(dataDir)

(stimulus_information, trialCoor, frameRate, depth, x_size, y_size, pixelArea,
 stimType, stimOutFile, epochCount, stimName, layerPosition,
 stimInputFile, xmlFile, trialCount, isRandom, stimInputData,
 rawStimData) = \
    cluster_analysis_functions_v2.get_stim_xml_params(t_series_path, stimInputDir)
stimulus_information['stim_name'] = stimType.split('\\')[-1]



if 50 in stimulus_information['stim_type']:
    edge_exists = True

    print('---> Edge epochs found')
else:
    edge_exists = False
    print('---> No edge epoch exists')

stimulus_information['edge_exists'] = edge_exists

if isRandom:
    stimulus_information['epoch_adjuster'] = 1
else:
    stimulus_information['epoch_adjuster'] = 0

## Generating pixel maps
sigma = 0.75  # Filtering just for pixel maps not used for further analysis
smooth_time_series = filters.gaussian(time_series, sigma=sigma)
(wholeTraces_allTrials_smooth, respTraces_allTrials_smooth, baselineTraces_allTrials_smooth) = \
    cluster_analysis_functions_v2.separate_trials_video(smooth_time_series, trialCoor, stimulus_information,
                                                        frameRate)

# Calculate maximum response
MaxResp_matrix_all_epochs, MaxResp_matrix_without_edge_epochs, \
maxEpochIdx_matrix_without_edge, maxEpochIdx_matrix_all, \
MeanResp_matrix_all_epochs, maxEpochIdx_matrix_all_mean = \
    cluster_analysis_functions_v2.calculate_pixel_max(respTraces_allTrials_smooth,
                                                      stimulus_information)

max_resp_matrix_all = np.nanmax(MaxResp_matrix_all_epochs, axis=2)
max_resp_matrix_all_mean = np.nanmax(MeanResp_matrix_all_epochs, axis=2)

SNR_image = cluster_analysis_functions_v2.calculate_pixel_SNR(baselineTraces_allTrials_smooth,
                                                              respTraces_allTrials_smooth,
                                                              stimulus_information, frameRate,
                                                              SNR_mode='Estimate')
mean_image = time_series.mean(0)
# plt.close('all')
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
# %% Part 3 Generating ROI_bg classes and processing ROI data
## Generate ROI classes
figtitle = 'Summary: %s Gen: %s | Age: %s | Z: %d | Extraction: %s' % \
           (current_exp_ID,
            Genotype, Age,
            depth, extraction_type)
rois = ROI_mod.generate_ROI_instances(roi_masks, layer_masks, layer_names,
                                      mean_image, Exp_ID=current_exp_ID)

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
map(lambda roi: roi.calculate_DSI_PD(), rois)
map(lambda roi: roi.set_z_depth(depth), rois)
map(lambda roi: roi.set_extraction_type(extraction_type), rois)

# Thresholding
dsi_t = 0
corr_t = 0.4
snr_t = 1
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
                                             max_resp_matrix_all, len(roi_masks), len(separated_rois),
                                             current_movie_ID, save_fig=save_fig, save_dir=figure_save_dir)

# Extract the data
data_to_extract = ['DSI', 'SNR', 'reliability', 'uniq_id',
                       'PD', 'exp_ID', 'stim_name']

all_rois = rois
all_roi_data = ROI_mod.data_to_list(all_rois, data_to_extract)
final_rois = separated_rois

# Summary of current experiment

final_roi_data = ROI_mod.data_to_list(final_rois, data_to_extract)
roi_traces = list(map(lambda roi: roi.df_trace, final_rois))
bf_image = ROI_mod.generate_colorMasks_properties(final_rois, 'BF')
rois_df = pd.DataFrame.from_dict(final_roi_data)
cluster_analysis_functions_v2.make_exp_summary(figtitle, extraction_type,
                                               mean_image, final_roi_image, roi_traces, rawStimData, bf_image,
                                               rois_df, final_rois, stimulus_information, save_fig, current_movie_ID,
                                               summary_save_dir)
for roi in final_rois:
    roi.imaging_info = {'frame_rate' : frameRate, 'pix_size' : x_size}
images = []
if stimulus_information['edge_exists']:
    properties = ['DSI', 'CSI', 'reliability', 'BF']
    images.append(ROI_mod.generate_colorMasks_properties(final_rois, 'DSI'))
    images.append(ROI_mod.generate_colorMasks_properties(final_rois, 'CS'))
    images.append(ROI_mod.generate_colorMasks_properties(final_rois, 'reliability'))
    images.append(ROI_mod.generate_colorMasks_properties(final_rois, 'BF'))

    colormaps = ['RdBu_r', 'PRGn', 'viridis', 'Paired']
    vminmax = [(-1, 1), (-1, 1), (0, 1), (0.1, 5)]
    cluster_analysis_functions_v2.plot_roi_properties(images, properties, colormaps, mean_image, vminmax,
                                                      current_movie_ID, depth, save_fig=True,
                                                      save_dir=figure_save_dir,
                                                      figsize=(12, 6))
else:
    properties = ['DSI', 'reliability', 'reliability', 'BF']

# Generate pandas dataframes

plot_x = 'SNR'
cluster_analysis_functions_v2.plot_df_dataset(rois_df, plot_x, properties, exp_ID=('%s_%s' % (current_movie_ID, extraction_type)),
                                              save_fig=True, save_dir=figure_save_dir)

plt.close('all')
# %% PART 4: Save data
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
varDict = locals()
pckl_save_name = ('%s_%s' % (current_movie_ID, extraction_type))
saveWorkspace(outDir=os.getcwd(),
              baseName=pckl_save_name, varDict=varDict, varFile='varSave_cluster_v2.txt',
              extension='.pickle')
cluster_analysis_params['save_output_dir']
print('%s saved...' % pckl_save_name)

# %% Plot distance to midline
import seaborn as sns
from matplotlib.colors import LogNorm

dist, distmask = ROI_mod.calculate_distance_from_region(final_rois)
dist = np.array(dist) * x_size
distmask = np.array(distmask) * x_size
bf_image = ROI_mod.generate_colorMasks_properties(final_rois, 'BF')
bfs = list(map(lambda roi: roi.BF, final_rois))
dist_df = pd.DataFrame.from_dict({'BF': bfs, 'Distance': dist})
plt.style.use("dark_background")
plt.close('all')
fig = plt.figure(figsize=(12, 4))
fig.suptitle('ROI distances to midline', fontsize=12)
ax1 = plt.subplot(121)
sns.heatmap(distmask, cmap='terrain', ax=ax1,
            cbar_kws={'fraction': 0.05,
                      'shrink': 1,
                      })
sns.heatmap(bf_image, cmap='plasma', alpha=.9, cbar=False, vmin=0.1, vmax=3,
            norm=LogNorm(vmin=0.1, vmax=3))
ax1.axis('off')
ax1.set_title('BF map vs distance')
ax1 = plt.subplot(122)
sns.regplot(x="BF", y="Distance", data=dist_df, x_jitter=.03, ax=ax1,
            color=plt.cm.Dark2(3), fit_reg=True)
ax1.set_xlabel('BF (Hz)')

ax1.set_xscale("log")
ax1.set_xlim((0, 5))
# %% Part 5: Plot all ROI properties and traces separately
plt.close('all')
roi_figure_save_dir = os.path.join(figure_save_dir, 'ROI_summaries')
if not os.path.exists(roi_figure_save_dir):
    os.mkdir(roi_figure_save_dir)

for roi in final_rois:
    ROI_mod.make_ROI_tuning_summary(rois_df, roi)
    if save_fig:
        # Saving figure 
        save_name = '%s_ROI_summary_%d' % (extraction_type, roi.uniq_id)
        os.chdir(roi_figure_save_dir)
        plt.savefig('%s.png' % save_name, bbox_inches='tight')
    plt.close('all')
# %% PLAY AROUND PART - Load a previous dataset
import cPickle
load_path = os.path.join(saveOutputDir, '181019_miriam_fly3-TSeries4_sima_STICA.pickle')
load_path = open(load_path, 'rb')
workspace = cPickle.load(load_path)
final_rois = workspace['final_rois']
# %% remoake summary
# %% Parameters to adjust
plt.close('all')
Genotype = 'R64G09-Recomb-Homo'
# Genotype = 'R64G09-Recomb-Hete'
# Genotype = 'R42F06-x2'
Age = '6days'
current_exp_ID = '190116bg_fly3'
current_t_series = 'TSeries-01162019-1212-003'

dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series
                       , 'motCorr.sima')
## Get stimulus and xml information

t_series_path = os.path.dirname(dataDir)
movie_path = os.path.join(dataDir, 'motCorr.tif')
raw_time_series = io.imread(movie_path)
time_series = copy.deepcopy(raw_time_series)

mean_image = time_series.mean(0)
(stimulus_information, trialCoor, frameRate, depth, x_size, y_size, pixelArea,
 stimType, stimOutFile, epochCount, stimName, layerPosition,
 stimInputFile, xmlFile, trialCount, isRandom, stimInputData,
 rawStimData) = \
    cluster_analysis_functions_v2.get_stim_xml_params(t_series_path, stimInputDir)

extraction_type = 'sima_STICA'
figtitle = 'Summary: %s Gen: %s | Age: %s | Z: %d | Extraction: %s' % \
           (current_exp_ID,
            Genotype, Age,
            depth, extraction_type)
figure_save_dir = os.path.join(dataDir, 'Results')
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)
save_fig = True
current_movie_ID = current_exp_ID + '-' + current_t_series
data_to_extract = ['DSI', 'BF', 'SNR', 'reliability', 'uniq_id',
                   'PD', 'exp_ID', 'stim_name']
final_roi_image = ROI_mod.get_masks_image(final_rois)
final_roi_data = ROI_mod.data_to_list(final_rois, data_to_extract)
roi_traces = list(map(lambda roi: roi.df_trace, final_rois))
bf_image = ROI_mod.generate_colorMasks_properties(final_rois, 'BF')
rois_df = pd.DataFrame.from_dict(final_roi_data)
cluster_analysis_functions_v2.make_exp_summary(figtitle, extraction_type,
                                               mean_image, final_roi_image, roi_traces, rawStimData, bf_image,
                                               rois_df, final_rois, stimulus_information, save_fig, current_movie_ID,
                                               summary_save_dir)
