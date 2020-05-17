#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:33:33 2019

@author: burakgur
"""
# %% Importing packages
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from skimage import filters
from skimage import exposure
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
from skimage import io
import ROI_mod
from core_functions import saveWorkspace
import process_mov_core
from scipy.stats.stats import pearsonr

# %% Setting the directories

initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/200128_T4T5_STF_TF')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '191210_T4T5_4th')
summary_save_dir = os.path.join(alignedDataDir,
                                '_summaries')

# %% Parameters to adjust
plt.close('all')

current_exp_ID = '200129bg_fly3'
current_t_series = 'TSeries-001'
Genotype = 'R64G09-Recomb-lexAopGC6f'
Age = '2-4'
Sex = 'f'

analysis_type = 'gratings_transfer_rois_save'
use_avg_data_for_roi_extract = False
plot_roi_summ = False
# '8D_edges_find_rois_save' 
# 'luminance_edges_OFF' 
# 'luminance_edges_ON' 
# 'gratings_transfer_rois_save'

# ROI transfer
transfer_data_name = '200129bg_fly3-TSeries-002_sima_STICA.pickle'

# ROI selection video
use_other_for_roiExtraction = True
roiExtraction_tseries = 'TSeries-002'

#%%

dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)
figure_save_dir = os.path.join(dataDir, 'Results')
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)
current_movie_ID = current_exp_ID + '-' + current_t_series



# %% PART 1: Load movie, get stimulus and imaging information



save_fig = True
try:
    movie_path = os.path.join(dataDir, '{t_name}_motCorr.tif'.format(t_name=current_t_series))
    raw_time_series = io.imread(movie_path)
except:
    movie_path = os.path.join(dataDir, 'motCorr.sima',
                              '{t_name}_motCorr.tif'.format(t_name=current_t_series))
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
    process_mov_core.get_stim_xml_params(dataDir, stimInputDir)
stimulus_information['input_data'] = stimInputData
stimulus_information['stim_name'] = stimType.split('\\')[-1]
stimulus_information['trial_coordinates'] = trialCoor
if isRandom:
    stimulus_information['epoch_adjuster'] = 1
else:
    stimulus_information['epoch_adjuster'] = 0

imaging_information = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                         'depth' : depth}
experiment_conditions = {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex,
                         'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}

# Generate an average dataset for epoch visualization if desired
(wholeTraces_allTrials_video, respTraces_allTrials, 
     baselineTraces_allTrials) = \
        process_mov_core.separate_trials_video(time_series, trialCoor,
                                               stimulus_information,
                                               imaging_information['frame_rate'])
process_mov_core.generate_avg_movie(dataDir, stimulus_information,
                                     wholeTraces_allTrials_video)


#%% Define analysis parameters and run region selection

analysis_params = {'save_output_dir': saveOutputDir, 'deltaF_method': 'mean', 
                   'save_figs': True ,'analysis_type': analysis_type} 


if ((analysis_type == '2D_edges_find_rois_delay_profile_save') or \
    (analysis_type == '2D_edges_find_save') or \
    (analysis_type == 'stripes_ON_horRF_save_rois') or \
    (analysis_type == '8D_edges_find_rois_save')):
    
    analysis_params['roi_sel_type'] = 'sima_STICA'
    analysis_params['area_max_micron'] = 4
    analysis_params['area_min_micron'] = 1
    analysis_params['cluster_max_1d_size_micron'] = 4
    analysis_params['cluster_min_1d_size_micron'] = 1
    
    if (analysis_type == 'stripes_ON_horRF_save_rois'):
        analysis_params['snr_t'] = 0.75
        analysis_params['corr_t'] = 0.4
    else:
        analysis_params['snr_t'] = 0.75
        analysis_params['corr_t'] = 0.4
   
    if use_other_for_roiExtraction:
        print('\nUsing another time series for ROI extraction...')
        roiExt_dir = os.path.join(alignedDataDir, current_exp_ID, 
                                  roiExtraction_tseries)
        try:
            roi_ext_mov_path = \
                os.path.join(roiExt_dir, 
                             '{t_name}_motCorr.tif'.format(t_name=roiExtraction_tseries))
            raw_ts = io.imread(roi_ext_mov_path)
        except:
            roi_ext_mov_path = \
                os.path.join(roiExt_dir, 'motCorr.sima',
                             '{t_name}_motCorr.tif'.format(t_name=roiExtraction_tseries))
            raw_ts = io.imread(roi_ext_mov_path)
        roi_ext_t_ser = copy.deepcopy(raw_ts)
        mean_im = roi_ext_t_ser.mean(0)
        ## Get stimulus and xml information
        (stim_info_roi_ext, tc, fr, d, xs, _, _,
         st, stimOutFile, epochCount, stimName, layerPosition,
         stimInputFile, xmlFile, trialCount, ir, inp_data,
         rawStimData) = \
            process_mov_core.get_stim_xml_params(roiExt_dir, stimInputDir)
        stim_info_roi_ext['input_data'] = inp_data
        stim_info_roi_ext['stim_name'] = st.split('\\')[-1]
        stim_info_roi_ext['trial_coordinates'] = tc
        if ir:
            stim_info_roi_ext['epoch_adjuster'] = 1
        else:
            stim_info_roi_ext['epoch_adjuster'] = 0
        
        im_info_roi_ext = {'frame_rate' : fr, 'pixel_size': xs, 
                                 'depth' : d}

        extraction_series_name = roiExtraction_tseries
    else:
        roi_ext_t_ser = time_series
        tc = trialCoor
        stim_info_roi_ext = stimulus_information
        im_info_roi_ext = imaging_information
        roiExt_dir = dataDir
        mean_im = mean_image
        extraction_series_name = current_t_series
    
    analysis_params['extraction_series'] = extraction_series_name
    
    cat_masks, cat_names, roi_masks, all_rois_image = \
        process_mov_core.run_ROI_selection(analysis_params,roi_ext_t_ser,
                                               tc,stim_info_roi_ext,
                                               imaging_information=im_info_roi_ext,
                                               dataDir=roiExt_dir,
                                               image_to_select=mean_im,
                                               experiment_conditions=experiment_conditions,
                                               use_avg_data=use_avg_data_for_roi_extract)
    
    cat_bool = np.zeros(shape=np.shape(mean_image))
    for cat_name in cat_names:
        if cat_name.lower() == 'bg':
            continue
        elif cat_name.lower() =='otsu':
            continue
        cat_bool[cat_masks[cat_names.index(cat_name)]] = 1
        
    rois = ROI_mod.generate_ROI_instances(roi_masks, cat_masks, cat_names,
                                          mean_image, 
                                          experiment_info = experiment_conditions, 
                                          imaging_info =imaging_information)
    for roi in rois:
        roi.analysis_params= analysis_params
        
    threshold_dict = {'SNR': analysis_params['snr_t'],
                      'reliability': analysis_params['corr_t']}
        
elif ((analysis_type == 'stripes_ON_delay_profile') or \
      (analysis_type == 'stripes_OFF_delay_profile') or \
      (analysis_type == 'STF_1') or \
      (analysis_type == 'gratings_transfer_rois_save') or\
      (analysis_type == 'luminance_edges_OFF' ) or\
      (analysis_type == 'luminance_edges_ON' )):
    
    analysis_params['roi_sel_type'] = 'transfer_sima_STICA'
    analysis_params['transfer_data_name'] = transfer_data_name
    analysis_params['corr_t'] = 0.4
    
    rois, cat_masks, cat_names = process_mov_core.run_ROI_selection(analysis_params,time_series,
                                               trialCoor,stimulus_information,
                                               imaging_information=imaging_information,
                                               dataDir=dataDir,
                                               image_to_select=mean_image,
                                               experiment_conditions=experiment_conditions)
    analysis_params['snr_t'] = 0.3
    # Update analysis parameters of transferred ROIs
    for roi in rois:
        for param in analysis_params.keys():
            roi.analysis_params[param] = analysis_params[param]
            
    threshold_dict = {'SNR': analysis_params['snr_t'],
                      'reliability': analysis_params['corr_t']}
            
elif ((analysis_type == 'stripes_ON_vertRF_transfer') or \
      (analysis_type == 'stripes_ON_horRF_transfer') or \
      (analysis_type == 'stripes_OFF_vertRF_transfer') or \
      (analysis_type == 'stripes_OFF_horRF_transfer')):
    analysis_params['roi_sel_type'] = 'transfer'
    analysis_params['transfer_data_name'] = transfer_data_name    
    rois, cat_masks, cat_names = process_mov_core.run_ROI_selection(analysis_params,time_series,
                                               trialCoor,stimulus_information,
                                               imaging_information=imaging_information,
                                               dataDir=dataDir,
                                               image_to_select=mean_image,
                                               experiment_conditions=experiment_conditions)
    # Update analysis parameters of transferred ROIs
    for roi in rois:
        for param in analysis_params.keys():
            roi.analysis_params[param] = analysis_params[param]
    threshold_dict = None
            
elif (analysis_type == '5sFFF_analyze_save'):
    
    analysis_params['roi_sel_type'] = 'manual'

    cat_masks, cat_names, roi_masks, \
        all_rois_image = \
            process_mov_core.run_ROI_selection(analysis_params,time_series,
                                               trialCoor,stimulus_information,
                                               imaging_information=imaging_information,
                                               dataDir=dataDir,
                                               image_to_select=mean_image,
                                               experiment_conditions=experiment_conditions)
    cat_bool = np.zeros(shape=np.shape(mean_image))
    for cat_name in cat_names:
        cat_bool[cat_masks[cat_names.index(cat_name)]] = 1
    rois = ROI_mod.generate_ROI_instances(roi_masks, cat_masks, cat_names,
                                          mean_image, 
                                          experiment_info = experiment_conditions, 
                                          imaging_info =imaging_information)
    for roi in rois:
        roi.analysis_params= analysis_params
        
    threshold_dict = None

# %% Part 3 Generating ROI_bg classes and processing ROI data
## Gener ate ROI classes
plt.style.use("default")
plt.style.use("seaborn-talk")

# BG subtraction
bg_mask = cat_masks[int(np.where(np.array(cat_names) == 'BG')[0])]
raw_time_series = \
    np.transpose(np.subtract(np.transpose(raw_time_series), 
                             raw_time_series[:,bg_mask].mean(axis=1)))
print('\n Background subtraction done...')
    
# ROI responses
(wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,
 baselineTraces_allTrials_ROIs,
 all_clusters_dF_whole_trace) = process_mov_core.separate_trials_ROI_v3(
    raw_time_series, trialCoor, rois, stimulus_information,
    frameRate, analysis_params['deltaF_method'],
    max_resp_trial_len='max')
     
map(lambda roi: roi.findMaxResponse_all_epochs(), rois)
map(lambda roi: roi.appendStimInfo(stimulus_information,
                                   raw_stim_info=rawStimData), rois)
map(lambda roi: roi.setSourceImage(mean_image), rois)

if (analysis_type == '2D_edges_find_rois_delay_profile_save') or\
    (analysis_type == '2D_edges_find_save'):
    map(lambda roi: roi.calculate_DSI_PD(method='PDND'), rois)
    map(lambda roi: roi.calculate_CSI(frameRate=frameRate), rois)
elif analysis_type == '8D_edges_find_rois_save':
    map(lambda roi: roi.calculate_DSI_PD(method='vector'), rois)
    map(lambda roi: roi.calculate_CSI(frameRate=frameRate), rois)
elif analysis_type == 'gratings_transfer_rois_save':
    map(lambda roi: roi.calculateTFtuning_BF(), rois)

# No dF/F for SNR
if ((analysis_type != '5sFFF_analyze_save')):
    (_, respTraces_SNR, baseTraces_SNR, _) = process_mov_core.separate_trials_ROI_v3(
        raw_time_series, trialCoor, rois, stimulus_information,
        frameRate, analysis_params['deltaF_method'],
        df_use=False)
    
    # SNR and reliability
    if not (stimulus_information['random']):
        epoch_to_exclude = stimulus_information['baseline_epoch']
    else:
        epoch_to_exclude = None
    
    [SNR_rois, corr_rois] = process_mov_core.calculate_SNR_Corr(baseTraces_SNR,
                                                                respTraces_SNR, 
                                                                rois,
                                                                epoch_to_exclude=epoch_to_exclude)


# Thresholding
thresholded_rois = ROI_mod.threshold_ROIs(rois, threshold_dict)




# Exclude and separate overlapping clusters
if analysis_params['roi_sel_type'] == 'sima_STICA':
    # Otsu mask for excluding background clusters
    otsu_mask = cat_masks[int(np.where(np.array(cat_names) == 'otsu')[0])]
    otsu_threshold_Value = filters.threshold_otsu(mean_image[otsu_mask])
    otsu_threshold_Value2 = filters.threshold_otsu(mean_image)
    
    otsu_thresholded_mask = mean_image > otsu_threshold_Value

    cluster_1d_max_size_pixel = analysis_params['cluster_max_1d_size_micron'] / x_size
    cluster_1d_min_size_pixel = analysis_params['cluster_min_1d_size_micron'] / x_size
    [separated_rois, separated_roi_image] = \
        process_mov_core.clusters_restrict_size_regions \
        (thresholded_rois,cat_bool,cluster_1d_max_size_pixel,
         cluster_1d_min_size_pixel,otsu_thresholded_mask)

    final_roi_image = separated_roi_image
    final_rois = separated_rois

else:
    final_rois = thresholded_rois
    final_roi_image = ROI_mod.get_masks_image(final_rois)
    
    

# Plotting ROIs and properties
process_mov_core.plot_roi_masks(final_roi_image,mean_image,len(final_rois),
                                current_movie_ID,save_fig=save_fig, 
                                save_dir=figure_save_dir,alpha=0.4)

if analysis_type == 'gratings_transfer_rois_save' or\
    (analysis_type == 'STF_1'):
    
    
    properties = ['PD', 'DSI', 'CS','BF']
    colormaps = ['hsv', 'viridis', 'PRGn', 'inferno']
    
    if (analysis_type == 'STF_1'):
        vminmax = [(0,360), (0, 1), (-1, 1), (0, 1)]
    else:
        vminmax = [(0,360), (0, 2), (-1, 1), (0, 1.5)]
        
    
    data_to_extract = ['DSI', 'CSI', 'SNR', 'reliability','BF']
    
elif ((analysis_type == 'luminance_edges_OFF' ) or\
      (analysis_type == 'luminance_edges_ON' )):
    
    properties = ['PD', 'SNR', 'slope','reliability']
    colormaps = ['hsv', 'viridis', 'PRGn', 'viridis']
    vminmax = [(0,360), (0, 3), (-1, 2), (0, 1)]
    data_to_extract = ['DSI', 'CSI', 'slope', 'reliability']

elif (analysis_type == '5sFFF_analyze_save'):
    
    max_d = ROI_mod.data_to_list(final_rois, ['max_response'])
    properties = ['corr_fff', 'max_response']
    colormaps = ['PRGn', 'viridis']
    vminmax = [(-1,1), (0, np.max(max_d['max_response']))]
    data_to_extract = ['corr_fff', 'max_response']
    
elif ((analysis_type == 'stripes_ON_vertRF_transfer') or \
      (analysis_type == 'stripes_ON_horRF_transfer') or \
      (analysis_type == 'stripes_OFF_vertRF_transfer') or \
      (analysis_type == 'stripes_OFF_horRF_transfer')):
    
    max_d = ROI_mod.data_to_list(final_rois, ['max_response'])
    max_snr = ROI_mod.data_to_list(final_rois, ['SNR'])
    properties = ['corr_fff', 'max_response' ,'SNR','reliability']
    colormaps = ['PRGn', 'viridis','inferno','inferno']
    vminmax = [(-1,1), (0, np.max(max_d['max_response'])),
               (0, np.max(max_snr['SNR'])),(0, 1)]
    data_to_extract = ['stripe_gauss_fwhm', 'max_response' ,'SNR','reliability']

elif ((analysis_type == 'stripes_ON_delay_profile') or \
      (analysis_type == 'stripes_OFF_delay_profile')):
    properties = ['PD', 'SNR', 'CS','reliability']
    colormaps = ['hsv', 'viridis', 'PRGn', 'viridis']
    vminmax = [(0,360), (0, 25), (-1, 1), (0, 1)]
    data_to_extract = ['DSI', 'CSI', 'resp_delay_deg', 'reliability']
    
else:
    properties = ['PD', 'SNR', 'CS','reliability']
    colormaps = ['hsv', 'viridis', 'PRGn', 'viridis']
    vminmax = [(0,360), (0, 3), (-1, 1), (0, 1)]
    data_to_extract = ['DSI', 'CSI', 'SNR', 'reliability']
    
if analysis_type !=  (analysis_type == '2D_edges_find_save'):
    final_rois = process_mov_core.run_analysis(analysis_type,final_rois,
                                               analysis_params,experiment_conditions,
                                               imaging_information,summary_save_dir,
                                               save_fig=True,
                                               fig_save_dir = figure_save_dir,
                                               exp_ID=('%s_%s' % (current_movie_ID, 
                                                                  analysis_params['roi_sel_type'])))
    
images = []
for prop in properties:
    images.append(ROI_mod.generate_colorMasks_properties(final_rois, prop))
    
process_mov_core.plot_roi_properties(images, properties, colormaps, mean_image, 
                                     vminmax,current_movie_ID, depth, 
                                     save_fig=True,save_dir=figure_save_dir,
                                                  figsize=(8, 6),alpha=0.5)

final_roi_data = ROI_mod.data_to_list(final_rois, data_to_extract)
rois_df = pd.DataFrame.from_dict(final_roi_data)

process_mov_core.plot_df_dataset(rois_df, 
                                 data_to_extract,
                                 exp_ID=('%s_%s' % (current_movie_ID, 
                                                    analysis_params['roi_sel_type'])),
                                 save_fig=True, save_dir=figure_save_dir)
# plt.close('all')
 
# %% PART 4: Save data
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
varDict = locals()
pckl_save_name = ('%s_%s' % (current_movie_ID, analysis_params['roi_sel_type']))
saveWorkspace(analysis_params['save_output_dir'],pckl_save_name, varDict, 
              varFile='varSave_cluster_v2.txt',
              extension='.pickle')

print('\n\n%s saved...\n\n' % pckl_save_name)

#%% Plot ROI summarries
if plot_roi_summ:
    if analysis_type == 'gratings_transfer_rois_save' :
        import random
        plt.close('all')
        data_to_extract = ['DSI', 'BF', 'SNR', 'reliability', 'uniq_id','CSI',
                               'PD', 'exp_ID', 'stim_name']
        
        
        roi_figure_save_dir = os.path.join(figure_save_dir, 'ROI_summaries')
        if not os.path.exists(roi_figure_save_dir):
            os.mkdir(roi_figure_save_dir)
        copy_rois = copy.deepcopy(final_rois)
        random.shuffle(copy_rois)
        roi_d = ROI_mod.data_to_list(copy_rois, data_to_extract)
        rois_df = pd.DataFrame.from_dict(roi_d)
        for n,roi in enumerate(copy_rois):
            if n>10:
                break
            fig = ROI_mod.make_ROI_tuning_summary(rois_df, roi,cmap='coolwarm')
            save_name = '%s_ROI_summary_%d' % (roi.analysis_params['roi_sel_type'], roi.uniq_id)
            os.chdir(roi_figure_save_dir)
            fig.savefig('%s.png' % save_name,bbox_inches='tight',
                               transparent=False,dpi=300)
                
            plt.close('all')
    elif (analysis_type == 'STF_1'):
        data_to_extract = ['reliability', 'uniq_id','CSI']
        
        roi_figure_save_dir = os.path.join(figure_save_dir, 'ROI_summaries')
        if not os.path.exists(roi_figure_save_dir):
            os.mkdir(roi_figure_save_dir)
        
        roi_d = ROI_mod.data_to_list(final_rois, data_to_extract)
        rois_df = pd.DataFrame.from_dict(roi_d)
        for n,roi in enumerate(final_rois):
            if n>40:
                break
            fig = ROI_mod.plot_stf_map(roi,rois_df)
            save_name = '%s_ROI_STF_%d' % (roi.analysis_params['roi_sel_type'], roi.uniq_id)
            os.chdir(roi_figure_save_dir)
            fig.savefig('%s.png' % save_name,bbox_inches='tight',
                               transparent=False,dpi=300)
                
            plt.close('all')
        
        
    
    
    
    
    
    
    
    
    
    
    
    