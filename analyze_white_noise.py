#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:58:17 2020

@author: burakgur
"""
# %% Importing packages
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import h5py

os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
import ROI_mod
import white_noise_core as wnc
from core_functions import saveWorkspace
import process_mov_core as pmc

# %% Setting the directories


initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/200310_GluClalpha_NI_cut_experiments')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '200310_GluClalpha_NI_cut_experiments')
summary_save_dir = os.path.join(alignedDataDir,
                                '_summaries')

# %% Parameters to adjust
plt.close('all')

# Experimental parameters
current_exp_ID = '200316bg_fly4'
current_t_series ='TSeries-03162020-1023-002'
Genotype ='Mi1Rec_UAS-ICAM-TEVp__GluClflpTEVNI_Df-elavGal4'
Age = '5'
Sex = 'f'

# Analysis parameters
analysis_type = 'ternaryWN_elavation_RF'
# 'Dark_screen'
# 'ternaryWN_elavation_RF'

# ROI selection/extraction parameters
extraction_type = 'transfer' # 'SIMA-STICA' 'transfer' 'manual'
transfer_data_name = '200316bg_fly4-TSeries-03162020-1023-001_manual.pickle'



#%% Get the stimulus and imaging information
dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)

(time_series, stimulus_information,imaging_information) = \
    pmc.pre_processing_movie (dataDir,stimInputDir)
mean_image = time_series.mean(0)
current_movie_ID = current_exp_ID + '-' + current_t_series
figure_save_dir = os.path.join(dataDir, 'Results')
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)

experiment_conditions = \
    {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex,
     'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}
    

#%% Define analysis/extraction parameters and run region selection
#   generate ROI objects.

# Organizing extraction parameters
   
extraction_params = \
    pmc.organize_extraction_params(extraction_type,
                               current_t_series=current_t_series,
                               current_exp_ID=current_exp_ID,
                               alignedDataDir=alignedDataDir,
                               stimInputDir=stimInputDir,
                               use_other_series_roiExtraction = False,
                               use_avg_data_for_roi_extract = False,
                               roiExtraction_tseries=0,
                               transfer_data_n = transfer_data_name,
                               transfer_data_store_dir = saveOutputDir,
                               transfer_type = analysis_type,
                               imaging_information=imaging_information,
                               experiment_conditions=experiment_conditions)
        
    
analysis_params = {'deltaF_method': 'gray',
                   'analysis_type': analysis_type} 


# Select/extract ROIs
(cat_masks, cat_names, roi_masks, all_rois_image, rois,
threshold_dict) = \
    pmc.run_ROI_selection(extraction_params,image_to_select=mean_image)

# A mask needed in SIMA STICA to exclude ROIs based on regions
cat_bool = np.zeros(shape=np.shape(mean_image))
for idx, cat_name in enumerate(cat_names):
    if cat_name.lower() == 'bg':
        bg_mask = cat_masks[idx]
        continue
    elif cat_name.lower() =='otsu':
        otsu_mask = cat_masks[idx]
        continue
    cat_bool[cat_masks[cat_names.index(cat_name)]] = 1
    
# Generate ROI_bg instances
if rois == None:
    del rois
    rois = ROI_mod.generate_ROI_instances(roi_masks, cat_masks, cat_names,
                                          mean_image, 
                                          experiment_info = experiment_conditions, 
                                          imaging_info =imaging_information)
    
# We can store the parameters inside the objects for further use
for roi in rois:
    roi.extraction_params = extraction_params
    if extraction_type == 'transfer': # Update transferred ROIs
        roi.experiment_info = experiment_conditions
        roi.imaging_info = imaging_information
        for param in analysis_params.keys():
            roi.analysis_params[param] = analysis_params[param]
    else:
        roi.analysis_params= analysis_params


# %% 
# BG subtraction
time_series = np.transpose(np.subtract(np.transpose(time_series),
                                       time_series[:,bg_mask].mean(axis=1)))
print('\n Background subtraction done...')
# Stimulus 
stimpath = os.path.join(stimInputDir,
                         'StimulusData_Discrete_1_12_100000_Seed_735723.mat' )
stim = h5py.File(stimpath)
#stim file has group names 'stimulus' and 'stimulusMetadata'
#to reach group keys run list(stim.keys())
stim = stim['stimulus'][()]  ##stim is np array with frames
# ROI raw signals
for iROI, roi in enumerate(rois):
    roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
    roi.wn_stim = stim

# Append relevant information and calculate some parameters
map(lambda roi: roi.appendStimInfo(stimulus_information), rois)
map(lambda roi: roi.setSourceImage(mean_image), rois)


#%% White noise analysis

rois = ROI_mod.reverse_correlation_analysis(rois)
final_rois = rois
#%% Save

os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
varDict = locals()
pckl_save_name = ('%s_%s' % (current_movie_ID, extraction_params['type']))
saveWorkspace(saveOutputDir,pckl_save_name, varDict, 
              varFile='data_save_vars.txt',extension='.pickle')

print('\n\n%s saved...\n\n' % pckl_save_name)
#%% Plotting the STAs
roi_im = ROI_mod.get_masks_image(rois)
    
# Plotting ROIs and properties
pmc.plot_roi_masks(roi_im,mean_image,len(rois),
                   current_movie_ID,save_fig=True,
                   save_dir=figure_save_dir,alpha=0.4)
fig1= ROI_mod.plot_STRFs(rois, f_w=None,number=None,cmap='coolwarm')
fig1.suptitle(experiment_conditions['Genotype'])
f1_n = 'STRFs_%s' % (current_exp_ID)
os.chdir(figure_save_dir)
fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
            transparent=False,dpi=300)
os.chdir(summary_save_dir)
f1_n = 'Summary_%s' % (current_movie_ID)
fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
            transparent=False,dpi=300)
    
    
    
    
    
    
    
    
    