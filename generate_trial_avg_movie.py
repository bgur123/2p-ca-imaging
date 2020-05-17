#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:32:58 2020

@author: burakgur
"""

# %% Importing packages
import os
import copy
import numpy as np
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
from skimage import io
import process_mov_core

# %% Setting the directories

initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/200207_T4T5_luminance')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
summary_save_dir = os.path.join(alignedDataDir,
                                '_summaries')

# %% Parameters to adjust
current_exp_ID = '200212bg_fly4'
current_t_series = 'TSeries-02122020-1538-004'


#%%

dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)
current_movie_ID = current_exp_ID + '-' + current_t_series

save_fig = True
try:
    movie_path = os.path.join(dataDir, '{t_name}_motCorr.tif'.format(t_name=current_t_series))
    raw_time_series = io.imread(movie_path)
except:
    movie_path = os.path.join(dataDir, 'motCorr.sima',
                              '{t_name}_motCorr.tif'.format(t_name=current_t_series))
    raw_time_series = io.imread(movie_path)
time_series = copy.deepcopy(raw_time_series)

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
# Generate an average dataset for epoch visualization 
(wholeTraces_allTrials_video, respTraces_allTrials, 
     baselineTraces_allTrials) = \
        process_mov_core.separate_trials_video(time_series, trialCoor,
                                               stimulus_information,
                                               imaging_information['frame_rate'])
process_mov_core.generate_avg_movie(dataDir, stimulus_information,
                                     wholeTraces_allTrials_video)