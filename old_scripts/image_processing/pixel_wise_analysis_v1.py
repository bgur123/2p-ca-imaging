#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:40:02 2019

@author: burakgur
"""

import os
import time

from cluster_analysis_functions import batchPixelAnalysis_Plots
#%% Setting the directories
#from xmlUtilities import getFramePeriod, getLayerPosition, getMicRelativeTime
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,'motion_corrected/chosen_alignment')
stimInputDir = os.path.join(initialDirectory,'stimulus_types')

figureSaveDir = os.path.join(initialDirectory,'results')
#%%  Setting the parameters
pixel_analysis_params = {}
pixel_analysis_params['use_otsu'] = False
pixel_analysis_params['cropping'] = False 
pixel_analysis_params['use_smoothing'] = True
pixel_analysis_params['sigma'] = 0.75
pixel_analysis_params['save_figures'] = True
pixel_analysis_params['stim_input_directory'] = stimInputDir
pixel_analysis_params['save_fig_dir'] = figureSaveDir
pixel_analysis_params['save_example_movie'] = True
#%%
print('Performing pixel-wise analysis...\n')
start1 = time.time()

batchPixelAnalysis_Plots(alignedDataDir, pixel_analysis_params)

end1 = time.time()
time_passed = end1-start1
print('Pixel wise analysis successfully done in %d minutes\n' % \
      round(time_passed/60) )