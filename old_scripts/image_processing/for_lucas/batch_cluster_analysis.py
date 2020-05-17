#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:40:02 2019

@author: burakgur
"""

import os
import time
functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
from cluster_analysis_functions import batchClusterAnalysis_SaveData
from batch_analysis_functions import moveProcessedData
from core_functions import extractMetadata 
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
print('Performing cluster analysis...\n')
start1 = time.time()

batchClusterAnalysis_SaveData(alignedDataDir, cluster_analysis_params)

end1 = time.time()
time_passed = end1-start1
print('Cluster analysis successfully done in %d minutes\n' % \
      round(time_passed/60) )

#%%
os.chdir(functionPath)
extractMetadata(varDict={}, batch=True, batchRegEx='*.pickle', IdName='ID',
                    batchDir=saveOutputDir, metaDb=metaDataBaseFile, 
                    getDate=False, termSep='_',  metaVars='metadataVar.txt')
moveProcessedData(processedDataDir=saveOutputDir, dataBaseDir=dataBaseDir,
                  saveProcessedDataDir=processedDataStoreDir)
print('Meta data appended. Please modify the necessary variables')