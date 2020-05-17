#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 18:19:37 2018

@author: burakgur / Pre-processing of 2 photon calcium imaging data
"""

import os
import time
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
from batch_analysis_functions import batchSignalPlotSave, batchMotionAlignment
from batch_analysis_functions import preProcessDataFolders, batchDataSave
from batch_analysis_functions import moveProcessedData
from core_functions import extractMetadata, autoSegmentROIs


# %% User dependent directories, modify them according to your folder structure
# Directory where all the other directories located
initialDirectory = '/Volumes/HD-SP1/Data_processing/Python_data'
# initialDirectory = '/Users/burakgur/Documents/data'
#

rawDataDir = os.path.join(initialDirectory, 'raw_data')
stimDir = os.path.join(initialDirectory, 'stimuli')
# Directory where the stimulus input files, which contain the information about
# the stimulus are located
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
# Directory where the processed data will be saved
alignedDataDir = os.path.join(initialDirectory, 'motion_corrected')
# Directory where the processed data will be saved
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data')
# Database directory
dataBaseDir = os.path.join(initialDirectory, 'database')
metaDataBaseFile = os.path.join(dataBaseDir, 'metaDataBase.txt')
# Directory where the processed data will be stored for further analysis
processedDataStoreDir = \
    os.path.join(dataBaseDir, 'processed_data')

# %% User defined variables
# use_aligned = True  # Define which datasets will be used for signal extraction
combined = False
# Define dataset type: 'Stimulus' or 'MarkPoints'
# # mode = 'Stimulus'
#%% Organizing the folders of images with stimuli for further processing
print('Organizing the folders of images with stimuli')
preProcessDataFolders(rawDataDir, stimDir)

# %% Batch Motion correction (line by line Hidden Markov)
print('Performing line by line batch motion alignment...\n')
line_align_path = os.path.join(alignedDataDir, 'line_alignment')
start1 = time.time()
batchMotionAlignment(rawDataDir, line_align_path, granularity='row',combined=combined)
end1 = time.time()
time_passed = end1 - start1
print('Batch motion alignment successfully done in %d minutes\n' % \
      round(time_passed / 60))

# %% Batch Motion correction (plane based Hidden Markov)
print('Performing plane batch motion alignment...\n')
plane_align_path = os.path.join(alignedDataDir, 'plane_alignment')
start1 = time.time()
batchMotionAlignment(rawDataDir, plane_align_path, granularity='plane',
                     combined=combined)
end1 = time.time()
time_passed = end1 - start1
print('Batch motion alignment successfully done in %d minutes\n' % \
      round(time_passed / 60))

# %% ROI Selection
# For ROI selection use roibuddy
# IMPORTANT NOTICE: For selecting ROIs you need the raw data files in the 
# raw data folder. Otherwise it will break.
imageToSegmentDir = "/Volumes/HD-SP1/Burak_data/Python_data/motion_corrected/chosen_alignment/190327_jv_fly1/TSeriestest-001/motCorr.sima"
autoSegmentROIs(imageToSegmentDir, strategy='STICA', channel=0)

# %% Cluster analysis
alignment_path = os.path.join(alignedDataDir, 'chosen_alignment')

# %% Making figures and saving the plots.
# Existing signals, if found, can be used
# IMPORTANT NOTICE: For extracting signals you need the raw data files in the 
# raw data folder. Otherwise it will break.
print('Extracting signals and making figures ...\n')
start1 = time.time()
batchSignalPlotSave(line_align_path, initialDirectory=initialDirectory,
                    use_aligned=use_aligned, mode='Stimulus')
end1 = time.time()
time_passed = end1 - start1
print('Figures are successfully created in %d minutes\n' % \
      round(time_passed / 60))

# %% Save data
# always extract signals again in case if ROI selection has changed, also extracts
# the ROI identities etc. so it's required to extract signals again.
print('Data processing and saving ...\n')

os.chdir(functionPath)
start1 = time.time()
batchDataSave(stimInputDir, saveOutputDir, line_align_path, use_aligned=use_aligned,
              intRate=10)
end1 = time.time()
time_passed = end1 - start1
print('All data saved in %d minutes\n' % \
      round(time_passed / 60))

# %% Extract to database and copy the data to the storing folder
os.chdir(functionPath)
extractMetadata(varDict={}, batch=True, batchRegEx='*.pickle', IdName='ID',
                batchDir=saveOutputDir, metaDb=metaDataBaseFile,
                getDate=False, termSep='_', metaVars='metadataVar.txt')
moveProcessedData(processedDataDir=saveOutputDir, dataBaseDir=dataBaseDir,
                  saveProcessedDataDir=processedDataStoreDir)
print('Meta data appended. Please modify the necessary variables')
