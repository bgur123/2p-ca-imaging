#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:18:45 2019

@author: burakgur

Organizing the folders, motion alignment using SIMA.
"""

#%% Importing required packages 
import os
import time
functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
from batch_analysis_functions import batchMotionAlignment, preProcessDataFolders

#%% User dependent directories, modify them according to your folder structure

# Directory where all the other directories located
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'

# Directory containing the raw data and raw stimuli outputs
#initialDirectory = "/Users/burakgur/2p/Python_data"
rawDataDir = os.path.join(initialDirectory,'raw_data')
stimDir = os.path.join(initialDirectory,'stimuli')

# Directory where the motion corrected movies and datasets will be saved
alignedDataDir = os.path.join(initialDirectory,'motion_corrected')    

#%% Organizing the folders of images with stimuli for further processing
print('Organizing the folders of images with stimuli')
preProcessDataFolders(rawDataDir, stimDir)

#%% Batch Motion correction (line by line Hidden Markov)
print('Performing line by line batch motion alignment...\n')
line_align_path = os.path.join(alignedDataDir,'line_alignment')
start1 = time.time()
batchMotionAlignment(rawDataDir, line_align_path,granularity ='row')
end1 = time.time()
time_passed = end1-start1
print('Batch motion alignment successfully done in %d minutes\n' % \
      round(time_passed/60) )

#%% Batch Motion correction (plane based Hidden Markov)
print('Performing plane batch motion alignment...\n')
plane_align_path = os.path.join(alignedDataDir,'plane_alignment')
start1 = time.time()
batchMotionAlignment(rawDataDir, plane_align_path,granularity ='plane')
end1 = time.time()
time_passed = end1-start1
print('Batch motion alignment successfully done in %d minutes\n' % \
      round(time_passed/60) )

#%% Run roibuddy to select layers for cluster analysis