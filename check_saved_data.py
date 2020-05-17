#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:23:59 2020

@author: burakgur
"""

import cPickle
import os

# %% Setting the directories
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/selected')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '200310_GluClalpha_NI_cut_experiments','WN')
# %% Load datasets and desired variables
datasets_to_load = ['200309bg_fly1-TSeries-002_manual.pickle']

for idataset, dataset in enumerate(datasets_to_load):
    load_path = os.path.join(saveOutputDir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']