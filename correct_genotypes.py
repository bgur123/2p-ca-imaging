#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:45:59 2020

@author: burakgur
"""

import cPickle
import os
import ROI_mod
from core_functions import saveWorkspace

#%% Dirs
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'

saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '191220_GluClflpTEV_NI_1')
#%% Combine the datasets and save as new
flyID = '191218bg_fly4'
fly_files = [file_n for file_n in os.listdir(saveOutputDir) \
                              if flyID in file_n.lower()]

for data_file in fly_files:
    #Check if it is one of the desired tseries

    data_path = os.path.join(saveOutputDir,data_file)
    load_path = open(data_path, 'rb')
    workspace = cPickle.load(load_path)
    rois = workspace['final_rois']
    
    for roi in rois:
        roi.experiment_info['Genotype'] = 'Pos_Mi1Rec__plus_GluClflpSTOPD'
   
    corrected_rois = rois
    os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
    varDict = {'final_rois':corrected_rois}

    saveWorkspace(outDir=saveOutputDir,baseName=data_file.split('.')[0], varDict=varDict, 
                  varFile='varSave_cluster_v2.txt',extension='.pickle')
    
    print(' %s saved...' % data_file.split('.')[0])
print('\n%s genotypes adjusted...\n'%flyID)


    
        
        
    
    
