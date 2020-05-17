#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:22:39 2020

@author: burakgur
"""



import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import ROI_mod
from core_functions import saveWorkspace
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
from process_mov_core import get_stim_xml_params
# %% Setting the directories
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
source_data_dir = os.path.join(initialDirectory,
                              'selected_experiments/200213_luminance')
all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')

# %% Load datasets and desired variables
exp_folder = '200217_others_luminance/11steps'
exp_t = '200217_others_luminance_11steps'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)

# Initialize variables
for idataset, dataset in enumerate(datasets_to_load):
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    final_rois = workspace['final_rois']
    
    for roi in final_rois:
        try:
            roi.stim_info['output_data']
        except KeyError:
            current_exp_ID = roi.experiment_info['FlyID']
            current_t_series = '-'.join(roi.experiment_info['MovieID'].split('-')[1:])
            dataDir = os.path.join(source_data_dir, 
                                   re.sub(r'-/{0,1}',"/",roi.experiment_info['MovieID'],1))
            (stimulus_information, imaging_information) = \
                get_stim_xml_params(dataDir, stimInputDir)
            roi.stim_info['output_data'] = stimulus_information['output_data']
            
    os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
    varDict = locals()
    pckl_save_name = dataset.split('.')[0]
    saveWorkspace(outDir=data_dir,baseName=pckl_save_name, varDict=varDict, 
                  varFile='varSave_cluster_v2.txt',extension='.pickle')
    
    print('%s saved...' % pckl_save_name)
    # Rsq
    # data_to_extract = ['SNR', 'CSI','reliability','edge_r_squared']
    # final_roi_data = ROI_mod.data_to_list(final_rois, data_to_extract)
    # rois_df = pd.DataFrame.from_dict(final_roi_data)
    # combined_df = combined_df.append(curr_df, ignore_index=True, sort=False)

        
    
        
    # print('{ds} successfully loaded\n'.format(ds=dataset))
