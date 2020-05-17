#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:20:44 2020

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
flyID = '191220bg_fly1'
t_nums = [5,6,9,10,11]
fly_files = [file_n for file_n in os.listdir(saveOutputDir) \
                              if flyID in file_n.lower()]

rois_start = True
images_str = 'T'
for data_file in fly_files:
    image_t_num = int(data_file.split('_')[-2].split('-')[-1])
    #Check if it is one of the desired tseries
    if not(image_t_num in t_nums):
        continue
    data_path = os.path.join(saveOutputDir,data_file)
    load_path = open(data_path, 'rb')
    workspace = cPickle.load(load_path)
    rois = workspace['final_rois']
    images_str = images_str + ('_%d'%image_t_num)
    # Initialize a new set of ROIs
    if rois_start:
        rois_start = False
        properties = ['category','experiment_info','imaging_info',
                      'source_image']
        combined_rois = ROI_mod.transfer_masks(rois, properties)
        for roi in combined_rois:
            roi.experiment_info['MovieIDs'] =[]
            roi.experiment_info['MovieIDs'].append(rois[0].experiment_info['MovieID'])
            roi.stripe_Rsq_vals =[]
            roi.reliabilities =[]
            roi.SNRs = []
            del roi.experiment_info['MovieID']
        print('ROIs created.')
    
    # Transfer data according to the stimulus
    if 'LocalCircle' in rois[0].stim_info['stim_name']:
        
        for idx,roi in enumerate(combined_rois):
            if False in (rois[idx].mask == roi.mask):
                raise TypeError("Masks do not match for idx: {idx}".format(idx=idx))
            
            roi.experiment_info['MovieIDs'].append(rois[idx].experiment_info['MovieID'])
            
            transfer_props =['corr_fff','int_con_trace','corr_pval',
                             'int_stim_trace']
            roi = ROI_mod.transfer_props(roi,rois[idx],transfer_props)
        print('Flash data acquired...')
    elif 'stripe_ONcont1_vert' in rois[0].stim_info['stim_name']:
        for idx,roi in enumerate(combined_rois):
            if False in (rois[idx].mask == roi.mask):
                raise TypeError("Masks do not match for idx: {idx}".format(idx=idx))
            
            roi.experiment_info['MovieIDs'].append(rois[idx].experiment_info['MovieID'])
            
            transfer_props =['vert_RF_ON_trace','vert_RF_ON_gauss']
            roi = ROI_mod.transfer_props(roi,rois[idx],transfer_props)
            roi.SNRs.append(rois[idx].SNR)
            roi.reliabilities.append(rois[idx].reliability)
            roi.stripe_Rsq_vals.append(rois[idx].stripe_r_squared)
            roi.vert_RF_ON_fwhm = rois[idx].stripe_gauss_fwhm
            roi.vert_RF_ON_Rsq = rois[idx].stripe_r_squared
            
        print('On vertical RF data acquired...')
    elif 'stripe_ONcont1_hor' in rois[0].stim_info['stim_name']:
        for idx,roi in enumerate(combined_rois):
            if False in (rois[idx].mask == roi.mask):
                raise TypeError("Masks do not match for idx: {idx}".format(idx=idx))
            
            roi.experiment_info['MovieIDs'].append(rois[idx].experiment_info['MovieID'])
            
            transfer_props =['hor_RF_ON_trace','hor_RF_ON_gauss']
            roi = ROI_mod.transfer_props(roi,rois[idx],transfer_props)
            roi.SNRs.append(rois[idx].SNR)
            roi.reliabilities.append(rois[idx].reliability)
            roi.stripe_Rsq_vals.append(rois[idx].stripe_r_squared)
            roi.hor_RF_ON_fwhm = rois[idx].stripe_gauss_fwhm
            roi.hor_RF_ON_Rsq = rois[idx].stripe_r_squared
            
        print('ON horizontal RF data acquired...')
    elif 'stripe_OFFcont1_vert' in rois[0].stim_info['stim_name']:
        for idx,roi in enumerate(combined_rois):
            if False in (rois[idx].mask == roi.mask):
                raise TypeError("Masks do not match for idx: {idx}".format(idx=idx))
            
            roi.experiment_info['MovieIDs'].append(rois[idx].experiment_info['MovieID'])
            
            transfer_props =['vert_RF_OFF_trace','vert_RF_OFF_gauss']
            roi = ROI_mod.transfer_props(roi,rois[idx],transfer_props)
            roi.SNRs.append(rois[idx].SNR)
            roi.reliabilities.append(rois[idx].reliability)
            roi.stripe_Rsq_vals.append(rois[idx].stripe_r_squared)
            roi.vert_RF_OFF_fwhm = rois[idx].stripe_gauss_fwhm
            roi.vert_RF_OFF_Rsq = rois[idx].stripe_r_squared
            
        print('OFF vertical RF data acquired...')
    elif 'stripe_OFFcont1_hor' in rois[0].stim_info['stim_name']:
        for idx,roi in enumerate(combined_rois):
            if False in (rois[idx].mask == roi.mask):
                raise TypeError("Masks do not match for idx: {idx}".format(idx=idx))
            
            roi.experiment_info['MovieIDs'].append(rois[idx].experiment_info['MovieID'])
            
            transfer_props =['hor_RF_OFF_trace','hor_RF_OFF_gauss']
            roi = ROI_mod.transfer_props(roi,rois[idx],transfer_props)
            roi.SNRs.append(rois[idx].SNR)
            roi.reliabilities.append(rois[idx].reliability)
            roi.stripe_Rsq_vals.append(rois[idx].stripe_r_squared)
            roi.hor_RF_OFF_fwhm = rois[idx].stripe_gauss_fwhm
            roi.hor_RF_OFF_Rsq = rois[idx].stripe_r_squared
            
        print('OFF horizontal RF data acquired...')

#%%
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
varDict = {'final_rois':combined_rois}
pckl_save_name = ('%s_combined_data_%s' % (combined_rois[0].experiment_info['FlyID'],images_str))
saveWorkspace(outDir=saveOutputDir,baseName=pckl_save_name, varDict=varDict, 
              varFile='varSave_cluster_v2.txt',extension='.pickle')

print('%s saved...' % pckl_save_name)

print('\nNew ROIs with combined data are created.')
    
        
        
    
    
