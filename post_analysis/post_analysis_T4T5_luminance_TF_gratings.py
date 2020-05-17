#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:07:02 2020

@author: burakgur
"""


import cPickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.cluster.hierarchy as shc
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy import ndimage, stats

os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging/post_analysis')

import ROI_mod
import post_analysis_core as pac

# %% Setting the directories
initialDirectory = '/Volumes/Backup Plus/PhD_Archive/Data_ongoing/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/selected')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '200210_T4T5_luminance','luminance_gratings')
summary_save_dir = os.path.join(initialDirectory,
                                'results/200302_luminance/T4T5')

# Plotting parameters
colors, _ = pac.run_matplotlib_params()
color = colors[5]
color_pair = colors[2:4]
roi_plots=False
# %% Load datasets and desired variables
exp_t = '200210_T4T5_luminance_gratings_contrast1'
datasets_to_load = os.listdir(saveOutputDir)

                    
properties = ['BF', 'PD', 'SNR','Reliab','depth','DSI']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
tfl_maps = []
# Initialize variables
flyNum = 0
for dataset in datasets_to_load:
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(saveOutputDir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    
    if 'lightBG' in curr_rois[0].stim_name:
        curr_stim_type = 'dark background'
        continue
    # if 'darkBG' in curr_rois[0].stim_name:
    #     curr_stim_type = 'bright background'
    #     continue
    
    # Reliability thresholding
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.4})

    all_rois.append(curr_rois)
    tfl_maps.append((np.array(map(lambda roi: np.array(roi.tfl_map),curr_rois))))
    data_to_extract = ['SNR','reliability','slope','DSI','CS']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    depths = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))

    df_c = {}
    df_c['depth'] = depths
    df_c['CS'] = roi_data['CS']
    df_c['SNR'] = roi_data['SNR']
    df_c['Reliab'] = roi_data['reliability']
    df_c['DSI'] = roi_data['DSI']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['flyNum'] = np.tile(flyNum,len(curr_rois))
    flyNum = flyNum +1
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))


all_rois=np.concatenate(all_rois)
mean_tfls_fly = np.array(map(lambda tfl : tfl.mean(axis=0),tfl_maps))

ROI_num = len(all_rois)
flyNum = len(mean_tfls_fly)
#%% plot avg TFL maps
plt.close("all")
pac.run_matplotlib_params()


fig = plt.figure(figsize = (5,5))
plt.title('TFL map {st} - {f}({roi})'.format(st=curr_stim_type,
                                             f = len(mean_tfls_fly), 
                                             roi=ROI_num))
          
ax=sns.heatmap(mean_tfls_fly.mean(axis=0), cmap='coolwarm',center=0,
               xticklabels=np.array(all_rois[0].tfl_map.columns.levels[1]).astype(float),
               yticklabels=np.array(all_rois[0].tfl_map.index),
               cbar_kws={'label': '$\Delta F/F$'})
ax.invert_yaxis()
ax.set_xlabel('Luminance')
ax.set_ylabel('Hz')
# Saving figure
save_name = '{exp_t}_TFL_{st}'.format(exp_t=exp_t,
                                           st=curr_stim_type)
os.chdir(summary_save_dir)
plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

#%% Plot at 1 Hz
plt.close("all")
fig = plt.figure(figsize = (5,5))
plt.title('Tunings {st} - {f}({roi})'.format(st=curr_stim_type,
                                                  f = flyNum,
                                                  roi=ROI_num))

luminances = np.array(all_rois[0].tfl_map.columns.levels[1]).astype(float)
freqs = np.array(all_rois[0].tfl_map.index)
mean = mean_tfls_fly.mean(axis=0)[3,:]
error = mean_tfls_fly.std(axis=0)[3,:]/np.sqrt(flyNum)
ub = mean + error
lb = mean - error
plt.plot(luminances,mean,'o-',lw=3,alpha=.8,color=[0,0,0],label='1Hz tuning')
plt.fill_between(luminances, ub, lb,color=[0,0,0], alpha=.2)
plt.xlabel('Luminance')
plt.ylabel('dF/F')
plt.ylim((0, 1))
plt.legend()

save_name = '{exp_t}_1Hztuning_{st}'.format(exp_t=exp_t,
                                           st=curr_stim_type)
os.chdir(summary_save_dir)
plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)


#%% Plot all  freqs
plt.close("all")
fig = plt.figure(figsize = (5,5))
plt.title('Tunings {st} - {f}({roi})'.format(st=curr_stim_type,
                                                  f = flyNum,
                                                  roi=ROI_num))

luminances = np.array(all_rois[0].tfl_map.columns.levels[1]).astype(float)
freqs = np.array(all_rois[0].tfl_map.index)
for idx, freq in enumerate(freqs):
    mean = mean_tfls_fly.mean(axis=0)[idx,:]
    error = mean_tfls_fly.std(axis=0)[idx,:]/np.sqrt(flyNum)
    ub = mean + error
    lb = mean - error
    plt.plot(luminances,mean,'o-',lw=3,alpha=.8,label='{freq}'.format(freq=freq))
    plt.fill_between(luminances, ub, lb, alpha=.2)
    plt.xlabel('Luminance')
    plt.ylabel('dF/F')
    # plt.ylim((0, 1))
    plt.legend()

save_name = '{exp_t}_AllFreqsTuning_{st}'.format(exp_t=exp_t,
                                           st=curr_stim_type)
os.chdir(summary_save_dir)
plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)


#%% Plot ON OFF
plt.close("all")
colorss,_ = pac.run_matplotlib_params()
colors = [colorss[-1],colorss[-2]]

fig = plt.figure(figsize = (5,5))
plt.title('1 Hz tunings {st}'.format(st=curr_stim_type))
cs_s = ['OFF', 'ON']
conc_tfls = np.concatenate(tfl_maps)
for idx, cs in enumerate(cs_s):
    
    
    curr_mask = combined_df['CS'] == cs
    curr_roi_num = (combined_df['CS'] == cs).astype(int).sum()
    label = '{cs} neurons - {roi}rois'.format(cs=cs,roi=curr_roi_num)
    
    curr_tfls = conc_tfls[curr_mask]
    mean = curr_tfls.mean(axis=0)[3,:]
    error = curr_tfls.std(axis=0)[3,:]/np.sqrt(curr_roi_num)
    ub = mean + error
    lb = mean - error
    plt.plot(luminances,mean,'o-',lw=3,alpha=.8,color=colors[idx],label=label)
    plt.fill_between(luminances, ub, lb,color=colors[idx], alpha=.2)
    plt.xlabel('Luminance')
    plt.ylabel('dF/F')
    plt.ylim((0, 1))
    plt.legend()
    
save_name = '{exp_t}_1Hztuning_{st}_T45_separated'.format(exp_t=exp_t,
                                           st=curr_stim_type)
os.chdir(summary_save_dir)
plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)