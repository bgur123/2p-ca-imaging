#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:16:23 2020

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

os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')

import ROI_mod
import post_analysis_core as pac

# %% Setting the directories
initialDirectory = '/Volumes/Backup Plus/PhD_Archive/Data_ongoing/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/selected')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '200210_T4T5_luminance','luminance_edges')
summary_save_dir = os.path.join(initialDirectory,
                                'results/200302_luminance/T4T5')

# Plotting parameters
colors, _ = pac.run_matplotlib_params()
color = colors[5]
color_pair = colors[2:4]
roi_plots=False
# %% Load datasets and desired variables
exp_t = '200210_T4T5_luminance_edges_OFF'
datasets_to_load = os.listdir(saveOutputDir)

                    
properties = ['BF', 'PD', 'SNR','Reliab','depth','DSI','slope']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
all_traces=[]
tunings = []
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
    
    # Reliability thresholding
    curr_rois = ROI_mod.analyze_luminance_edges(curr_rois)
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.4})
    
    if 'ON' in curr_rois[0].stim_name:
        curr_stim_type = 'OFF'
        continue
        
    
    all_rois.append(curr_rois)
    data_to_extract = ['SNR','reliability','slope','DSI']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    
    # There is just one ROI with 3Hz so discard that one
    
    tunings.append(np.squeeze\
                   (list(map(lambda roi: roi.edge_resps,curr_rois))))
    
    all_traces.append(np.mean(np.array\
                      (map(lambda roi : \
                           np.roll(roi.edge_resp_traces_interpolated,
                                   roi.edge_resp_traces_interpolated.shape[1]/2\
                                       -int(np.argmax(roi.edge_resp_traces_interpolated,
                                                axis=1).mean()),axis=1),
                                   curr_rois)),axis=0))
    
    
    depths = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))

    df_c = {}
    df_c['depth'] = depths
    if "RF_map_norm" in curr_rois[0].__dict__.keys():
        df_c['RF_map_center'] = list(map(lambda roi : (roi.RF_map_norm>0.95).astype(int)
                                             , curr_rois))
        df_c['RF_map_bool'] = np.tile(True,len(curr_rois))
        screen = np.zeros(np.shape(curr_rois[0].RF_map))
        screen[np.isnan(curr_rois[0].RF_map)] = -0.1
        
        
        for roi in curr_rois:
            curr_map = (roi.RF_map_norm>0.95).astype(int)
    
            x1,x2 = ndimage.measurements.center_of_mass(curr_map)
            s1,s2 = ndimage.measurements.center_of_mass(np.ones(shape=screen.shape))
            roi.distance_to_center = np.sqrt(np.square(x1-s1) + np.square(x2-s2))
        df_c['RF_distance_to_center'] = list(map(lambda roi : roi.distance_to_center, 
                                             curr_rois))
        print('RFs found')
    else:
        df_c['RF_map_center'] = np.tile(None,len(curr_rois))
        df_c['RF_map_bool'] = np.tile(False,len(curr_rois))
        df_c['RF_distance_to_center'] = np.tile(np.nan,len(curr_rois))
            
    df_c['slope'] = roi_data['slope']
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

ROI_num = np.sum(map(lambda rois : len(rois),all_rois))

resp_len = np.array(map(lambda trace : trace.shape[1], all_traces)).min()
resps = np.full((flyNum,
                 all_traces[0].shape[0], resp_len),np.nan)
all_rois=np.concatenate(all_rois)
for idx,traces in enumerate(all_traces):
    resps[idx,:,:] = traces[:,:resp_len]
# %%  Slope
fig = plt.figure(figsize=(16, 3))
grid = plt.GridSpec(1, 3, wspace=0.3, hspace=1)

ax1=plt.subplot(grid[0,0])
ax2=plt.subplot(grid[0,1])
ax3 = plt.subplot(grid[0,2])
diff_luminances = all_rois[0].luminances
cmap = matplotlib.cm.get_cmap('inferno')
norm = matplotlib.colors.Normalize(vmin=0, 
                                   vmax=np.max(diff_luminances))

sensitivities = np.concatenate(tunings)
properties = ['Luminance', 'Response']
senst_df = pd.DataFrame(columns=properties)
colors_lum = []
for idx, luminance in enumerate(diff_luminances):
    curr_color = cmap(norm(luminance))
    colors_lum.append(curr_color)
    mean = resps.mean(axis=0)[idx]
    x = np.linspace(0,len(mean),len(mean))
    error = resps.std(axis=0)[idx]/np.sqrt(flyNum)
    ub = mean + error
    lb = mean - error
    ax1.plot(x,mean,'-',lw=3,color=curr_color,alpha=.8,
                 label=luminance)
    ax1.fill_between(x, ub, lb,
                      color=np.array(curr_color[:3]), alpha=.2)
    
    curr_sensitivities=sensitivities[:,idx]
    curr_luminances = np.ones(curr_sensitivities.shape) * luminance
    df = pd.DataFrame.from_dict({'Luminance':curr_luminances,
                                 'Response':curr_sensitivities}) 
    rois_df = pd.DataFrame.from_dict(df)
    senst_df = senst_df.append(rois_df, ignore_index=True, sort=False)
ax1.set_ylabel('$\Delta F/F$')
ax1.set_title('Aligned mean responses n:%d(%d)' % (flyNum, ROI_num))   

tuning_curves = np.concatenate((tunings[:]))
all_mean_data = np.mean(tuning_curves, axis=0)
all_yerr = np.std(tuning_curves, axis=0)/np.sqrt(flyNum)
lums = curr_rois[0].luminances
ax2.errorbar(lums,all_mean_data, all_yerr,
             fmt='-s',alpha=.9,color=colors[3])
stim_frames = all_rois[-1].stim_info['output_data'][:,7]  # Frame information
stim_vals = all_rois[-1].stim_info['output_data'][:,3] # Stimulus value
uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
stim_vals = stim_vals[uniq_frame_id]

    
ax2.set_ylim((0,2.5))
ax2.set_ylabel('$\Delta F/F$')
if 'OFF' in all_rois[0].stim_name:
    ax1.legend(title='X -> 0',loc ='upper left')
    ax2.set_xlabel('Preceeding luminance')
    lum_stim = stim_vals.copy()
    lum_stim[stim_vals==0] = 0
    for iEpoch in range(1,len(np.unique(stim_vals))):
        
        curr_lum = lums[np.mod(iEpoch-1,len(lums))]
        lum_stim[stim_vals==iEpoch] = ((3*curr_lum)+(2*0))/5
    ax22 = ax2.twinx()
    ax22.plot(lums,((0-lums)/lum_stim.mean()),'-o',alpha=.9,color='k',label='Weber-allmean')
    # ax22.plot(lums,((0-lums)/0),'-o',alpha=.9,color='g',label='Weber-BGmean')
    ax22.plot(lums,((0-lums)/lums),'-o',alpha=.9,color='r',label='Weber-Corrected')
    ax22.legend()
    ax22.set_ylim((ax22.get_ylim()[0],0))
else:
    ax1.legend(title='0 -> X',loc ='upper left')
    ax2.set_xlabel('Following luminance')
    lum_stim = stim_vals.copy()
    lum_stim[stim_vals==0] = 1
    for iEpoch in range(1,len(np.unique(stim_vals))):
        
        curr_lum = lums[np.mod(iEpoch-1,len(lums))]
        lum_stim[stim_vals==iEpoch] = ((3*0)+(2*curr_lum))/5
    ax22 = ax2.twinx()
    ax22.plot(lums,((lums-0)/lum_stim.mean()),'-o',alpha=.9,color='k',
             label='Weber-allmean')
    ax22.plot(lums,((lums-0)/1),'-o',alpha=.9,color='g',label='Weber-BGmean')
    # ax2.plot(lums,(lums-0)/lums,'-o',alpha=.9,color='r',label='Weber-Corrected')
    ax22.legend()
    ax22.set_ylim((0,ax22.get_ylim()[1]))

    


sns.distplot(combined_df['slope'],ax=ax3,hist_kws={"alpha": .5,'cumulative': False},
                 kde=True,kde_kws={"alpha": .8,'cumulative': False},
                 color=(colors[3][0],colors[3][1],colors[3][2],1.0),
                 hist=True,bins=20)
ax3.set_title('Luminance sensitivity')   
ax3.set_xlabel('Slope')
if 1:
    # Saving figure
    save_name = '{exp_t}_Edge_summary_{st}'.format(exp_t=exp_t,
                                                   st=curr_stim_type)
    os.chdir(summary_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

#%%
b = all_rois[np.argsort(combined_df['slope'])].copy()
roi = b[-1]
plt.imshow(roi.source_image,cmap='Greys_r')
a = roi.mask.copy().astype(float)
a[a==False] = np.nan
a[a==True] = 100
plt.imshow(a,cmap='Reds_r')


#%%
sns.violinplot(y=combined_df['slope'],color=colors[3])
# sns.swarmplot(y=combined_df['slope'],color='.2',s=2)
#%%
plt.close('all')
sns.jointplot('RF_distance_to_center','slope',data=combined_df,
            scatter_kws={"s": 30, "alpha":.5},kind='reg')
plt.xlim((0,35))

ax = plt.gca()
x = np.array(combined_df[np.isfinite(combined_df['RF_distance_to_center'])]['RF_distance_to_center'])
y = np.array(combined_df[np.isfinite(combined_df['RF_distance_to_center'])]['slope'])
r, pval = stats.pearsonr(x, y)
ax.annotate("r = {:.2f},p = {:.3f}".format(r, pval),
            xy=(.1, .9), xycoords=ax.transAxes)

# sns.jointplot('SNR','slope',data=combined_df,
#               s=20, alpha=.6)

#%%
plt.close('all')
sns.jointplot('Reliab','slope',data=combined_df,
            scatter_kws={"s": 30, "alpha":.5},kind='reg')
# plt.xlim((0,35))

ax = plt.gca()
x = np.array(combined_df[np.isfinite(combined_df['RF_distance_to_center'])]['Reliab'])
y = np.array(combined_df[np.isfinite(combined_df['RF_distance_to_center'])]['slope'])
r, pval = stats.pearsonr(x, y)
ax.annotate("r = {:.2f},p = {:.3f}".format(r, pval),
            xy=(.1, .9), xycoords=ax.transAxes)
#%% RF analysis
plt.close('all')
fig = plt.figure(figsize=(15, 8))


neurons_with_RF = combined_df[combined_df['RF_map_bool']]


curr_maps = np.array(neurons_with_RF['RF_map_center'])

slopes = neurons_with_RF['slope']
num=len(curr_maps)
fig.suptitle("RF distributions {fly} flies, {n} ROIs".format(n=num,
                                                fly=neurons_with_RF['flyID'].unique().shape[0]))
plt.imshow(screen, cmap='binary', alpha=.5)
plt_map = ((slopes * curr_maps).sum().astype(float)  / curr_maps.sum().astype(float))
plt_map[plt_map==0] = np.nan
plt.imshow(plt_map,alpha=.9,cmap='PRGn')

cbar = plt.colorbar(shrink=.5)
cbar.set_clim((-2,2))
cbar.set_label('Avg slope')
# plt.xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
# plt.ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
# plt.axis('off')
# ax.set_title('{bf}Hz | {s} ROIs'.format(bf= bf,s=num))
    
# if 1:
#     # Saving figure
#     save_name = '{exp_t}_RF_of_diff_BF'.format(exp_t=exp_t)
#     os.chdir(summary_save_dir)
#     plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
# plt.close('all')
#%%


# plt.scatter(x2,x1)
# curr_map[curr_map==0] = np.nan
# plt.imshow(curr_map,alpha=.6)
# plt.imshow(screen, cmap='binary', alpha=.5)



# #%%

# ex_slope = slopes[((slopes>1) & (slopes<1.5))]
# roi = all_rois[764]
# plt.plot(roi.edge_resp_traces_interpolated.T)
            