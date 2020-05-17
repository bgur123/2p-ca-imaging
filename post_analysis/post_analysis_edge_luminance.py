#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:38:55 2020

@author: burakgur
"""
#%%
import cPickle
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress, pearsonr
from sklearn import preprocessing
from scipy import ndimage

import seaborn as sns
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')

import ROI_mod
import post_analysis_core as pac
#%%
#initialDirectory = '/Volumes/Backup Plus/PhD_Archive/Data_ongoing/Python_data'
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
results_save_dir = os.path.join(initialDirectory,
                                'results/L2_lumCon')


# %% Load datasets and desired variables
exp_folder = 'L2_lumCon'
exp_t = '200513_L2lumedges'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)

                    
properties = ['SNR','Reliab','depth','slope']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
all_traces=[]
tunings = []
# Initialize variables
flyNum = 0

polarity_dict={'Mi1' : 'ON','Tm9' : 'OFF','Mi4' : 'ON',
         'L1_' : 'OFF','L2_' : 'OFF',
         'L3_': 'OFF',
         'Tm3' : 'ON'}

plot_only_cat = True
cat_dict={
         'L1_' : 'M1',
         'Tm3' : 'M9'}
for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        warnings.warn('Skipping non pickle file: {f}\n'.format(f=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    
    if (not('edge_20dps' in curr_rois[0].stim_name)):
        continue
    if 'ON' in curr_rois[0].stim_name:
        curr_stim_type ='ON'
    elif 'OFF' in curr_rois[0].stim_name:
        curr_stim_type ='OFF'
    # Thresholding
    curr_rois = ROI_mod.analyze_luminance_edges(curr_rois)
    curr_rois = ROI_mod.calculate_correlation(curr_rois, stim_type='LuminanceEdges')
    geno = curr_rois[0].experiment_info['Genotype'][:3]
    if (geno =='Mi1') or (geno =='Tm3') or (geno =='Mi4'):
        corr_t = ('b',0.3)
    elif (geno =='L3_') or (geno =='L1_') or (geno =='Tm9') or (geno =='L2_'):
        corr_t = ('s',-0.3)
    else:
        raise TypeError('Correlation threshold can not be determined for this genotype.')
    # Thresholding might be needed for good ROIs
    
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {
                                                   'reliability' : 0.1,
                                                   'max_response' : 0.3})
    if len(curr_rois) <= 1:
        warnings.warn('{d} contains 1 or no ROIs, skipping'.format(d=dataset))
        continue
    all_rois.append(curr_rois)
    data_to_extract = ['SNR','reliability','slope','category']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    
    # There is just one ROI with 3Hz so discard that one
    
    tunings.append(np.squeeze\
                   (list(map(lambda roi: roi.edge_resps,curr_rois))))
    
    roi_traces = np.zeros((len(curr_rois),len(curr_rois[0].luminances),
                           49))
    
        
    all_traces.append(np.array\
                      (map(lambda roi : roi.max_aligned_traces[:,:49],
                                   curr_rois)))
    # print(all_traces[0].shape)
    
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
    df_c['category'] = roi_data['category']
    df_c['reliability'] = roi_data['reliability']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['Geno'] = np.tile(geno,len(curr_rois))
    df_c['flyNum'] = np.tile(flyNum,len(curr_rois))
    df_c['stim_type'] = np.tile(curr_stim_type,len(curr_rois))
    df_c['uniq_id'] = np.array(map(lambda roi : roi.uniq_id, curr_rois)) 
    flyNum = flyNum +1
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))

all_rois=np.concatenate(all_rois)
all_traces = np.concatenate(all_traces)
tunings = np.concatenate(tunings)
#%%
_, colors = pac.run_matplotlib_params()

c_dict = {k:colors[k] for k in colors if k in combined_df['Geno'].unique()}

        
   #%% Plotting reliability etc
# ax2 = plt.axes()
sns.pairplot(hue="Geno", data=combined_df[['Geno','SNR','reliability']], 
             palette=colors,plot_kws = {'alpha':0.8,'s':10},
             hue_order=c_dict.keys())


#%%
ax2 = plt.axes()
sns.violinplot(x="Geno", y="slope", data=combined_df, palette=colors)

ax2.set_xlabel('Neuron')
ax2.set_ylabel('Slope')
ax2.set_ylim((-3,4))


# %%  Slope
plt.close('all')

for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno]
    neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
    if plot_only_cat:
        try:
            cat_dict[geno]
            curr_neuron_mask = ((combined_df['Geno']==geno) & \
                (combined_df['category']==cat_dict[geno]) & \
                    (combined_df['stim_type']==polarity_dict[geno]))
        except KeyError:
            print('No pre-defined category for {g} found. Taking all...\n'.format(g=geno))
            curr_neuron_mask = (combined_df['Geno']==geno)
    else:
        curr_neuron_mask = ((combined_df['Geno']==geno) &\
                            (combined_df['stim_type']==polarity_dict[geno]))
            
    curr_df = combined_df[curr_neuron_mask]
    fig = plt.figure(figsize=(16, 3))
    grid = plt.GridSpec(1, 3, wspace=0.3, hspace=1)
    
    ax1=plt.subplot(grid[0,0])
    ax2=plt.subplot(grid[0,1])
    ax3 = plt.subplot(grid[0,2])
    diff_luminances = all_rois[curr_neuron_mask][0].luminances
    if 'ON' in all_rois[curr_neuron_mask][0].stim_name:
        curr_stim_type ='ON'
    elif 'OFF' in all_rois[curr_neuron_mask][0].stim_name:
        curr_stim_type ='OFF'
        
     
    cmap = matplotlib.cm.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmin=0, 
                                       vmax=np.max(diff_luminances))
    
    sensitivities = tunings[curr_neuron_mask]
    properties = ['Luminance', 'Response']
    senst_df = pd.DataFrame(columns=properties)
    colors_lum = []
    for idx_lum, luminance in enumerate(diff_luminances):
        curr_color = cmap(norm(luminance))
        colors_lum.append(curr_color)
        curr_traces = all_traces[curr_neuron_mask,idx_lum,:]
        
        a=pac.compute_over_samples_groups(data = curr_traces, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

        label = '{g} n: {f}({ROI})'.format(g=geno,
                                           f=len(a['experiment_ids'][geno]['over_samples_means']),
                                           ROI=len(a['experiment_ids'][geno]['all_samples']))
        
        
        mean_r = a['experiment_ids'][geno]['over_groups_mean'][:]
        err = a['experiment_ids'][geno]['over_groups_error'][:]
        
        ub = mean_r + err
        lb = mean_r - err
        x = np.linspace(0,len(mean_r),len(mean_r))
        ax1.plot(x,mean_r,'-',lw=2,color=cmap(norm(luminance)),alpha=.7,
                 label=luminance)
        ax1.fill_between(x, ub, lb,color=cmap(norm(luminance)), alpha=.3)
        ax1.legend()
        
        curr_sensitivities=sensitivities[:,idx]
        curr_luminances = np.ones(curr_sensitivities.shape) * luminance
        df = pd.DataFrame.from_dict({'Luminance':curr_luminances,
                                     'Response':curr_sensitivities}) 
        rois_df = pd.DataFrame.from_dict(df)
        senst_df = senst_df.append(rois_df, ignore_index=True, sort=False)
    ax1.set_ylabel('$\Delta F/F$')
    ax1.set_title('Aligned mean responses {l}'.format(l=label) )   
    
    
    tuning_curves = tunings[curr_neuron_mask]
    
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    all_yerr = a['experiment_ids'][geno]['over_groups_error']
    ax2.errorbar(diff_luminances,all_mean_data,all_yerr,
                 fmt='-s',alpha=1,color=geno_color,label=label)
    ax2.set_ylim((0,ax2.get_ylim()[1]))
    stim_frames = all_rois[curr_neuron_mask][-1].stim_info['output_data'][:,7]  # Frame information
    stim_vals =  all_rois[curr_neuron_mask][-1].stim_info['output_data'][:,3] # Stimulus value
    uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
    stim_vals = stim_vals[uniq_frame_id]

    # 
#    if curr_stim_type == 'OFF':
#        ax1.legend(title='X -> 0',loc ='upper left')
#        ax2.set_xlabel('Preceeding luminance')
#        lum_stim = stim_vals.copy()
#        lum_stim[stim_vals==0] = 0
#        for iEpoch in range(1,len(np.unique(stim_vals))):
#            
#            curr_lum = diff_luminances[np.mod(iEpoch-1,len(diff_luminances))]
#            lum_stim[stim_vals==iEpoch] = ((3*curr_lum)+(2*0))/5
#        ax22 = ax2.twinx()
#        ax22.plot(diff_luminances,((0-diff_luminances)/diff_luminances.mean()),
#                  '-o',alpha=.9,color='k',label='Weber-allmean')
#        # ax22.plot(lums,((0-lums)/0),'-o',alpha=.9,color='g',label='Weber-BGmean')
#        ax22.plot(diff_luminances,((0-diff_luminances)/diff_luminances),'-o',alpha=.9,
#                  color='r',label='Weber-Corrected')
#        ax22.legend()
#        ax22.set_ylim((ax22.get_ylim()[0],0))
#    elif curr_stim_type == 'ON':
#        ax1.legend(title='0 -> X',loc ='upper left')
#        ax2.set_xlabel('Following luminance')
#        lum_stim = stim_vals.copy()
#        lum_stim[stim_vals==0] = 1
#        for iEpoch in range(1,len(np.unique(stim_vals))):
#            
#            curr_lum = diff_luminances[np.mod(iEpoch-1,len(diff_luminances))]
#            lum_stim[stim_vals==iEpoch] = ((3*0)+(2*curr_lum))/5
#        ax22 = ax2.twinx()
#        ax22.plot(diff_luminances,((diff_luminances-0)/lum_stim.mean()),
#                  '-o',alpha=.9,color='k',
#                 label='Weber-allmean')
#        ax22.plot(diff_luminances,((diff_luminances-0)/1),'-o',alpha=.9,
#                  color='g',label='Weber-BGmean')
#        # ax2.plot(lums,(lums-0)/lums,'-o',alpha=.9,color='r',label='Weber-Corrected')
#        ax22.legend()
#        ax22.set_ylim((0,ax22.get_ylim()[1]))
#            
    sns.violinplot(y=curr_df['slope'],color=geno_color,
                   ax=ax3)

    
    ax3.set_title('Luminance sensitivity')   
    ax3.set_xlabel('Slope')
    # Saving figure
    save_name = '{exp_t}_Edge_summary_{st}_{geno}'.format(exp_t=exp_t,
                                                   st=curr_stim_type,
                                                   geno=geno)
    os.chdir(neuron_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    
    
    
    sns.jointplot('RF_distance_to_center','slope',data=curr_df,
            scatter_kws={"s": 30, "alpha":.5},kind='reg',
            color=(geno_color[0],geno_color[1],geno_color[2],1.0))
    plt.xlim((0,35))
    
    ax4 = plt.gca()
    
    x = np.array(curr_df[np.isfinite(curr_df['RF_distance_to_center'])]['RF_distance_to_center'])
    y = np.array(curr_df[np.isfinite(curr_df['RF_distance_to_center'])]['slope'])
    r, pval = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f},p = {:.3f}".format(r, pval),
                xy=(.1, .9), xycoords=ax.transAxes)
    save_name = '{exp_t}_slope-RF_{st}_{geno}'.format(exp_t=exp_t,
                                                   st=curr_stim_type,
                                                   geno=geno)
    os.chdir(neuron_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
        


#%% RF analysis
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    neuron_save_dir = os.path.join(results_save_dir,geno)
    curr_color = c_dict[geno]
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
        
    if plot_only_cat:
        try:
            cat_dict[geno]
            curr_mask = (combined_df['Geno']==geno) & \
                (combined_df['category']==cat_dict[geno])
        except KeyError:
            print('No pre-defined category for {g} found. Taking all...\n'.format(g=geno))
            curr_mask = (combined_df['Geno']==geno)
    
    plt.close('all')
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("RF distributions {s}".format(s=geno))
    
    curr_df = combined_df[curr_mask]
    neurons_with_RF = curr_df[curr_df['RF_map_bool']]
    
    curr_maps = np.array(neurons_with_RF['RF_map_center'])
    
    slopes = neurons_with_RF['slope']
    num=len(curr_maps)
    plt.imshow(screen, cmap='binary', alpha=.5)
    
    plt_map = ((slopes * curr_maps).sum().astype(float)  / curr_maps.sum().astype(float))
    plt_map[plt_map==0] = np.nan
    plt.imshow(plt_map,alpha=.9,cmap='PRGn')
    
    cbar = plt.colorbar(shrink=.5)
    cbar.set_clim((-1,1))
    cbar.set_label('Avg slope')
    plt.xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
    plt.ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
    plt.axis('off')
        
    save_name = '{exp_t}_slope-RFMAP_{st}_{geno}'.format(exp_t=exp_t,
                                                   st=curr_stim_type,
                                                   geno=geno)
    os.chdir(neuron_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
#%%
ex_slope = slopes[((slopes>1) & (slopes<1.5))]
roi = all_rois[764]
plt.plot(roi.edge_resp_traces_interpolated.T)
            