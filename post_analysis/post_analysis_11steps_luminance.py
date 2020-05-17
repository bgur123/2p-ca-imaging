#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:23:13 2020

@author: burakgur
"""
import cPickle
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
from sklearn import preprocessing

import seaborn as sns

import ROI_mod
import post_analysis_core as pac
#%%
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
results_save_dir = os.path.join(initialDirectory,
                                'results/200302_luminance')


# %% Load datasets and desired variables
exp_folder = '200217_others_luminance'
exp_t = '200217_others_luminance_11steps'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
# Initialize variables
properties = ['Reliab','category','slope']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
all_traces=[]
baseline_tunings = []

# Initialize variables
flyNum = 0

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
    if (not('11' in curr_rois[0].stim_name)):
        continue
    
    
    
    # Thresholding might be needed for good ROIs
    # curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.4})
    
    
    # Luminance slopes
    luminances= curr_rois[0].luminances
    curr_rois = ROI_mod.analyze_luminance_steps(curr_rois)
    curr_rois = ROI_mod.calculate_correlation(curr_rois, stim_type='11LuminanceSteps')
    geno = curr_rois[0].experiment_info['Genotype'][:3]
    if (geno =='Mi1') or (geno =='Tm3') or (geno =='Mi4'):
        corr_t = ('b',0)
    elif (geno =='L3_') or (geno =='L1_') or (geno =='Tm9'):
        corr_t = ('s',0)
    else:
        raise TypeError('Correlation threshold can not be determined for this genotype.')
    # Thresholding might be needed for good ROIs
    
    if geno == "L1_":
        rel_t = 0.1
    else:
        rel_t=0.4
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'correlation':corr_t,
                                                   'reliability' : rel_t,
                                                   'lum_resp_max' : 0.4})
    
    if len(curr_rois) <= 1:
        warnings.warn('{d} contains 1 or no ROIs, skipping'.format(d=dataset))
        continue
    
    
    base_resps = np.squeeze\
                    (list(map(lambda roi: \
                              np.mean((roi.lum_resp_traces_interpolated[:,65:70]-\
                                      roi.lum_resp_traces_interpolated.min())/(roi.lum_resp_traces_interpolated.max()-roi.lum_resp_traces_interpolated.min()),
                                      axis=1),curr_rois)))
                       
    # base_resps = np.squeeze\
    #                (list(map(lambda roi: \
    #                          np.mean(roi.lum_resp_traces_interpolated[:,65:70], 
    #                                   axis=1),curr_rois)))
    
    # Stimulus
    try:
        raw_stim = curr_rois[0].stim_info['output_data']
    except KeyError:
        warnings.warn('Raw stim not found skipping: {data}'.format(data=dataset))
        continue
    stim_frames = raw_stim[:,7]  # Frame information
    stim_vals = raw_stim[:,3] # Stimulus value
    uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
    stim_vals = stim_vals[uniq_frame_id]
    
    

    for idx, roi in enumerate(curr_rois):

        p2 = np.poly1d(np.polyfit(roi.luminances, base_resps[idx], 4))
        p1 = np.poly1d(np.polyfit(roi.luminances, base_resps[idx], 1))
        
        xp = np.linspace(0, 1, 100)
        
        non_linearity = np.sqrt(np.sum(np.square(p2(xp)) - np.square(p1(xp))))
        roi.NL_idx = non_linearity
        
        # plt.scatter(roi.luminances, base_resps[idx],label='Data')
        # plt.plot(xp, p1(xp),'--g',label='1st order')
        # plt.plot(xp, p2(xp),'--r', label='2nd order')
        # plt.legend()
        # plt.waitforbuttonpress()
        # plt.close('all')
        
        roi.slope = linregress(roi.luminances,base_resps[idx])[0]
        fps = roi.imaging_info['frame_rate']
        filtered = ROI_mod.low_pass(roi.df_trace.copy(), fps, crit_freq=1,plot=False)
        y = filtered.copy()
        
        x = stim_vals[:len(y)]
        n_bins = 20
        mi = pac.calc_MI(x, y, n_bins)
        roi.MI = mi
    
    
    baseline_tunings.append(base_resps)
    all_rois.append(curr_rois)
    data_to_extract = ['SNR','reliability','slope','MI','category','NL_idx']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    
    df_c = {}
    df_c['MI'] = roi_data['MI']
    df_c['NL_idx'] = roi_data['NL_idx']
    df_c['slope'] = roi_data['slope']
    df_c['category'] = roi_data['category']
    df_c['Reliab'] = roi_data['reliability']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['Geno'] = np.tile(geno,len(curr_rois))
    df_c['flyNum'] = np.tile(flyNum,len(curr_rois))
    df_c['uniq_id'] = np.array(map(lambda roi : roi.uniq_id, curr_rois)) 
    
    if "RF_map_norm" in curr_rois[0].__dict__.keys():
        df_c['RF_map_center'] = list(map(lambda roi : (roi.RF_map_norm>0.95).astype(int)
                                             , curr_rois))
        df_c['RF_map_bool'] = np.tile(True,len(curr_rois))
        screen = np.zeros(np.shape(curr_rois[0].RF_map))
        screen[np.isnan(curr_rois[0].RF_map)] = -0.1
        print('RFs found')
    else:
        df_c['RF_map_center'] = np.tile(None,len(curr_rois))
        df_c['RF_map_bool'] = np.tile(False,len(curr_rois))
        
    flyNum = flyNum +1
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))
baseline_tunings = np.concatenate((baseline_tunings[:]))
all_rois = np.concatenate(all_rois)
#%%
_, colors = pac.run_matplotlib_params()

c_dict = {k:colors[k] for k in colors if k in combined_df['Geno'].unique()}
#%%
ax2 = plt.axes()
sns.violinplot(x="Geno", y="Reliab", data=combined_df, palette=c_dict.values(), 
               inner="quartile", ax=ax2,scale='width',hue_order=c_dict.keys())

ax2.set_xlabel('Neuron')
ax2.set_ylabel('Reliability')


if 1:
    # Saving figure
    save_name = '{exp_t}_Reliability'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
# plt.close('all')
    
#%%
ax2 = plt.axes()
sns.violinplot(x="Geno", y="MI", data=combined_df, palette=c_dict.values(),
               inner="quartile", ax=ax2,scale='width',hue_order=c_dict.keys())

ax2.set_xlabel('Neuron')
ax2.set_ylabel('MI')
# ax2.set_ylim((0,1.75))

if 1:
    # Saving figure
    save_name = '{exp_t}_MI'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

#%%
fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
sns.violinplot(x="Geno", y="Reliab", data=combined_df, 
               inner="quartile", ax=ax2[0])

ax2[0].set_xlabel('Neuron')
ax2[0].set_ylabel('Mutual information (bits)')

sns.violinplot(x="Geno", y="slope", data=combined_df, scale="count", 
               inner="quartile", ax=ax2[1])

ax2[1].set_xlabel('Neuron')
ax2[1].set_ylabel('Luminance sensitivity')

if 1:
    # Saving figure
    save_name = '{exp_t}_MIandSlope'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
plt.close('all')

#%% Neuron by neuron analysis
plt.close('all')
fig = plt.figure(figsize=(10, 6))
grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.5)

ax1=plt.subplot(grid[0,0])
ax2=plt.subplot(grid[0,1])
ax3=plt.subplot(grid[1,0])
ax4=plt.subplot(grid[1,1])
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    
    if plot_only_cat:
        try:
            cat_dict[geno]
            curr_mask = (combined_df['Geno']==geno) & \
                (combined_df['category']==cat_dict[geno])
        except KeyError:
            print('No pre-defined category for {g} found. Taking all...\n'.format(g=geno))
            curr_mask = (combined_df['Geno']==geno)
    
    
    tuning_curves = baseline_tunings[curr_mask]
    
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_mask]['flyNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    all_yerr = a['experiment_ids'][geno]['over_groups_error']
    ax1.errorbar(luminances,all_mean_data-np.min(all_mean_data),all_yerr,
                 fmt='-o',alpha=0.8,color=c_dict[geno],label=label,
                 ms=5,lw=2,ls='--')
    

sns.violinplot(x="Geno", y="slope", data=combined_df, palette=c_dict.values(),
               ax=ax2,scale='width',hue_order=c_dict.keys())

sns.violinplot(x="Geno", y="MI", data=combined_df, palette=c_dict.values(),
               ax=ax3,scale='width',hue_order=c_dict.keys())

sns.violinplot(x="Geno", y="NL_idx", data=combined_df, palette=c_dict.values(),
               ax=ax4,scale='width',hue_order=c_dict.keys())

    

ax1.set_title('Baseline vs luminance')
ax1.set_ylabel('$\Delta F/F$')
ax1.set_xlabel('Luminances')
ax1.set_ylim((-0.1,1))
ax1.legend()

ax2.set_title('Luminance sensitivity')
ax2.set_xlabel('slope')

ax3.set_title('Mutual information')
ax3.set_xlabel('MI (bits)')

ax4.set_title('Non linearity')
ax4.set_xlabel('NL')

# Saving figure
save_name = '{exp_t}_Summary'.format(exp_t=exp_t)
os.chdir(results_save_dir)
plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

# #%% Check all raw traces
# plt.close('all')

# for idx, geno in enumerate(np.unique(combined_df['Geno'])):
#     neuron_save_dir = os.path.join(results_save_dir,geno,"11steps")
#     curr_color = c_dict[geno]
#     if not os.path.exists(neuron_save_dir):
#         os.mkdir(neuron_save_dir)
    
    
#     curr_neuron_mask = combined_df['Geno']==geno
#     flyIDs = np.array(combined_df[curr_neuron_mask]['flyID']).astype(str)
    
#     plt.close('all')
#     fig = plt.figure(figsize=(16, 6))
#     grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.5)
    
#     ax1=plt.subplot(grid[0,0])
#     ax2=plt.subplot(grid[0,1])
#     ax3=plt.subplot(grid[1,0])
#     ax4=plt.subplot(grid[1,1])
    
#     for category in combined_df[curr_neuron_mask]['category'].unique():
#         try :
#             if np.isnan(category):
#                 continue
#         except TypeError:
#             category_is_nan = False
            
#         curr_cat_mask = (combined_df['category']==category) & (curr_neuron_mask)
#         tuning_curves = baseline_tunings[curr_cat_mask]
        
#         a=pac.compute_over_samples_groups(data = tuning_curves, 
#                                     group_ids= np.array(combined_df[curr_cat_mask]['flyNum']), 
#                                     error ='SEM',
#                                     experiment_ids = np.array(combined_df[curr_cat_mask]['category']))
    
#         label = '{g} n: {f}({ROI})'.format(g=category,
#                                            f=len(a['experiment_ids'][category]['over_samples_means']),
#                                            ROI=len(a['experiment_ids'][category]['all_samples']))
#         all_mean_data = a['experiment_ids'][category]['over_groups_mean']
#         all_yerr = a['experiment_ids'][category]['over_groups_error']
#         ax1.errorbar(luminances,all_mean_data- all_mean_data.min(), all_yerr,
#                      fmt='-s',alpha=.8,label=label)
    
#     sns.violinplot(x="Geno", y="slope", hue='category',
#                    data=combined_df[curr_neuron_mask],inner="quartile", ax=ax2)
    
#     sns.violinplot(x="Geno", y="MI", hue='category',
#                    data=combined_df[curr_neuron_mask],inner="quartile", ax=ax3)
    
#     sns.violinplot(x="Geno", y="Reliab", hue='category',
#                    data=combined_df[curr_neuron_mask],inner="quartile", ax=ax4)
    
    
    
#     ax1.legend()
#     ax1.set_title('Baseline vs luminance')
#     ax1.set_ylabel('$\Delta F/F$')
#     ax1.set_xlabel('Luminances')
#     ax1.set_ylim((0,1))
    
#     # Saving figure
#     save_name = '_Summary_{g}_norm_minmax_filtered'.format(g=geno)
#     os.chdir(neuron_save_dir)
#     plt.savefig('%s.pdf' % save_name, bbox_inches='tight')


#     for flyID in np.unique(flyIDs):
#         fly_save_dir = os.path.join(neuron_save_dir,flyID)
#         if not os.path.exists(fly_save_dir):
#             os.mkdir(fly_save_dir)
            
#         fly_mask = combined_df['flyID'] == flyID
#         indices = np.argwhere(fly_mask).astype(int)
#         fly_df = combined_df[fly_mask]
#         for idd in indices:
#             plt.close('all')
#             curr_idx = idd[0]
            
#             roi_id = combined_df.iloc[curr_idx]['uniq_id']
#             curr_roi_props = fly_df[fly_df['uniq_id']==roi_id]
            
#             fig = plt.figure(figsize=(10, 6))
#             grid = plt.GridSpec(2, 3, wspace=0.3, hspace=0.5)
#             ax1=plt.subplot(grid[0,:])
#             ax2=plt.subplot(grid[1,0])
#             ax3=plt.subplot(grid[1,1])
#             ax4=plt.subplot(grid[1,2])
        
#             roi = all_rois[curr_idx]
#             raw_stim = roi.stim_info['output_data']
#             stim_frames = raw_stim[:,7]  # Frame information
#             stim_vals = raw_stim[:,3] # Stimulus value
#             uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
#             stim_vals = stim_vals[uniq_frame_id]
#             for stim_val in np.unique(stim_vals):
#                 stim_vals[stim_vals==stim_val] = luminances[int(stim_val)]
#             stim_plot = \
#                 stim_vals[:len(roi.df_trace)] +np.around(roi.df_trace.max(),1)
            
#             time = \
#                 np.linspace(0,len(roi.df_trace[50:])/roi.imaging_info['frame_rate'],
#                             len(roi.df_trace[50:]))
                
#             ax1.plot(time,roi.df_trace[50:],color=curr_color,
#                       label=geno)
#             ax1.plot(time,stim_plot[50:],'k',
#                       label='stimulus')
#             ax1.legend()
#             ax1.set_ylabel('$\Delta F/F$')
#             ax1.set_xlabel('Time(s)')
#             ax1.plot([0 ,time.max()],[stim_plot.min() ,stim_plot.min()],'--r',lw=1)
            
            
#             # ax2
#             ax2.plot(luminances,baseline_tunings[curr_idx],'-s',
#                       color=curr_color)
#             ax2.set_title('Baseline vs luminance')
#             ax2.set_ylabel('$\Delta F/F$')
#             ax2.set_xlabel('Luminances')
#             ax2.set_ylim((0,1))
#             ax2.legend()
            
#             # ax3
#             sns.scatterplot(x='MI', y='Reliab',alpha=.8,color='grey',
#                     data =fly_df,legend=False,s=100,ax=ax3)
#             sns.scatterplot(x='MI', y='Reliab',color=curr_color,
#                     data =curr_roi_props,legend=False,s=100,ax=ax3)
    
#             ax3.set_xlim(0, fly_df['MI'].max()+0.3)
#             ax3.set_ylim(0, 1)
#             ax3.set_title('MI vs Reliability')
            
#             # ax4
#             sns.scatterplot(x='slope', y='Reliab',alpha=.8,color='grey',
#                     data =fly_df,legend=False,s=100,ax=ax4)
#             sns.scatterplot(x='slope', y='Reliab',color=curr_color,
#                     data =curr_roi_props,legend=False,s=100,ax=ax4)
    
#             # ax4.set_xlim(0, fly_df['slope'].max()+0.3)
#             ax4.set_ylim(0, 1)
#             ax4.set_title('Luminance sensitivity vs Reliability')
            
#               # Saving figure
#             save_name = 'ROI_{idd}'.format(idd=int(roi_id))
#             os.chdir(fly_save_dir)
#             plt.savefig('%s.png' % save_name, bbox_inches='tight')
            
            
            
        
        
        
        
# #
# #%% Luminance sensitivity
# for idx, geno in enumerate(np.unique(combined_df['Geno'])):
#     mask = combined_df[combined_df['Geno']==geno]['flyNum'].unique().astype(int)-1
#     tuning_curves = np.concatenate((baseline_tunings[mask]))
#     all_mean_data = np.mean(tuning_curves, axis=0)
#     all_yerr = np.std(tuning_curves, axis=0)
#     lums = luminances
    
#     ## Tuning curve
#     mean_t = all_mean_data
#     std_t = all_yerr
#     ub = mean_t + std_t
#     lb = mean_t - std_t
#     plt.errorbar(lums,all_mean_data, all_yerr,
#                  fmt='-s',label=geno,alpha=1)
# plt.legend()
# # plt.xscale('log',basex=2)
# if 1:
#     # Saving figure
#     save_name = '{exp_t}_LumSensitivity'.format(exp_t=exp_t)
#     os.chdir(results_save_dir)
#     plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    
# #%%
# #%% RF analysis

# for idx, geno in enumerate(np.unique(combined_df['Geno'])):
#     neuron_save_dir = os.path.join(results_save_dir,geno,"11steps")
#     curr_color = c_dict[geno]
#     if not os.path.exists(neuron_save_dir):
#         os.mkdir(neuron_save_dir)
        
#     if plot_only_cat:
#         try:
#             cat_dict[geno]
#             curr_mask = (combined_df['Geno']==geno) & \
#                 (combined_df['category']==cat_dict[geno])
#         except KeyError:
#             print('No pre-defined category for {g} found. Taking all...\n'.format(g=geno))
#             curr_mask = (combined_df['Geno']==geno)
    
#     plt.close('all')
#     fig = plt.figure(figsize=(15, 8))
#     fig.suptitle("RF distributions {s}".format(s=geno))
    
#     curr_df = combined_df[curr_mask]
#     neurons_with_RF = curr_df[curr_df['RF_map_bool']]
    
#     curr_maps = np.array(neurons_with_RF['RF_map_center'])
    
#     slopes = neurons_with_RF['slope']
#     num=len(curr_maps)
#     plt.imshow(screen, cmap='binary', alpha=.5)
    
#     plt_map = ((slopes * curr_maps).sum().astype(float)  / curr_maps.sum().astype(float))
#     plt_map[plt_map==0] = np.nan
#     plt.imshow(plt_map,alpha=.9,cmap='PRGn')
    
#     cbar = plt.colorbar(shrink=.5)
#     cbar.set_clim((-1,1))
#     cbar.set_label('Avg slope')
#     plt.xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
#     plt.ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
#     plt.axis('off')
        
#     # if 1:
#     #     # Saving figure
#     #     save_name = '{exp_t}_RF_of_diff_BF'.format(exp_t=exp_t)
#     #     os.chdir(neuron_save_dir)
#     #     plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
#     # plt.close('all')
#     plt.waitforbuttonpress()
    
    