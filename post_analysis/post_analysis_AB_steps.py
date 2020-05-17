#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:16:04 2020

@author: burakgur
"""

import cPickle
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import os
import random
from scipy.stats import linregress
from scipy.stats.stats import pearsonr
import seaborn as sns
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')

import ROI_mod
import post_analysis_core as pac
#%%
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
results_save_dir = os.path.join(initialDirectory,
                                'results/L2_lumCon')


# %% Load datasets and desired variables
exp_folder = 'L2_lumCon/A_B'
exp_t = '200513_L2AB_step'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)


# Initialize variables
properties = ['Reliab','category']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
all_traces=[]
a_step_responses = []
b_step_responses = []
a_step_baseline_responses = []
b_step_baseline_responses = []
a_to_b_step_responses = []
a_step_base_sub = []
b_step_base_sub = []
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
    if (not('A_B_step' in curr_rois[0].stim_name) or \
        (not('7sBG' in curr_rois[0].stim_name))):
        continue
    curr_rois=ROI_mod.sep_trial_compute_df(curr_rois,df_method='Baseline_epoch',
                                        df_base_dur=1,filtering=True)
    curr_rois = ROI_mod.analyze_A_B_step(curr_rois,int_rate=10)
    
    
    # Thresholding might be needed for good ROIs
#    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.4})
    
    
    # Luminance slopes
    curr_rois = ROI_mod.calculate_correlation(curr_rois, stim_type='AB_steps')
    geno = curr_rois[0].experiment_info['Genotype'][:3]
    if (geno =='Mi1') or (geno =='Tm3') or (geno =='Mi4'):
        corr_t = ('b',0)
    elif (geno =='L3_') or (geno =='L1_') or (geno =='Tm9')or (geno =='L2_'):
        corr_t = ('s',0)
    else:
        raise TypeError('Correlation threshold can not be determined for this genotype.')
    # Thresholding might be needed for good ROIs
    
#    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'correlation':corr_t,
#                                                   'reliability' : 0.1,
#                                                   'AB_resp_max' : 0.4,
#                                                   'mean_base_resp_diff' : 0})
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'correlation':corr_t,
                                                   'reliability' : 0.3,
                                                   'AB_resp_max' : 0.2})
    if len(curr_rois) <= 1:
        warnings.warn('{d} contains 1 or no ROIs, skipping'.format(d=dataset))
        continue
    
    
    all_rois.append(curr_rois)
    
    
    curr_all_traces = \
        np.array(map(lambda roi : roi.resp_traces_interpolated[:,:200],
                         curr_rois))
        
    all_traces.append(curr_all_traces)
    
    a_step_responses.append(np.array(map(lambda roi : roi.a_step_responses[1:],
                         curr_rois)))
    b_step_responses.append(np.array(map(lambda roi : roi.b_step_responses[1:],
                         curr_rois)))
    a_step_baseline_responses.append(np.array(map(lambda roi : roi.a_step_baseline_responses[1:],
                         curr_rois)))
    b_step_baseline_responses.append(np.array(map(lambda roi : roi.b_step_baseline_responses[1:],
                         curr_rois)))
    
    a_to_b_step_responses.append(np.array(map(lambda roi : roi.a_to_b_step_responses[1:],
                         curr_rois)))
    
    a_step_base_sub.append(np.array(map(lambda roi : roi.a_step_responses[1:],
                                        curr_rois)) - \
                           np.array(map(lambda roi : roi.a_step_baseline_responses[1:],
                                        curr_rois)))
    
    b_step_base_sub.append(np.array(map(lambda roi : roi.b_step_responses[1:],
                                        curr_rois)) - \
                           np.array(map(lambda roi : roi.b_step_baseline_responses[1:],
                                        curr_rois)))
        
    data_to_extract = ['SNR','reliability','category']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    
    df_c = {}
    df_c['SNR'] = roi_data['SNR']
    df_c['category'] = roi_data['category']
    df_c['reliability'] = roi_data['reliability']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['Geno'] = np.tile(geno,len(curr_rois))
    df_c['flyNum'] = np.tile(flyNum,len(curr_rois))
    df_c['uniq_id'] = np.array(map(lambda roi : roi.uniq_id, curr_rois)) 
    flyNum = flyNum +1
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))
    
a_step_responses = np.concatenate((a_step_responses[:]))
b_step_responses = np.concatenate((b_step_responses[:]))
a_step_baseline_responses = np.concatenate((a_step_baseline_responses[:]))
b_step_baseline_responses = np.concatenate((b_step_baseline_responses[:]))
a_to_b_step_responses = np.concatenate((a_to_b_step_responses[:]))
a_step_base_sub = np.concatenate((a_step_base_sub[:]))
b_step_base_sub = np.concatenate((b_step_base_sub[:]))

all_rois = np.concatenate(all_rois)
all_traces = np.concatenate(all_traces)

#%%
_, colorss = pac.run_matplotlib_params()
colors = [colorss['magenta'],colorss['orange'],colorss['purple'],
          colorss['red'],colorss['yellow'],colorss['green1']]

color_dict = {'L1_' : colorss}

#%% Plotting reliability etc
# ax2 = plt.axes()
sns.pairplot(hue="Geno", data=combined_df[['Geno','SNR','reliability']], 
             palette=colors,plot_kws = {'alpha':0.8,'s':10})


#%%
ax2 = plt.axes()
sns.boxplot(x="Geno", y="SNR", data=combined_df, palette=colors,notch=True)

ax2.set_xlabel('Neuron')
ax2.set_ylabel('SNR')


if 1:
    # Saving figure
    save_name = '{exp_t}_SNR'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
# plt.close('all')
#%%


plt.close('all')

for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    neuron_save_dir = os.path.join(results_save_dir,geno,'AB')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    
    if plot_only_cat:
        try:
            cat_dict[geno]
            curr_neuron_mask = (combined_df['Geno']==geno) & \
                (combined_df['category']==cat_dict[geno])
        except KeyError:
            print('No pre-defined category for {g} found. Taking all...\n'.format(g=geno))
            curr_neuron_mask = (combined_df['Geno']==geno)
    else:
        curr_neuron_mask = (combined_df['Geno']==geno)
     
    flyIDs = np.array(combined_df[curr_neuron_mask]['flyID']).astype(str)
    
    plt.close('all')
    fig = plt.figure(figsize=(7, 7))
    grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.5)
    
    ax1=plt.subplot(grid[0,0])
    ax2=plt.subplot(grid[0,1])
    ax3=plt.subplot(grid[1,0])
    ax4=plt.subplot(grid[1,1])
    
    a_lums = all_rois[curr_neuron_mask][0].epoch_luminance_A_steps[1:]
    b_lums = all_rois[curr_neuron_mask][0].epoch_luminance_B_steps[1:]
    a_cont = all_rois[curr_neuron_mask][0].epoch_contrast_A_steps[1:]
    b_cont = all_rois[curr_neuron_mask][0].epoch_contrast_B_steps[1:]
    b_cont_BG_weber = all_rois[curr_neuron_mask][0].epoch_contrast_B_steps_BGweber[1:]
    
    diff_luminances = all_rois[curr_neuron_mask][0].epoch_luminance_B_steps[1:]
    cmap = matplotlib.cm.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmin=np.min(diff_luminances), 
                                       vmax=np.max(diff_luminances))
    plot_max = 0
    for idx_lum, luminance in enumerate(diff_luminances):
        
        curr_traces = all_traces[curr_neuron_mask,idx_lum,:]
        
        a=pac.compute_over_samples_groups(data = curr_traces, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

        label = '{g} n: {f}({ROI})'.format(g=geno,
                                           f=len(a['experiment_ids'][geno]['over_samples_means']),
                                           ROI=len(a['experiment_ids'][geno]['all_samples']))
        
        
        mean_r = a['experiment_ids'][geno]['over_groups_mean'][10:-10] 
        err = a['experiment_ids'][geno]['over_groups_error'][10:-10] 
        # mean_r -= mean_r[:50].mean()
        if plot_max < (mean_r).max():
            plot_max = (mean_r).max()
            
        ub = mean_r + err
        lb = mean_r - err
        x = np.linspace(0,len(mean_r),len(mean_r))
        ax1.plot(x,mean_r,'-',lw=2,color=cmap(norm(luminance)),alpha=.7,
                 label=luminance)
        # ax1.fill_between(x, ub, lb,color=cmap(norm(luminance)), alpha=.3)
        # ax1.legend()
        # ax.axis('off')
    ax1.set_ylim(ax1.get_ylim()[0],plot_max+0.1)
    ax1.set_xlabel('Time (decaseconds)')
    ax1.set_ylabel('$\Delta F/F$')
    
    a_step = a_step_responses[curr_neuron_mask,:]
    b_step = b_step_responses[curr_neuron_mask,:]
    a_base = a_step_baseline_responses[curr_neuron_mask,:]
    b_base = b_step_baseline_responses[curr_neuron_mask,:]
    a_to_b =  a_to_b_step_responses[curr_neuron_mask,:]
    a_step_base_subtracted =  a_step_base_sub[curr_neuron_mask,:]
    b_step_base_subtracted =  b_step_base_sub[curr_neuron_mask,:]
    
    astep=pac.compute_over_samples_groups(data = a_step, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))
    
    bstep=pac.compute_over_samples_groups(data = b_step, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))
    astep_mean = astep['experiment_ids'][geno]['over_groups_mean']
    astep_err = astep['experiment_ids'][geno]['over_groups_error']
    
    bstep_mean = bstep['experiment_ids'][geno]['over_groups_mean']
    bstep_err = bstep['experiment_ids'][geno]['over_groups_error']
    
    
    alpha = np.linspace(0.2,1,len(b_cont))
    if 'ON' in all_rois[curr_neuron_mask][0].stim_name:
        alpha=np.flip(alpha)
    a_colors = np.tile(colorss['magenta'],len(b_cont)).reshape(len(b_cont),3)
    b_colors = np.tile(colorss['green3'],len(b_cont)).reshape(len(b_cont),3)
    
    a_colors = np.c_[ a_colors, alpha ]
    b_colors = np.c_[ b_colors, alpha ]
    ax2.errorbar(a_lums,astep_mean, astep_err,linewidth=2,
                 fmt='--',color=colorss['magenta'],label='A step',alpha=.9)
    ax2.scatter(a_lums,astep_mean ,color=a_colors,s=40)
    ax2.errorbar(b_lums,bstep_mean, bstep_err,linewidth=2,
                 fmt='--',color=colorss['green3'],label='B step',alpha=.9)
    ax2.scatter(b_lums,bstep_mean ,color=b_colors,s=40)
    ax2.set_xlabel('Luminance')
    ax2.set_ylabel('$\Delta F/F$')
    ax2.set_ylim(-0.1,ax2.get_ylim()[1])
    ax2.legend()
    
    ax3.errorbar(a_cont,astep_mean, astep_err,linewidth=2,
                 fmt='--',color=colorss['magenta'],label='A step',alpha=.9)
    ax3.scatter(a_cont,astep_mean ,color=a_colors,s=40)
    ax3.errorbar(b_cont,bstep_mean, bstep_err,linewidth=2,
                 fmt='.',color=colorss['green3'],label='B step',alpha=.4)
    ax3.scatter(b_cont,bstep_mean ,color=b_colors,s=40)
    ax3.set_xlabel('Contrast')
    ax3.set_ylabel('$\Delta F/F$')
    ax3.set_ylim(-0.1,ax3.get_ylim()[1])
    ax3.legend()
    
    ax4.errorbar(a_cont,astep_mean, astep_err,linewidth=2,
                 fmt='--',color=colorss['magenta'],label='A step',alpha=.9)
    ax4.scatter(a_cont,astep_mean ,color=a_colors,s=40)
    ax4.errorbar(b_cont_BG_weber,bstep_mean, bstep_err,linewidth=2,
                 fmt='.',color=colorss['green3'],label='B step',alpha=.4)
    ax4.scatter(b_cont_BG_weber,bstep_mean ,color=b_colors,s=40)
    ax4.set_xlabel('Weber BG Contrast')
    ax4.set_ylabel('$\Delta F/F$')
    ax4.set_ylim(-0.1,ax3.get_ylim()[1])
    ax4.legend()
    
    
    
    

    title = 'Peak {g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    
    fig.suptitle(title)
    # Saving figure
    save_name = '_Summary7sBG_Peak_{g}_'.format(g=geno)
    os.chdir(neuron_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

