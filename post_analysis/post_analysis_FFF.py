#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:33:56 2020

@author: burakgur
"""


import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import post_analysis_core as pac
# %% Setting the directories
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
results_save_dir = os.path.join(initialDirectory,
                                'results/200310_GluClalpha_NI_cut_experiments')


# %% Load datasets and desired variables
exp_folder = '200310_GluClalpha_NI_cut_experiments/5sFFF'
exp_t = '200310_GluClalpha_NI_cut_experiments_FFF'
genotype_labels = ['Exp', 'TEVsite ctrl', 'ICAM-TEVp ctrl']
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
# Initialize variables
final_rois_all = []
flyIDs = []
flash_resps = []
ON_steps = []
ON_plateau = []
ON_int = []
flash_corr = []
genotypes = []

for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        warnings.warn('Skipping non pickle file: {f}\n'.format(f=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    final_rois_all.append(workspace['final_rois'])
    
    stim_t = curr_rois[0].int_stim_trace[:94]
    fff_stim_trace = \
        np.around(np.concatenate((stim_t[20:],stim_t[0:30]),axis=0))
    for roi in curr_rois:
        if roi.corr_fff < 0:
            continue
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['Genotype'])
        trace = roi.int_con_trace[:94]
        
        
        flash_t = np.concatenate((trace[20:],trace[0:30]),axis=0)
        # Compute flash properties
        flash_resps.append(flash_t)
        on_step = trace[50:60].max()-trace[45:50].mean()
        ON_steps.append(on_step)
        on_plat = trace[84:94].mean()-trace[45:50].mean()
        ON_plateau.append(on_plat)
        on_int = trace[50:94].sum()
        ON_int.append(on_int)
        flash_corr.append(roi.corr_fff)
        
    
        
    print('{ds} successfully loaded\n'.format(ds=dataset))

#%% Flash figures
# Over flies and experiments
fff_dict = pac.compute_over_samples_groups(data = flash_resps,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
on_step_dict = pac.compute_over_samples_groups(data = ON_steps,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
on_plat_dict = pac.compute_over_samples_groups(data = ON_plateau,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
on_int_dict = pac.compute_over_samples_groups(data = ON_int,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
# Conc responses
# Plotting parameters
_, colors_d = pac.run_matplotlib_params()
colors = [colors_d['magenta'],colors_d['green1'],colors_d['green3'],
              colors_d['magenta']]
plt.close('all')
fig = plt.figure(figsize=(15, 3))
fig.suptitle('5sFFF properties',fontsize=12)

grid = plt.GridSpec(1,6 ,wspace=1, hspace=1)

# FFF responses
ax=plt.subplot(grid[0,:3])
ax2=plt.subplot(grid[0,3])
ax3=plt.subplot(grid[0,4])
ax4=plt.subplot(grid[0,5])

labels = ['']
fff_all_max = 0.0
t_trace = np.linspace(0,len(fff_stim_trace),len(fff_stim_trace))/10

if not genotype_labels:
    genotype_labels = fff_dict['experiment_ids'].keys()
for idx, genotype in enumerate(fff_dict['experiment_ids'].keys()):
    
    
    
    curr_data = fff_dict['experiment_ids'][genotype]
    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype_labels[idx],
                                           nflies =\
                                               len(curr_data['over_samples_means']),
                                           nROIs=\
                                               len(curr_data['all_samples']))
    mean = curr_data['over_groups_mean']
    if np.max(mean) > fff_all_max:
        fff_all_max = np.max(mean)
    error = curr_data['over_groups_error']
    ub = mean + error
    lb = mean - error
    ax.plot(t_trace,mean,color=colors[idx],alpha=.8,lw=3,label=gen_str)
    ax.fill_between(t_trace, ub, lb,
                     color=colors[idx], alpha=.4)
    scaler = np.abs(np.max(mean) - np.min(mean))
    
    curr_data = on_step_dict['experiment_ids'][genotype]
    error = curr_data['over_groups_error']
    pac.bar_bg(curr_data['over_samples_means'], idx+1, color=colors[idx], 
               ax=ax2,yerr=error)
    
    curr_data = on_plat_dict['experiment_ids'][genotype]
    error = curr_data['over_groups_error']
    pac.bar_bg(curr_data['over_samples_means'], idx+1, color=colors[idx], 
               ax=ax3,yerr=error)
    
    curr_data = on_int_dict['experiment_ids'][genotype]
    error = curr_data['over_groups_error']
    pac.bar_bg(curr_data['over_samples_means'], idx+1, color=colors[idx], 
               ax=ax4,yerr=error)
    
    
    labels.append(genotype_labels[idx])
    
    
plot_stim = (fff_stim_trace) 
ax.plot(t_trace,
        plot_stim/6+ fff_all_max,'--k',lw=1.5,alpha=.8)

ax.set_title('Response')  
ax.set_xlabel('Time (s)')
ax.set_ylabel('$\Delta F/F$')
ax.legend()

ax2.set_title('ON step')  
ax2.set_ylabel('$\Delta F/F$')
ax2.set_xticks(range(idx+2))
ax2.set_xlim((0,idx+2))
ax2.set_xticklabels(labels,rotation=45)
ax2.plot(list(ax2.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

ax3.set_title('ON plateau')  
ax3.set_ylabel('$\Delta F/F$')
ax3.set_xticks(range(idx+2))
ax3.set_xlim((0,idx+2))
ax3.set_xticklabels(labels,rotation=45)
ax3.plot(list(ax2.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

ax4.set_title('ON integral')  
ax4.set_ylabel('$\Delta F/F$')
ax4.set_xticks(range(idx+2))
ax4.set_xlim((0,idx+2))
ax4.set_xticklabels(labels,rotation=45)
ax4.plot(list(ax2.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

fig.tight_layout()

if 1:
    # Saving figure
    save_name = '{exp_t}_FFF'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
# plt.close('all')


