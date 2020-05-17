#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:39:02 2020

@author: burakgur
"""

import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import post_analysis_core as pac
import ROI_mod
import seaborn as sns
# %% Setting the directories
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'analyzed_data')
results_save_dir = os.path.join(initialDirectory,
                                'results/200310_GluClalpha_NI_cut_experiments')


# %% Load datasets and desired variables
exp_folder = '200310_GluClalpha_NI_cut_experiments/WN'
exp_t = '200310_GluClalpha_NI_cut_experiments_WN'
genotype_labels = ['ICAM-TEVp ctrl','Exp','TEVsite ctrl']
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
# Initialize variables
final_rois_all =np.array([])
flyIDs = []
flash_resps = []
genotypes = []

for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        warnings.warn('Skipping non pickle file: {f}\n'.format(f=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    final_rois_all = np.append(final_rois_all,np.array(workspace['final_rois']))
    for roi in curr_rois:
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['Genotype'])
    print('{ds} successfully loaded\n'.format(ds=dataset))

genotypes = np.array(genotypes)  
        
    
        
    
    
#%% Plot all STRFs
for idx,genotype in enumerate(np.unique(genotypes)):
    mask = genotypes == genotype
    mask = genotypes == genotype
    curr_rois = final_rois_all[mask]
    curr_filters = np.array(map(lambda roi : \
                                       roi.sta.T[np.where(roi.sta.T==roi.sta.max())[0]],
                     curr_rois))
        
    filter_masks =curr_filters.max(axis=2)>0.03
    
    curr_fly_ids = np.array(flyIDs)[mask]
    curr_fly_filtered = curr_fly_ids[np.array( filter_masks[:,0])]
    curr_rois_filtered = curr_rois[np.array( filter_masks[:,0])]
    
    fig1= ROI_mod.plot_STRFs(curr_rois_filtered, f_w=10,
                             number=len(curr_rois_filtered),cmap='coolwarm')
    fig1.suptitle(genotype)
    f1_n = 'All_STRFs_%s' % (genotype)
    os.chdir(results_save_dir)
    fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                transparent=False,dpi=300)
    
#%% Avg STRFs
plt.close('all')
fig1 = plt.figure(figsize=(16, 12))
grid = plt.GridSpec(3, 3, wspace=0.2, hspace=0.2)

_, colors_d = pac.run_matplotlib_params()
colors = [colors_d['green2'],colors_d['magenta'],colors_d['green1']]
ax2 = plt.subplot(grid[2:,:])
for idx,genotype in enumerate(np.unique(genotypes)):
    ax = plt.subplot(grid[:2,idx])
    mask = genotypes == genotype
    curr_rois = final_rois_all[mask]
    curr_filters = np.array(map(lambda roi : \
                                       roi.sta.T[np.where(roi.sta.T==roi.sta.max())[0]],
                     curr_rois))
        
    filter_masks =curr_filters.max(axis=2)>0.01
    
    curr_fly_ids = np.array(flyIDs)[mask]
    curr_fly_filtered = curr_fly_ids[np.array( filter_masks[:,0])]
    curr_rois_filtered = curr_rois[np.array( filter_masks[:,0])]
    curr_filters_filtered = curr_filters[filter_masks]
    mean_strf = np.mean(np.array(map(lambda roi : \
                 np.roll(roi.sta.T, roi.sta.T.shape[0]/2- \
                         int(np.where(roi.sta.T==roi.sta.max())[0]), axis=0),
                     curr_rois_filtered)),axis=0)
    sns.heatmap(mean_strf, cmap='coolwarm', ax=ax,center=0,cbar=False)
    ax.axis('off')
    ax.set_title(genotype,fontsize='xx-small')
    
    ax2 = plt.subplot(grid[2:,1])
    
    mean = np.mean(curr_filters_filtered,axis=0).T
    # mean = (mean-mean.min())/(mean.max() - mean.min())
    # mean = (mean)/(mean.max())
    error = np.std(curr_filters_filtered,axis=0).T / np.sqrt(curr_filters_filtered.shape[0])
    
    label = "{l}, {f} {ROI}".format(l=genotype_labels[idx],
                                    f=len(np.unique(curr_fly_filtered)),
                                    ROI = curr_filters_filtered.shape[0])
    
    ub = mean + error
    lb = mean - error
    t_trace = np.linspace(-len(mean),0,len(mean))*50/1000
    ax2.plot(t_trace,mean,alpha=.8,lw=3,color=colors[idx],
             label=label)
    ax2.fill_between(t_trace, ub[:], lb[:], alpha=.4,color=colors[idx])
    
    ax2.legend()
    ax2.set_xlabel('Time(s)')
f1_n = 'Mean_STRFs_together_0p1threshold_%s' % (exp_t)
os.chdir(results_save_dir)
fig1.savefig('%s.pdf'% f1_n, bbox_inches='tight',
            transparent=False,dpi=300)
    
    