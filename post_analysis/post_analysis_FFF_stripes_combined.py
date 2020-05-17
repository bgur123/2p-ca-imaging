#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:14:49 2020

@author: burakgur
"""

import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import ROI_mod
import post_analysis_core as pac

# %% Setting the directories
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/selected')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '191220_GluClflpTEV_NI_1')
results_save_dir = os.path.join(initialDirectory,
                                'results/191220_GluClflpTEV_NI_1')



# %% Load datasets and desired variables
exp_t = '191220_GluCl-flpTEV-NI-1_combinedROIs'
datasets_to_load = ['191216bg_fly5_combined_data_T_1_4_5_6_7.pickle',
                    '191216bg_fly6_combined_data_T_1_3_4_5_8.pickle',
                    '191216bg_fly7_combined_data_T_1_2_3_5_6.pickle',
                    '191218bg_fly1_combined_data_T_1_2_3_4_5.pickle',
                    '191218bg_fly2_combined_data_T_5_6_7.pickle',
                    '191218bg_fly3_combined_data_T_1_2_3_4_5.pickle',
                    '191218bg_fly4_combined_data_T_1_2_3_4_5.pickle',
                    '191218bg_fly5_combined_data_T_1_2_3_4_5.pickle',
                    '191218bg_fly6_combined_data_T_1_2_3_4_5.pickle',
                    '191219bg_fly3_combined_data_T_1_2_5_6_7.pickle',
                    '191218bg_fly6_combined_data_T_1_2_3_4_5.pickle',
                    '191220bg_fly1_combined_data_T_5_6_9_10_11.pickle']

# Initialize variables
final_rois_all = []
flyIDs = []
flash_corr = []
reliabilities = []
SNRs = []

genotypes = []

vert_RF_ON = []
hor_RF_ON = []
hor_RF_ON_centered = []
hor_RF_ON_gauss = []
hor_RF_ON_gauss_fwhm = []
hor_RF_ON_gauss_rsq = []
vert_RF_OFF = []
hor_RF_OFF = []
SNRs = []
Rsq_vals = []


for idataset, dataset in enumerate(datasets_to_load):
    load_path = os.path.join(saveOutputDir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    final_rois_all.append(workspace['final_rois'])
    fff_stim_trace = \
        np.around(np.concatenate((curr_rois[0].int_stim_trace[20:],
                                  curr_rois[0].int_stim_trace[0:30]),axis=0))
        
    
    for roi in curr_rois:
        if (roi.hor_RF_ON_gauss is None) or \
            (roi.hor_RF_ON_Rsq <0.5):
            continue
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(roi.experiment_info['Genotype'])

        flash_corr.append(roi.corr_fff)
        
        # Rel SNR
        reliabilities.append(np.mean(roi.reliabilities))
        SNRs.append(np.mean(roi.SNRs))
        
        # RF props
        vert_RF_ON.append(roi.vert_RF_ON_trace)
        hor_RF_ON.append(roi.hor_RF_ON_trace)
        centered = np.full(np.shape(roi.hor_RF_ON_trace),np.nan)
        center = np.argmax(roi.hor_RF_ON_trace)
        if center > 30:
            centered[:30] = roi.hor_RF_ON_trace[center-30:center]
            centered[30:30+60-center] = roi.hor_RF_ON_trace[center:]
        else:
            centered[30-center:30] = roi.hor_RF_ON_trace[:center]
            centered[30:] = roi.hor_RF_ON_trace[center:center+30]
        
        hor_RF_ON_centered.append(centered)
        
        
        hor_RF_ON_gauss.append(roi.hor_RF_ON_gauss)
        hor_RF_ON_gauss_fwhm.append(roi.hor_RF_ON_fwhm)
        hor_RF_ON_gauss_rsq.append(roi.hor_RF_ON_Rsq)
        # vert_RF_OFF.append(roi.vert_RF_OFF_trace)
        # hor_RF_OFF.append(roi.hor_RF_OFF_trace)
        
        rsq = np.array(roi.stripe_Rsq_vals)
        Rsq_vals.append(np.min(rsq[rsq.argsort()[-2:]]))
        SNRs.append(np.max(roi.SNRs))
        
        
        
    
        
    print('{ds} successfully loaded\n'.format(ds=dataset))



unique_gens = np.unique(genotypes)
#%% Stripes analysis
fwhm = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_fwhm,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)
rsqd = pac.compute_over_samples_groups(data = hor_RF_ON_gauss_rsq,
                                       group_ids= flyIDs, error ='SEM',
                                       experiment_ids = genotypes)

df = pd.DataFrame(list(zip(hor_RF_ON_gauss_fwhm, hor_RF_ON_gauss_rsq,genotypes,
                           SNRs)), 
               columns =['fwhm', 'rsq','gen','SNR']) 
df['gen'][df['gen']=='Pos_Mi1Rec__plus_GluClflpSTOPD'] = 'Pos'
df['gen'][df['gen']=='Exp_Mi1Rec__GluClflpTEVNI_GluClflpSTOPD'] = 'Exp'

colorss = pac.run_matplotlib_params()
colors = [colorss[5],colorss[2]]
plt.close('all')
fig = plt.figure(figsize=(10, 6))
fig.suptitle('RF properties',fontsize=12)
grid = plt.GridSpec(len(unique_gens),7 ,wspace=1, hspace=0.5)
ax1 = plt.subplot(grid[0,5:])
ax2 = plt.subplot(grid[1,5:])
labels = []
for idx , genotype in enumerate(unique_gens):
    labels.append(genotype[:3])
    ax = plt.subplot(grid[idx,:5])
    sns.heatmap(np.array(hor_RF_ON_centered)[np.array(genotypes) == genotype],
                cmap='coolwarm',center=0,ax=ax,yticklabels=20,
                xticklabels=10,vmax=4.5,vmin=-0.5)
    
    
    ax.set_xlabel('Position ($^\circ$)')
    ax.set_ylabel('ROI #')
    
    
    
    fwhm_data = fwhm['experiment_ids'][genotype]['all_samples']
    rsqd_data = rsqd['experiment_ids'][genotype]['all_samples']
    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype[:3],
                                           nflies =\
                                               len(fwhm['experiment_ids'][genotype]['over_samples_means']),
                                           nROIs=\
                                               len(fwhm['experiment_ids'][genotype]['all_samples']))
    ax.set_title(gen_str)      
    sns.distplot(rsqd_data,ax=ax2,hist_kws={"alpha": .5,'cumulative': False},
                 kde=True,kde_kws={"alpha": .8,'cumulative': False},
                 color=(colors[idx][0],colors[idx][1],colors[idx][2],1.0),
                 hist=False,bins=10)
    ax2.set_xlim((0.5, 1))

# sns.stripplot(x="gen", y="fwhm", hue="gen",palette=colors,data=df, 
#               dodge=True, jitter=True,order=labels,
#               alpha=.5,ax=ax1)
sns.violinplot(x='gen',y='fwhm',data=df,order=labels ,width=0.5,
                ax=ax1,palette=colors,inner="quartile",linewidth=1.5)
ax1.set_ylim((0, 25))
ax1.set_xlabel('')
ax1.set_title('FWHM') 
ax1.set_ylabel('FWHM ($^\circ$)')
ax1.legend('')
ax2.set_title('R$^2$') 
ax2.set_xlabel('R$^2$')
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)

#%%
import random
plt.close('all')
# Randomize ROIs
a=np.array(range(len(genotypes)))
random.shuffle(a)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams["axes.titlesize"] = 'x-small'
plt.rcParams["axes.labelsize"] = 'x-small'

fig1, ax1 = plt.subplots(ncols=3, nrows=3, figsize=(8,8))
axs = ax1.flatten()
for idx, ax in enumerate(axs):
    curr_idx = a[idx]
    if genotypes[curr_idx][:3] =='Exp':
        curr_color = colors[0]
    else:
        curr_color = colors[1]
    rsq1=np.around(hor_RF_ON_gauss_rsq[curr_idx],2)
    fwhm=np.around(hor_RF_ON_gauss_fwhm[curr_idx],1)
    ax.plot(hor_RF_ON_gauss[curr_idx], 'k',lw=2, 
            label="FWHM: {fwhm}$^\circ$ R$^2$ {rsq1}".format(fwhm=fwhm,rsq1=rsq1))
    ax.plot(hor_RF_ON[curr_idx],color=curr_color,lw=2)

    ax.legend()
fig1.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_randomexp'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.close('all')
#%%
sns.lmplot('rsq','fwhm',data=df,hue='gen',hue_order=labels,palette=colors,
           scatter_kws={'alpha':0.7, 's':20})
plt.ylim((0, 25))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    