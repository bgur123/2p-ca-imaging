#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:08:09 2020

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
                             '191210_T4T5_4th', 'delay_calculation_data')
results_save_dir = os.path.join(initialDirectory,
                                'results/200116_T4T5_delay_experiments/Delay_exps')


# %% Load datasets and desired variables
exp_t = '4th_exp_R64G09_Recomb_delay_calculation'
datasets_to_load = ['191209bg_fly1-TSeries-002_transfer_sima_STICA.pickle',
                    '191209bg_fly1-TSeries-003_transfer_sima_STICA.pickle',
                    '191209bg_fly2-TSeries-004_transfer_sima_STICA.pickle',
                    '191209bg_fly2-TSeries-005_transfer_sima_STICA.pickle',
                    '191210bg_fly1-TSeries-12102019-0944-009_transfer_sima_STICA.pickle',
                    '191210bg_fly1-TSeries-12102019-0944-010_transfer_sima_STICA.pickle',
                    '191210bg_fly2-TSeries-12102019-0944-002_transfer_sima_STICA.pickle',
                    '191210bg_fly2-TSeries-12102019-0944-003_transfer_sima_STICA.pickle',
                    '191210bg_fly2-TSeries-12102019-0944-009_transfer_sima_STICA.pickle',
                    '191210bg_fly2-TSeries-12102019-0944-010_transfer_sima_STICA.pickle',
                    '191212bg_fly2-TSeries-002_transfer_sima_STICA.pickle',
                    '191212bg_fly2-TSeries-004_transfer_sima_STICA.pickle',
                    '191212bg_fly3-TSeries-004_transfer_sima_STICA.pickle',
                    '191212bg_fly3-TSeries-005_transfer_sima_STICA.pickle',
                    '191213bg_fly2-TSeries-12132019-0909-005_transfer_sima_STICA.pickle',
                    '191213bg_fly2-TSeries-12132019-0909-006_transfer_sima_STICA.pickle',
                    '191213bg_fly3-TSeries-12132019-0909-004_transfer_sima_STICA.pickle',
                    '191216bg_fly3-TSeries-004_transfer_sima_STICA.pickle',
                    '191216bg_fly3-TSeries-005_transfer_sima_STICA.pickle',
                    ]
properties = ['Delay', 'Rsq', 'PD', 'SNR', 'CSI','Reliab','CS']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
# Initialize variables
for idataset, dataset in enumerate(datasets_to_load):
    load_path = os.path.join(saveOutputDir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    # for roi in curr_rois:
    #     for idx,trace in enumerate(roi.whole_trace_all_epochs.values()):
    #         plt.plot(trace,color=cmap(norm(idx)),label=str(idx))
    #         all_rois.append(curr_rois)
    # Rsq
    data_to_extract = ['SNR', 'CSI','reliability','resp_delay_fits_Rsq','PD',
                       'resp_delay_deg','CS']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    
    rsq = np.array(map(np.nanmin,roi_data['resp_delay_fits_Rsq']))
    rsq[rsq==None] = np.nan
    rsq[rsq<0] = np.nan
    
    delay = np.array(map(np.nanmin,roi_data['resp_delay_deg']))
    delay[delay==None] = np.nan
    delay = delay.astype(float)

    pref_dir = roi_data['PD']
    pd_S = map(str,list(map(int,pref_dir)))
    
    
    df_c = {}
    df_c['Delay'] = delay
    df_c['Rsq'] = rsq
    df_c['PD'] = pd_S
    df_c['SNR'] = roi_data['SNR']
    df_c['CSI'] = roi_data['CSI']
    df_c['Reliab'] = roi_data['reliability']
    df_c['CS'] = roi_data['CS']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))

#%% Thresholding
rsq_t = 0.7
rel_t = 0.55
threshold_dict = {'Rsq': rsq_t,
                      'Reliab': rel_t}
threshold_df = pac.apply_threshold_df(threshold_dict, combined_df)

# sns.pairplot(threshold_df, hue="CS",plot_kws=dict(s=20, linewidth=0,alpha=.5))
#%% Figures
plt.close('all')
colorss = pac.run_matplotlib_params()
colors = [colorss[-1],colorss[-2]]


ax= sns.jointplot(x='Delay', y='Rsq', kind="kde", color=colorss[5],
              data=combined_df)
ax.plot_joint(plt.scatter, c='w', s=10, linewidth=0.5, marker="o",
                       alpha=.0)
ax.set_axis_labels(xlabel='Degrees ($^\circ$)',ylabel='$R^2$')
ax.fig.suptitle('Thresholding: {p}/{a} ROIs'.format(p = len(threshold_df), a=len(combined_df)))

ax.fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_delay_all'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.close('all')
#%%
fig = plt.figure(figsize=(14, 3))

grid = plt.GridSpec(1,3 ,wspace=0.3, hspace=1)

ax1=plt.subplot(grid[0,0])
ax2=plt.subplot(grid[0,1])
ax3=plt.subplot(grid[0,2])

sns.scatterplot(x='Delay',y='Rsq',s=20, linewidth=0,alpha=.3,
             ax=ax1,data=combined_df,hue='Reliab',palette='inferno')

# sns.scatterplot(x='Delay',y='Rsq',s=20, linewidth=0,alpha=.5,hue='Reliab',
             # ax=ax1,data=threshold_df)

plt.setp(ax1.get_legend().get_texts(), fontsize='6')
ax1.plot([0, 60],[rsq_t,rsq_t],'--',color = 'k',alpha=.8)
ax1.set_xlim((0,60))
ax1.set_ylabel('R$^2$')
ax1.set_xlabel('Delay ($^\circ$)')
ax1.set_title('Thresholding: {p}/{a} ROIs'.format(p = len(threshold_df), a=len(combined_df)))
for idx, cs in enumerate(np.unique(threshold_df['CS'])):
    curr_df = threshold_df[threshold_df['CS'] == cs]
    lb = '{cs} {f} ({roi}) '.format(cs=cs,f=len(np.unique(curr_df['flyID'])),
                                   roi=len(curr_df))
    sns.distplot(curr_df['Delay'],ax=ax2,kde=True,
                 kde_kws={"alpha": .7,'cumulative': False},
                 color=(colors[idx][0],colors[idx][1],colors[idx][2],1.0),
                     hist=False,bins=10,label=lb)
    
ax2.set_xlabel('Delay ($^\circ$)')
ax2.set_title('Edge delay from RF center')

sns.violinplot(x='CS',y='Delay',width=0.5,ax=ax3,linewidth=1.5,data=threshold_df,
               palette=colors,inner="quartile")
# ax3.set_ylim((0, 25))
ax3.set_xlabel('')
ax3.set_ylabel('Delay ($^\circ$)')
ax3.legend('')
mon = threshold_df[threshold_df['CS']=='ON']['Delay'].mean()
moff = threshold_df[threshold_df['CS']=='OFF']['Delay'].mean()
son = threshold_df[threshold_df['CS']=='ON']['Delay'].std()
soff = threshold_df[threshold_df['CS']=='OFF']['Delay'].std()
ax3.set_title('OFF: {moff}$^\circ$$\pm${soff}$^\circ$   ON: {mon}$^\circ$$\pm${son}$^\circ$'.format(moff=np.around(moff,1),
                                                           mon=np.around(mon,1),
                                                           soff=np.around(soff,1),
                                                           son=np.around(son,1)))
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_delay_summary'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.close('all')
#%%

fig1 = ROI_mod.plot_delay_profile_examples(all_rois[6],number=None,f_w=None,lw=2,
                                           alpha=.8,
                                           colors=[colorss[5],colorss[0]])
        
if 1:
    # Saving figure 
    save_name = 'DelayProfileExamples'
    os.chdir(results_save_dir)
    fig1.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)