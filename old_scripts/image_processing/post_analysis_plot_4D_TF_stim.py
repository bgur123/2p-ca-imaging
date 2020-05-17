#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:49:31 2019

@author: burakgur

Continuation for analysis and plotting of stimulus with different frequencies and 
directions.
"""

#%% Process data

# Get rid of directions
roi_data = big_df[['Freq','Max_resp','ROI_ID','flyID','toxin','ROI_type']].\
            groupby(['toxin','Freq','ROI_ID','flyID','ROI_type'],as_index=False).max()

# Calculating best frequency
BF_idx = roi_data.groupby(['ROI_ID','flyID','toxin','ROI_type'],
                          as_index=False)['Max_resp'].idxmax()
roi_BF_data = roi_data.iloc[BF_idx]

# Getting fly means
fly_data = roi_data.groupby(['Freq','flyID','toxin'],as_index=False).mean()

control_roi_data = roi_data[roi_data['toxin'] == 'No_toxin']
toxin_roi_data = roi_data[roi_data['toxin'] == 'Agi_200nM']

roi_num_control = len(np.unique(control_roi_data['ROI_ID']).tolist())
roi_num_toxin = len(np.unique(toxin_roi_data['ROI_ID']).tolist())

fly_num_control = len(np.unique(control_roi_data['flyID']).tolist())
fly_num_toxin = len(np.unique(toxin_roi_data['flyID']).tolist())

fly_data_control = control_roi_data.groupby(['Freq','flyID'],as_index=False).mean()
fly_data_toxin = toxin_roi_data.groupby(['Freq','flyID'],as_index=False).mean()

#%% Find DSI
control_bf_data = roi_BF_data[roi_BF_data['toxin'] == 'No_toxin']
toxin_bf_data = roi_BF_data[roi_BF_data['toxin'] == 'Agi_200nM']

dsi_df = findDSI(control_big_df)
control_bf_data = pd.merge(control_bf_data, dsi_df, on='ROI_ID')

dsi_df = findDSI(toxin_big_df)
toxin_bf_data = pd.merge(toxin_bf_data, dsi_df, on='ROI_ID')

frames = [control_bf_data, toxin_bf_data]
roi_BF_data = pd.concat(frames)


#%%White plots
sns.set(style="ticks", context="talk",rc={"font.size":14,"axes.titlesize":12,
                                          "axes.labelsize":14,'xtick.labelsize':12,
                                          'ytick.labelsize' : 12,
                                          'legend.fontsize':12})
plt.style.use("default")
sns.set_palette("Set2")
colors = sns.color_palette(palette='Set2', n_colors=2)
#%%
#plot_byCategory(big_df, 'trace', 'ROI_type', conditions, save_dir = False)
no_dir_data_idx = big_df.groupby(['Freq','flyID','ROI_ID','toxin','ROI_type'],
                          as_index=False)['Max_resp'].idxmax()
data_trace_no_dir = big_df.iloc[no_dir_data_idx]
data_trace_no_dir = data_trace_no_dir[data_trace_no_dir['toxin']=='No_toxin']
#data_trace_no_dir = data_trace_no_dir[data_trace_no_dir['toxin']=='Agi_200nM']

#%%

#%% Plot according to categories
#plot_byCategory(data_trace_no_dir, 'trace', 'ROI_ID', 'Freq', conditions, 
#                stimulus_information['baseline_duration_before_epoch'],
#                title_string,
#                save_dir = os.path.join(save_dir,'ROI_responses'))

#%% Plot according to subcategories
plot_by_cat_single_ROI(data_trace_no_dir, 'trace', 'flyID', 
                           'ROI_ID', 'Freq',
                    conditions, stimulus_information['baseline_duration_before_epoch'],
                    title_string, save_dir = os.path.join(save_dir,'ROI_responses'),
                    toxin = 'No_toxin')
#%%Dark plots
#sns.set(style="ticks", context="talk",rc={"font.size":14,"axes.titlesize":14,
#                                          "axes.labelsize":14,'xtick.labelsize':12,
#                                          'ytick.labelsize' : 12,
#                                          'legend.fontsize':12})
#
#plt.style.use("dark_background")
#sns.set_palette('hls')
#flierprops = dict(markerfacecolor='w', markersize=5,
#              linestyle='none')



##%%PLOT OF ALL the ROI TF TUNINGS
#fig1, ax = plt.subplots(ncols=2,figsize=(10, 7), sharex=True, sharey=True,
#                        facecolor='w', edgecolor='k')
#fig1.suptitle(title_string,fontsize=12)
#
## Plotting all ROIs
#
#sns.lineplot(x="Freq", y="Max_resp", hue="ROI_ID",ci=68,data=control_roi_data,
#             ax=ax[0],lw=1,markers=True,legend=False,**{'alpha':0.3})
#sns.lineplot(x="Freq", y="Max_resp",ci=68,data=fly_data_control,
#             ax=ax[0],lw=4,markers=True,legend='full',**{'alpha':1,'color':colors[0]})
#
##ax[0].legend(['mean over flies','single ROIs'],fontsize = 10)
#ax[0].set(xscale="log")
#ax[0].set_title('Control N: %d(%d)'% (fly_num_control,
#                                         roi_num_control))
#ax[0].set(xlabel='TF (Hz)', ylabel='dF/F')
#
## Plotting means over flies
#
#sns.lineplot(x="Freq", y="Max_resp", hue="ROI_ID", ci=68,data=toxin_roi_data,
#             ax=ax[1],lw=1,markers=True,sizes=[1,5],legend = False,**{'alpha':0.3})
#sns.lineplot(x="Freq", y="Max_resp",ci=68,data=fly_data_toxin,
#             ax=ax[1],lw=4,markers=True,legend='full',**{'alpha':1,'color':'k'})
##ax[1].legend(['mean over flies','single ROIs'],fontsize = 10)
#ax[1].set(xscale="log")
#ax[1].set_title('Agitoxin 200nM N: %d(%d)'% (fly_num_toxin,
#                                                roi_num_toxin))
#ax[1].set(xlabel='TF (Hz)', ylabel='dF/F')
#
#if saveFig:
#    # Saving figure
#    save_name = 'Overall_TF_tunings_%s' % (title_string)
#    os.chdir(save_dir)
#    plt.savefig('%s.png'% save_name, bbox_inches='tight')
#    print('Figure saved')
##    plt.close(fig1)

#%% Fly means
fig1, ax = plt.subplots(ncols=2,figsize=(10, 7), sharex=True, sharey=True,
                        facecolor='w', edgecolor='k')
figtitle = 'Single flies %s' % title_string
fig1.suptitle(figtitle,fontsize=12)

# Plotting all flies
sns.lineplot(x="Freq", y="Max_resp", hue="flyID",ci=None,data=control_roi_data,
             ax=ax[0],lw=2,markers=True,legend='brief',
             palette= sns.color_palette("gray", fly_num_control),
             **{'alpha':0.5})

sns.lineplot(x="Freq", y="Max_resp",ci=68,data=fly_data_control,
             ax=ax[0],lw=4,markers=True,legend='full',**{'alpha':1,'color':colors[0]})

#ax[0].legend(['mean over flies','single fly'],fontsize = 10)
ax[0].set(xscale="log")
ax[0].set_title('Control N: %d(%d)'% (fly_num_control,
                                         roi_num_control))
ax[0].set(xlabel='TF (Hz)', ylabel='dF/F')

# Plotting means over flies

sns.lineplot(x="Freq", y="Max_resp", hue="flyID", ci=None,data=toxin_roi_data,
             ax=ax[1],lw=2,markers=True,sizes=[1,5],legend = 'brief',
             palette= sns.color_palette("gray", fly_num_toxin),
             **{'alpha':0.5})
sns.lineplot(x="Freq", y="Max_resp",ci=68,data=fly_data_toxin,
             ax=ax[1],lw=4,markers=True,legend='full',**{'alpha':1,'color':colors[1]})
#ax[1].legend(['mean over flies','single fly'],fontsize = 10)
ax[1].set(xscale="log")
ax[1].set_title('Agitoxin 200nM N: %d(%d)'% (fly_num_toxin,
                                                roi_num_toxin))
ax[1].set(xlabel='TF (Hz)', ylabel='dF/F')


if saveFig:
    # Saving figure
    save_name = 'Overall_TF_fly_tunings_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
#    plt.close(fig1)
    
#%% Extensive PLOT OF fly means
fig1, ax = plt.subplots(ncols=2,figsize=(14, 7), sharex=False, sharey=True,
                        facecolor='w', edgecolor='k')
fig1.suptitle(title_string,fontsize=12)


sns.lineplot(x="Freq", y="Max_resp",hue='toxin',ci=68,data=fly_data,hue_order = ['No_toxin','Agi_200nM'], 
             ax=ax[0],lw=3,markers=True,legend='full',**{'alpha':1,'color':'k'})

#ax[0].legend(['mean over flies','single fly'],fontsize = 10)
ax[0].set(xscale="log")
ax[0].set(xlabel='TF (Hz)', ylabel='dF/F')

# Plotting means over flies


#sns.swarmplot(x="Freq", y="Max_resp",hue='toxin',data=fly_data,
#             ax=ax[1])
#sns.barplot(x="Freq", y="Max_resp",data=fly_data_control,
#             ax=ax[1],dodge=False,alpha=1)
#sns.barplot(x="Freq", y="Max_resp",data=fly_data_toxin,
#             ax=ax[1],dodge=False,alpha=0.8,color='w')

sns.barplot(x="Freq", y="Max_resp",hue='toxin',hue_order = ['No_toxin','Agi_200nM'],
            data=fly_data,
             ax=ax[1],dodge=True,saturation=.7)

ax[1].set(xlabel='TF (Hz)', ylabel='dF/F')


if saveFig:
    # Saving figure
    save_name = 'Overall_analysis_fly_tunings_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
#    plt.close(fig1)

#%%
from scipy.stats import ttest_ind

cat1 = fly_data[fly_data['toxin']=='No_toxin']
cat2 = fly_data[fly_data['toxin']=='Agi_200nM']

cat11 = cat1[cat1['Freq']==0.1]['Max_resp']
cat22 = cat2[cat2['Freq']==0.1]['Max_resp']


ttest_ind(cat11, cat22)
#%% ROI tuning curves
unique_control_flies = np.unique(control_roi_data['flyID']).tolist()

subPlotNumbers = len(unique_control_flies)
nrows = int(round(float(subPlotNumbers)/float(2)))
if nrows == 1:
    ncols = 1
else:
    ncols = 2
        
fig2, ax = plt.subplots(ncols,nrows,figsize=(14, 12), sharex=True, 
                        sharey=True,
                        facecolor='w', edgecolor='k')
fig2.suptitle(title_string,fontsize=12)
ax = ax.flatten()

layer_filter = 'LobDen'
for ax_num, flyNum in enumerate(unique_control_flies):
    curr_fly_data = control_roi_data[control_roi_data['flyID']==flyNum]
    curr_fly_data = curr_fly_data[curr_fly_data['ROI_type']==layer_filter]
    sns.lineplot(x="Freq", y="Max_resp",data=curr_fly_data,hue='ROI_ID',
             ax=ax[ax_num],lw=2,alpha=1, markers=True)
#    sns.lineplot(x="Freq", y="Max_resp",data=curr_fly_data,
#             ax=ax[ax_num],lw=1,alpha=0.5, color='k',markers=True)
    ax[ax_num].legend(fontsize=6)
    ax[ax_num].set(xscale="log")
    ax[ax_num].set_title('Fly %d'% flyNum)
    ax[ax_num].set(xlabel='TF (Hz)', ylabel='dF/F')
    
if saveFig:
    # Saving figure
    save_name = 'Single_fly_ROIs_TF_tunings_%s_%s_control' % (title_string,layer_filter)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
#%% Fly to fly variability check
unique_control_flies = np.unique(control_roi_data['flyID']).tolist()
unique_toxin_flies = np.unique(toxin_roi_data['flyID']).tolist()
common_flies = [item for item in unique_toxin_flies if item in unique_control_flies]

subPlotNumbers = len(common_flies)
nrows = int(round(float(subPlotNumbers)/float(2)))
if nrows == 1:
    ncols = 1
else:
    ncols = 2
        
fig2, ax = plt.subplots(ncols,nrows,figsize=(14, 12), sharex=True, 
                        sharey=True,
                        facecolor='w', edgecolor='k')
fig2.suptitle(title_string,fontsize=12)
ax = ax.flatten()
    
for ax_num, flyNum in enumerate(common_flies):
    curr_fly_data = roi_data[roi_data['flyID']==flyNum]
    sns.lineplot(x="Freq", y="Max_resp",ci=68,data=curr_fly_data,hue='toxin',
             hue_order = ['No_toxin','Agi_200nM'],ax=ax[ax_num],lw=2,
             markers=True)
    ax[ax_num].legend(fontsize=10)
    ax[ax_num].set(xscale="log")
    ax[ax_num].set_title('Fly %d'% flyNum)
    ax[ax_num].set(xlabel='TF (Hz)', ylabel='dF/F')
    
if saveFig:
    # Saving figure
    save_name = 'Single_fly_TF_tunings_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
#    plt.close(fig1)
#%% Plot best frequency for all ROIs
fig3, ax = plt.subplots(1,figsize=(7, 7), sharex=True, 
                        sharey=True,
                        facecolor='w', edgecolor='k')
fig3.suptitle(title_string,fontsize=12)


normalized_counts = (roi_BF_data.groupby(['toxin'])['Freq']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(1)
                     .reset_index()
                     .sort_values('Freq'))

sns.barplot(x="Freq", y='percentage' , hue ='toxin', 
            hue_order = ['No_toxin','Agi_200nM'], data=normalized_counts,ax=ax);
ax.set(xlabel='Best Frequency (Hz)', ylabel='Fraction of ROIs')


    
if saveFig:
    # Saving figure
    save_name = 'Histogram_BF_ ROIs_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
#%%

##%% Plot best frequency for all ROIs
#plt.close('all')
#fig3, ax = plt.subplots(1,figsize=(7, 7), sharex=True, 
#                        sharey=True,
#                        facecolor='k', edgecolor='w')
#fig3.suptitle(title_string,fontsize=12)
#
##roi_BF_data[['Freq','toxin']].hist(by = 'toxin',ax=ax,bins=20);
#sns.countplot(x="Freq", hue ='ROI_type',data=control_bf_data,ax=ax);
##ax.set(xscale="log")
#ax.set(xlabel='Best Frequency (Hz)', ylabel='Counts')
#%% Plot DSI 
fig4, ax = plt.subplots(nrows=1,figsize=(8, 8),facecolor='w', edgecolor='k')
fig4.suptitle(title_string,fontsize=12)

#sns.catplot(x="ROI_type", y="DSI", hue="toxin", kind="swarm", data=roi_BF_data,
#            ax= ax);


sns.violinplot(x="ROI_type", y="DSI", hue="toxin",
               data=roi_BF_data, split=True,
               scale="count", inner="quartile",
               scale_hue=False, bw=.2, ax=ax,hue_order = ['No_toxin','Agi_200nM'])
ax.set_title('Control N: %d(%d), Toxin N: %d(%d)'% (fly_num_control,roi_num_control,
                                                     fly_num_toxin,roi_num_toxin))

if saveFig:
    # Saving figure
    save_name = 'DSI_distribution_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
    
#%% DSI distribution fly by fly
unique_control_flies = np.unique(control_roi_data['flyID']).tolist()
unique_toxin_flies = np.unique(toxin_roi_data['flyID']).tolist()
common_flies = [item for item in unique_toxin_flies if item in unique_control_flies]

subPlotNumbers = len(common_flies)
nrows = int(round(float(subPlotNumbers)/float(2)))
if nrows == 1:
    ncols = 1
else:
    ncols = 2
        
fig2, ax = plt.subplots(ncols,nrows,figsize=(20, 16), sharex=True, 
                        sharey=True,
                        facecolor='w', edgecolor='k')
fig2.suptitle(title_string,fontsize=12)
ax = ax.flatten()
    
for ax_num, flyNum in enumerate(common_flies):
    curr_fly_data = roi_BF_data[roi_BF_data['flyID']==flyNum]
    sns.barplot(x="ROI_type", y="DSI", hue="toxin",
                  data=curr_fly_data, ax=ax[ax_num],
                  hue_order = ['No_toxin','Agi_200nM'])
    
    ax[ax_num].set_title('Fly %d'% flyNum)
if saveFig:
    # Saving figure
    save_name = 'DSI_flyfly_distribution_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
    
#%% DSI fly by fly CONTROL
unique_control_flies = np.unique(control_bf_data['flyID']).tolist()

subPlotNumbers = len(unique_control_flies)
nrows = int(round(float(subPlotNumbers)/float(2)))
if nrows == 1:
    ncols = 1
else:
    ncols = 2
        
fig2, ax = plt.subplots(ncols,nrows,figsize=(15, 9), sharex=True, 
                        sharey=True,
                        facecolor='w', edgecolor='k')
fig2.suptitle(title_string,fontsize=12)
ax = ax.flatten()
    
for ax_num, flyNum in enumerate(unique_control_flies):
    curr_fly_data = control_bf_data[control_bf_data['flyID']==flyNum]
    sns.barplot(x="ROI_type", y="DSI", data=curr_fly_data, ax=ax[ax_num],
                linewidth=0, saturation=.7,ci='sd',
                errcolor=[0.3 ,0.3, 0.3 ,1])
    sns.swarmplot(x="ROI_type", y="DSI", data=curr_fly_data, ax=ax[ax_num],color='k')
#    sns.boxplot(x="ROI_type", y="DSI",notch=True,
#                  data=curr_fly_data, ax=ax[ax_num])
#    sns.swarmplot(x="ROI_type", y="DSI", data=curr_fly_data, 
#                  color="w",ax=ax[ax_num])
    ax[ax_num].set_title('Control Fly %d'% flyNum)
    
    if ax[ax_num].get_xticklabels():
        for tick in ax[ax_num].get_xticklabels():
            tick.set_rotation(45)
    
if saveFig:
    # Saving figure
    save_name = 'Ctrl_Single_fly_DSI_dist_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
    
#%% DSI fly by fly TOXIN
unique_toxin_flies = np.unique(toxin_bf_data['flyID']).tolist()

subPlotNumbers = len(unique_toxin_flies)
nrows = int(round(float(subPlotNumbers)/float(2)))
if nrows == 1:
    ncols = 1
else:
    ncols = 2
        
fig2, ax = plt.subplots(ncols,nrows,figsize=(15, 9), sharex=True, 
                        sharey=True,
                        facecolor='w', edgecolor='k')
fig2.suptitle(title_string,fontsize=12)
ax = ax.flatten()
    
for ax_num, flyNum in enumerate(unique_toxin_flies):
    curr_fly_data = toxin_bf_data[toxin_bf_data['flyID']==flyNum]

    sns.barplot(x="ROI_type", y="DSI", data=curr_fly_data, ax=ax[ax_num],
                linewidth=0, saturation=.7,ci='sd',
                errcolor=[0.3 ,0.3, 0.3 ,1])
    sns.swarmplot(x="ROI_type", y="DSI", data=curr_fly_data, 
                  color="k",ax=ax[ax_num])
    ax[ax_num].set_title('Toxin Fly %d'% flyNum)
    
    if ax[ax_num].get_xticklabels():
        for tick in ax[ax_num].get_xticklabels():
            tick.set_rotation(45)
    
if saveFig:
    # Saving figure
    save_name = 'Toxin_Single_fly_DSI_dist_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
#%% DSI vs BF fly by fly
unique_control_flies = np.unique(control_bf_data['flyID']).tolist()

subPlotNumbers = len(unique_control_flies)
nrows = int(round(float(subPlotNumbers)/float(2)))
if nrows == 1:
    ncols = 1
else:
    ncols = 2
        
fig2, ax = plt.subplots(ncols,nrows,figsize=(15, 9), sharex=True, 
                        sharey=True,
                        facecolor='w', edgecolor='k')
fig2.suptitle(title_string,fontsize=12)
ax = ax.flatten()
    
for ax_num, flyNum in enumerate(unique_control_flies):
    curr_fly_data = control_bf_data[control_bf_data['flyID']==flyNum]
    sns.regplot(x="Freq", y="DSI",ax=ax[ax_num], data=curr_fly_data,fit_reg=False,
            scatter_kws={'alpha':1, 'edgecolor':'k', 'linewidth':'1'})
    curr_corr = curr_fly_data['DSI'].corr(curr_fly_data['Freq'],method='spearman')
    ax[ax_num].set(xscale="log")
    ax[ax_num].set_title('Control Fly %d, Corr: %.3f'% (flyNum,curr_corr))
    
    ax[ax_num].legend(fontsize=10)
    ax[ax_num].set_ylim(-0.1,1)
    ax[ax_num].set(xlabel='Best Frequency (Hz)', ylabel='DSI')
    
if saveFig:
    # Saving figure
    save_name = 'Ctrl_Single_fly_DSI_vs_BF_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
#%%
unique_toxin_flies = np.unique(toxin_bf_data['flyID']).tolist()

subPlotNumbers = len(unique_toxin_flies)
nrows = int(round(float(subPlotNumbers)/float(2)))
if nrows == 1:
    ncols = 1
else:
    ncols = 2
        
fig2, ax = plt.subplots(ncols,nrows,figsize=(15, 9), sharex=True, 
                        sharey=True,
                        facecolor='2', edgecolor='k')
fig2.suptitle(title_string,fontsize=12)
ax = ax.flatten()
    
for ax_num, flyNum in enumerate(unique_toxin_flies):
    curr_fly_data = toxin_bf_data[toxin_bf_data['flyID']==flyNum]
    sns.regplot(x="Freq", y="DSI",ax=ax[ax_num], data=curr_fly_data,fit_reg=False,
            scatter_kws={'alpha':1, 'edgecolor':'k', 'linewidth':'1'})
    curr_corr = curr_fly_data['DSI'].corr(curr_fly_data['Freq'],method='spearman')
    ax[ax_num].set(xscale="log")
    ax[ax_num].set_title('Toxin Fly %d, Corr: %.3f'% (flyNum,curr_corr))
    
    ax[ax_num].legend(fontsize=10)
    ax[ax_num].set_ylim(-0.1,1)
    ax[ax_num].set(xlabel='Best Frequency (Hz)', ylabel='DSI')
    
if saveFig:
    # Saving figure
    save_name = 'Toxin_Single_fly_DSI_vs_BF_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.png'% save_name, bbox_inches='tight')
    print('Figure saved')
#%% Best frequency per layer
#%% Plot DSI  FOR CONTROL
fig4, ax = plt.subplots(nrows=1,figsize=(8, 8),facecolor='w', edgecolor='k')
fig4.suptitle(title_string,fontsize=12)

normalized_counts = (control_bf_data.groupby(['ROI_type'])['Freq']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(1)
                     .reset_index()
                     .sort_values('Freq'))

sns.barplot(x="Freq", y='percentage' , hue ='ROI_type', data=normalized_counts,ax=ax);
ax.set(xlabel='Best Frequency (Hz)', ylabel='Fraction of ROIs')
ax.legend(loc='upper right')
#sns.violinplot(x="Freq", y="ROI_type", hue="toxin",
#               data=roi_BF_data, split=True,
#               scale="count", inner="stick",
#               scale_hue=False, bw=.2, ax=ax,hue_order = ['Agi_200nM','No_toxin'])
#ax.set(xscale="log")

if saveFig:
    # Saving figure
    draw()
    save_name = 'BF_distribution_%s' % (title_string)
    os.chdir(save_dir)
    plt.savefig('%s.svg'% save_name, bbox_inches='tight')
    print('Figure saved')