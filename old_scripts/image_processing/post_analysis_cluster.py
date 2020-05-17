#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:50:22 2019

@author: burakgur
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw # Important for vector graphics figures
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import ttest_ind

functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
from post_analysis_functions import dataIntoPDFrame_cluster
from post_analysis_functions import dataReadOrganizeSelect , plotRawData
#%% Database directory
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
dataBaseDir = os.path.join(initialDirectory,'database')
metaDataBaseFile = os.path.join(dataBaseDir, 'metaDataBase.txt')
# Directory where the processed data are stored for analysis
processedDataStoreDir= os.path.join(dataBaseDir, 'processed_data')

saveFig = True
save_dir = "/Users/burakgur/Desktop/tempSave"
#%% Search database, extract and select datasets of interest

# Conditions to select from the database
conditions = {
 'genotype':'T4T5-GMR59E08-lexA-rec-homozygous',
 'experiment-type' : 'Agitoxin_T4T5_clust_1'
             }
# Data variables to extract for further analysis
data_var_to_select = ['interpolatedAllRoi', 'interpolationRate', 'layerPosition',
                      'stimulus_information', 'flyID','selected_clusters_SNR','selected_clusters_TF',
                      'selected_clusters_DSI','selected_cluster_TF_tuning_no_edge','interpolatedAllRoi',
                      'cluster_layer_information', 'toxin','cluster_numbers']
# Data variables from the metadatabase that are going to be used in further analysis
# imageID has to be always there for extracting data
var_from_database_to_select = ['flyID', 'toxin', 'imageID']
selected_data = dataReadOrganizeSelect(conditions, data_var_to_select,
                                       var_from_database_to_select,
                                       dataBaseDir = dataBaseDir,
                                       processedDataStoreDir = 
                                       processedDataStoreDir, metaDataFile = 
                                       'metaDataBase.txt', data_ext = '*.pickle')
#%%
[big_df, tf_tuning_data_frame] = dataIntoPDFrame_cluster(selected_data)
plt.close('all')

#%%
roi_data = tf_tuning_data_frame[['Freq','Response','ROI_ID','flyID','toxin','Cluster_layer']].\
            groupby(['toxin','Freq','ROI_ID','flyID','Cluster_layer'],as_index=False).max()

#%%
toxin_name = 'Agi_200nM'
# Getting fly means
fly_data = roi_data.groupby(['Freq','flyID','toxin'],as_index=False).mean()

control_roi_data = roi_data[roi_data['toxin'] == 'No_toxin']
toxin_roi_data = roi_data[roi_data['toxin'] == toxin_name]

roi_num_control = len(np.unique(control_roi_data['ROI_ID']).tolist())
roi_num_toxin = len(np.unique(toxin_roi_data['ROI_ID']).tolist())

fly_num_control = len(np.unique(control_roi_data['flyID']).tolist())
fly_num_toxin = len(np.unique(toxin_roi_data['flyID']).tolist())

fly_data_control = control_roi_data.groupby(['Freq','flyID'],as_index=False).mean()
fly_data_toxin = toxin_roi_data.groupby(['Freq','flyID'],as_index=False).mean()

#%%White plots
sns.set(style="ticks", context="talk",rc={"font.size":14,"axes.titlesize":12,
                                          "axes.labelsize":14,'xtick.labelsize':12,
                                          'ytick.labelsize' : 12,
                                          'legend.fontsize':12})
plt.style.use("default")
sns.set_palette("Set2")
colors = sns.color_palette(palette='Set2', n_colors=2)
#%% Fly means
fig1, ax = plt.subplots(ncols=2,figsize=(10, 7), sharex=True, sharey=True,
                        facecolor='w', edgecolor='k')

# Plotting all flies
sns.lineplot(x="Freq", y="Response", hue="flyID",ci=None,data=control_roi_data,
             ax=ax[0],lw=2,markers=True,legend='brief',
             palette= sns.color_palette("gray", fly_num_control),
             **{'alpha':0.5})

sns.lineplot(x="Freq", y="Response",ci=68,data=fly_data_control,
             ax=ax[0],lw=4,markers=True,legend='full',**{'alpha':1,'color':colors[0]})

#ax[0].legend(['mean over flies','single fly'],fontsize = 10)
ax[0].set(xscale="log")
ax[0].set_title('Control N: %d(%d)'% (fly_num_control,
                                         roi_num_control))
ax[0].set(xlabel='TF (Hz)', ylabel='dF/F')

# Plotting means over flies

sns.lineplot(x="Freq", y="Response", hue="flyID", ci=None,data=toxin_roi_data,
             ax=ax[1],lw=2,markers=True,sizes=[1,5],legend = 'brief',
             palette= sns.color_palette("gray", fly_num_toxin),
             **{'alpha':0.5})
sns.lineplot(x="Freq", y="Response",ci=68,data=fly_data_toxin,
             ax=ax[1],lw=4,markers=True,legend='full',**{'alpha':1,'color':colors[1]})
#ax[1].legend(['mean over flies','single fly'],fontsize = 10)
ax[1].set(xscale="log")
ax[1].set_title('%s N: %d(%d)'% (toxin_name,fly_num_toxin,
                                                roi_num_toxin))
ax[1].set(xlabel='TF (Hz)', ylabel='dF/F')

#%%
#sns.countplot(x='BestFreq',data=big_df[big_df['toxin'] == 'No_toxin'])

sns.countplot(x='BestFreq',data=big_df[big_df['toxin'] == 'Agi_200nM'])

sns.boxplot(data=big_df, orient="h", palette="Set2")
#%% Extensive PLOT OF fly means
fig1, ax = plt.subplots(ncols=2,figsize=(14, 7), sharex=False, sharey=True,
                        facecolor='w', edgecolor='k')


sns.lineplot(x="Freq", y="Response",hue='toxin',ci=68,data=fly_data,hue_order = ['No_toxin',toxin_name], 
             ax=ax[0],lw=3,markers=True,legend='full',**{'alpha':1,'color':'k'})

#ax[0].legend(['mean over flies','single fly'],fontsize = 10)
ax[0].set(xscale="log")
ax[0].set(xlabel='TF (Hz)', ylabel='dF/F')

ax[0].set_title('Control N: %d(%d)'% (fly_num_control,
                                         roi_num_control))


# Plotting means over flies


#sns.swarmplot(x="Freq", y="Max_resp",hue='toxin',data=fly_data,
#             ax=ax[1])
#sns.barplot(x="Freq", y="Max_resp",data=fly_data_control,
#             ax=ax[1],dodge=False,alpha=1)
#sns.barplot(x="Freq", y="Max_resp",data=fly_data_toxin,
#             ax=ax[1],dodge=False,alpha=0.8,color='w')

sns.barplot(x="Freq", y="Response",hue='toxin',hue_order = ['No_toxin',toxin_name],
            data=fly_data,
             ax=ax[1],dodge=True,saturation=.7)

ax[1].set(xlabel='TF (Hz)', ylabel='dF/F')
ax[1].set_title('%s N: %d(%d)'% (toxin_name,fly_num_toxin,
                                                roi_num_toxin))

#%% Analysis start

# Understanding stimulus
representative_dataset = selected_data[selected_data.keys()[0]] 
stimulus_information = representative_dataset['stimulus_information']
epoch_freqs = np.unique(stimulus_information['epoch_frequency'][1:])

#%%
plt.close('all')
sns.violinplot(x='BestFreq', y='Z_depth', data=big_df,scale="count", inner="stick")
#plt.xscale('log')
#%%
tuning_curves = np.empty((0,12), int)
norm_tuning_cur_flies = {}
toxin_name = 'No_toxin'
selected_dataset = big_df[big_df['toxin'] == toxin_name]
unique_flies = np.unique(selected_dataset['flyID'])
selected_fly_num = len(unique_flies)
selected_roi_num = len(selected_dataset['flyID'])

for index, row in selected_dataset.iterrows():
    curr_ROI_data = {}
    
    curr_tuning = np.transpose(row['TF_tuning'])
    if row['DSI'] > 0:
        tuning_curves = np.append(tuning_curves, [curr_tuning[:12]], axis=0)
    if row['DSI'] < 0:
        tuning_curves = np.append(tuning_curves, [curr_tuning[12:]], axis=0)
        
norm_tf_tuning = normalize(tuning_curves,axis=1,norm='max')

for fly_num in unique_flies:
    curr_ROI_mask = (selected_dataset['flyID'] == fly_num)
    norm_tuning_cur_flies[fly_num]= norm_tf_tuning[curr_ROI_mask,:]
    
#%%
plt.close('all')





all_yerr = np.std(norm_tf_tuning,axis=0)
all_mean_data =np.mean(norm_tf_tuning,axis=0)
plt.errorbar(epoch_freqs,all_mean_data,all_yerr,label=('Mean +- std, N: %d' % np.shape(norm_tf_tuning)[0]))
plt.legend()
plt.xscale('log')
plt.ylabel('dF/F')
plt.xlabel('Temporal Frequency (Hz)')
plt.title('%s N: %d (%d)' % (toxin_name,selected_fly_num,selected_roi_num))


#%%
plt.figure(figsize=(10, 7))  
plt.title("Cluster TF tuning dendrograms")  
dend = shc.dendrogram(shc.linkage(norm_tf_tuning, method='ward'))  
#%%
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
prediction = cluster.fit_predict(norm_tf_tuning)  
plt.close('all')

plt.figure()
plt.subplot(122)
for cl_type in np.unique(prediction):
    curr_data =norm_tf_tuning[np.where(prediction==cl_type)[0],:]
    yerr = np.std(curr_data,axis=0)
    mean_data =np.mean(curr_data,axis=0)
    plt.errorbar(epoch_freqs,mean_data,yerr=yerr,
                 label=('Cluster %d, N: %d ROIs' % (cl_type,len(np.where(prediction==cl_type)[0]))))
    
plt.legend()
plt.title('Error:std')
plt.xscale('log')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')

plt.subplot(121)
plt.errorbar(epoch_freqs,all_mean_data,all_yerr,label=('Mean +- std, N: %d' % np.shape(tuning_curves)[0]))
plt.legend()
plt.xscale('log')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')

#%% Single fly cluster fit
for flyID,tunings in norm_tuning_cur_flies.items():
    prediction = cluster.fit_predict(tunings)  
    plt.figure()

    for cl_type in np.unique(prediction):
        curr_data =tunings[np.where(prediction==cl_type)[0],:]
        yerr = np.std(curr_data,axis=0)
        mean_data =np.mean(curr_data,axis=0)
        plt.errorbar(epoch_freqs,mean_data,yerr=yerr,
                     label=('Cluster %d, N: %d ROIs' % (cl_type,len(np.where(prediction==cl_type)[0]))))
        
    plt.legend()
    plt.title('Fly ID: %d' % flyID)
    plt.xscale('log')
    plt.ylabel('Normalized dF/F')
    plt.xlabel('Temporal Frequency (Hz)')
    
    

#%%
plt.close('all')
df_interest = big_df[big_df['toxin']=='No_toxin']
big_df_no_tox = big_df[big_df['toxin']=='No_toxin']
normalized_counts = (big_df.groupby(['toxin','flyID'])['BestFreq']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(1)
                     .reset_index()
                     .sort_values('BestFreq'))

#ax= sns.boxplot(x="BestFreq", y='percentage' ,hue = 'toxin',data=normalized_counts,notch=True);
ax= sns.boxplot(x="BestFreq",y='percentage', hue='toxin' ,data=normalized_counts);

ax.set(xlabel='Best Frequency (Hz)', ylabel='Fraction of ROIs')

#%%
ax= sns.countplot(x="BestFreq",data=big_df_no_tox);
ax.set(xlabel='Best Frequency (Hz)', ylabel='Fraction of ROIs')
#%%
fly_data = big_df.groupby(['flyID','toxin'],as_index=False).mean()

#%%
plt.close('all')
sns.boxplot( x='toxin' , y= np.abs(np.asarray(fly_data['DSI'])), data=fly_data,
               notch="True")
plt.ylabel('DSI')
#plt.yscale('log')
#%%
sns.boxplot( x='toxin' , y= np.abs(np.asarray(big_df['DSI'])), data=big_df,
               notch="True")
plt.ylabel('DSI')
#%%
plt.close('all')
sns.boxplot( x='toxin' , y= np.abs(np.asarray(big_df['SNR'])), data=big_df,
               notch="True")
plt.ylabel('SNR')
plt.yscale('log')
#%%
plt.close('all')
sns.boxplot( x='toxin' , y= np.abs(np.asarray(fly_data['SNR'])), data=fly_data,
               notch="True")
plt.ylabel('SNR')
#%%
cat1 = fly_data[fly_data['toxin']=='No_toxin']
cat2 = fly_data[fly_data['toxin']=='Octo_100uM']
ttest_ind(np.abs(cat1['SNR']), np.abs(cat2['SNR']))
#%% Getting rid of ROIs who never passed the threshold
control_big_df = big_df[big_df['toxin'] == 'No_toxin']
toxin_big_df = big_df[big_df['toxin'] == 'Agi_200nM']
if filtering_options['threshold_filter']:
    
    thresholdDf = control_big_df.filter(items=['ThresholdPass', 'ROI_ID']).\
    groupby(['ROI_ID'],as_index=False).max()
    passing_ROIs = thresholdDf[thresholdDf['ThresholdPass']==1] 
    control_big_df = control_big_df[control_big_df['ROI_ID'].isin(passing_ROIs['ROI_ID'].tolist())]
    
    thresholdDf = toxin_big_df.filter(items=['ThresholdPass', 'ROI_ID']).\
    groupby(['ROI_ID'],as_index=False).max()
    passing_ROIs = thresholdDf[thresholdDf['ThresholdPass']==1] 
    toxin_big_df = toxin_big_df[toxin_big_df['ROI_ID'].isin(passing_ROIs['ROI_ID'].tolist())]
    
    big_df = pd.concat([control_big_df,toxin_big_df])
    big_df=big_df.reset_index()
    

