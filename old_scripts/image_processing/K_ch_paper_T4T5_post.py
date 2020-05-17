#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:07:35 2019

@author: burakgur
"""

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
from scipy.stats import ttest_ind, ttest_rel

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


#%%White plots
sns.set(style="ticks", context="talk",rc={"font.size":14,"axes.titlesize":12,
                                          "axes.labelsize":14,'xtick.labelsize':12,
                                          'ytick.labelsize' : 12,
                                          'legend.fontsize':12})
plt.style.use("default")
sns.set_palette("Set2")
colors = sns.color_palette(palette='Set2', n_colors=2)

#%%
plt.hist(big_df[big_df['toxin']=="No_toxin"]['BestFreq'])

#%%
plt.close('all')

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
#%%
cat1 = normalized_counts[(normalized_counts['toxin']=='No_toxin') & (normalized_counts['BestFreq']==1.5)]
cat2 = normalized_counts[(normalized_counts['toxin']=='Agi_200nM') & (normalized_counts['BestFreq']==1.5)]
ttest_rel(cat1['percentage'], cat2['percentage'])