#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 18:19:23 2018

@author: burakgur / Reading and organizing pickle format data also plotting raw 
traces
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw # Important for vector graphics figures
import numpy as np
import pandas as pd

functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
from post_analysis_functions import dataIntoPDFrame, getStimInfo, findDSI
from post_analysis_functions import dataReadOrganizeSelect , plotRawData
from post_analysis_functions import plot_by_cat_single_ROI,plot_byCategory
#%% Database directory
dataBaseDir = "/Users/burakgur/2p/Python_data/database"
metaDataBaseFile = os.path.join(dataBaseDir, 'metaDataBase.txt')
# Directory where the processed data are stored for analysis
processedDataStoreDir= \
"/Users/burakgur/2p/Python_data/database/processed_data"

saveFig = True
save_dir = "/Users/burakgur/Desktop/tempSave"
#%% Search database, extract and select datasets of interest

# Conditions to select from the database
conditions = {
 'stimName':'4Dir_DriftingSquare_0.1-30Hz_BG_4s_8_30deg_moving_BlankRand_4Dir.txt',
 'genotype':'T4T5-GMR64G09-lexA-rec',
 'comment-1' : 'responding'
             }
# Data variables to extract for further analysis
data_var_to_select = ['interpolatedAllRoi', 'interpolationRate','bgIndex', 
                      'stimInputData', 'flyID','bgSub','rawStimData',
                      'baselineEpochPresent','header','tags',
                      'baselineDurationBeforeEpoch', 'toxin']
# Data variables from the metadatabase that are going to be used in further analysis
# imageID has to be always there for extracting data
var_from_database_to_select = ['flyID', 'toxin', 'imageID']
selected_data = dataReadOrganizeSelect(conditions, data_var_to_select,
                                       var_from_database_to_select,
                                       dataBaseDir = dataBaseDir,
                                       processedDataStoreDir = 
                                       processedDataStoreDir, metaDataFile = 
                                       'metaDataBase.txt', data_ext = '*.pickle')
#%% Analysis start

# Understanding stimulus
representative_dataset = selected_data[selected_data.keys()[0]] 
stimulus_information = getStimInfo(representative_dataset)

#%% Organizing a pandas data frame with ROI data with desired filters

# Define desired filters
filtering_options = {
        'filter_string' : False,
        'filter_out_string':'',
        'filter_in_string':'Layer2',
        'threshold_filter' : True,
        'threshold_method' : 'STD',
        'threshold_std_multiplier' : 5
        
             }

#%%
big_df = dataIntoPDFrame(selected_data, stimulus_information, filtering_options)
plt.close('all')

#%% Arranging a figure title
if filtering_options['filter_string']:
    if filtering_options['filter_in_string']:
        title_string = 'ROIs of %s' % filtering_options['filter_in_string']
    else:
        title_string = 'All ROIs'
    if filtering_options['filter_out_string']:
        title_string = '%s except %s' % (title_string,
                                          filtering_options['filter_out_string'])
else:
    title_string = 'All ROIs'
    
if filtering_options['threshold_filter']:
    title_string = '%s, threshold: %d x %s' % (title_string, filtering_options['threshold_std_multiplier'],
                                               filtering_options['threshold_method'])
else:
    title_string = '%s, no threshold' % (title_string)
    
#%% Plot the raw data
plotRawData(selected_data, filtering_options, conditions,title_string,
            stimulus_information, frames_to_plot = 1000 , save_dir = save_dir)
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
    

