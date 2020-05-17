#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:50:37 2018

@author: burakgur
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import math
from collections import OrderedDict
from core_functions import searchDatabase, retrieveData, selectData

def setRCParams():
    """ Sets the desired parameters for plotting.

    Parameters
    ==========
   

    Returns
    =======
  
    """
    
    matplotlib.rc('font', family='sans-serif', serif='Helvetica')
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    matplotlib.rc('axes', labelsize=8)
    
    return None
    
def dataReadOrganizeSelect(conditions, data_var_to_select,
                           var_from_database_to_select,
                           dataBaseDir =\
                           "/Users/burakgur/2p/Python_data/database",
                           processedDataStoreDir= \
                           "/Users/burakgur/2p/Python_data/ \
                           database/processed_data", metaDataFile = 
                           'metaDataBase.txt', data_ext = '*.pickle'):
    """ Processes the data and saves the necessary variables
    
    Parameters
    ==========
    dataBaseDir : str
    
        Path of the T-series that includes the motion correction directory
        along with stimulus output and xml file.
    
    processedDataStoreDir : str
    
        Path of the folder where stimulus input files are located. These files
        contain information about all the stimuli used in the experiments.
    
    metaDataFile : str
    
        Path of the file where the meta data is contained
    
    data_ext : str
    
        Path of the folder where the data output files will be saved
    
    conditions : dict
    
        A dictionary with keys (conditions) that are the columns of meta database 
        file and values that are the conditions of interest.
    
    data_var_to_select : list
    
        A list of strings that indicate the data variables to be selected for 
        further analysis.
        
    var_from_database_to_select : list
    
        A list of strings that indicate the data variables from the meta database
        to be kept for further analysis.
    
    
    
    Returns
    =======
    selected_data: dict
    A dictionary containing the selected variables of each dataset with 
    unique keys.
    """
    metaDataBaseFile = os.path.join(dataBaseDir, metaDataFile)
    data_to_select = searchDatabase(metaDataBaseFile, conditions,
                                    var_from_database_to_select)
    all_data = retrieveData(processedDataStoreDir, data_to_select,
                            var_from_database_to_select, data_ext = data_ext)
    selected_data = selectData(all_data, data_var_to_select)
        
    return selected_data

def getStimInfo(representative_dataset):
    """ Organizes the dataset into a pandas dataframe.

    Parameters
    ==========
    representative_dataset : dict
        A dictionary containing the selected variables of each dataset with 
        unique keys.
    
    Returns
    =======
    
    stimulus_information : dict
        A dictionary containing the stimulus information.
    
  
    """
    stimulus_information ={}
    stimulus_data = representative_dataset['stimInputData']
    stimulus_information['baseline_exist'] = \
        representative_dataset['baselineEpochPresent']
    stimulus_information['baseline_duration_before_epoch'] = \
        representative_dataset['baselineDurationBeforeEpoch']
    stimulus_information['epoch_dir'] = \
        np.asfarray(stimulus_data['Stimulus.stimrot.mean'])
    epoch_speeds = np.asfarray(stimulus_data['Stimulus.stimtrans.mean'])
    stimulus_information['epoch_frequency'] = \
        epoch_speeds/np.asfarray(stimulus_data['Stimulus.spacing'])
    stimulus_information['baseline_duration'] = \
        np.asfarray(stimulus_data['Stimulus.duration'][0])
    stimulus_information['epochs_duration'] =\
         np.asfarray(stimulus_data['Stimulus.duration'])
         
    return stimulus_information
    
def applyThreshold(nullResponse, realResponse, thresholdType = 'STD',
                   stdMultiplier = 3):
    """ Applys a threshold comparing the null and real responses and returning a 
    boolean if threshold is passed or not.

    Parameters
    ==========
    nullResponse : numpy.ndarray
    
        The signal which the statistics of it will be compared with the 
        real response signal. Usually a background adapted epoch.
        
    realResponse : numpy.ndarray
    
        The signal to be determined as different than null response or not.
        
    thresholdType : str
        Default: 'STD'
        
        Method of thresholding. 
            'STD'  ->   Determining the signal as "responding" if 
                        max(realResponse) > mean(nullResponse) * stdMultiplier *
                        std(nullResponse)
    stdMultiplier : float
        Default: 3
        
        see 'STD' method for where this number is used
    
    Returns
    =======
    thresholdPass : bool
        
        A boolean if the realResponse passes the threshold defined by the user.
  
    """
    if thresholdType == 'STD':
        threshold_value = np.absolute(nullResponse.mean()) + \
            (stdMultiplier * nullResponse.std())
        thresholdPass = (threshold_value < realResponse.max())
    
    return thresholdPass

def applyROITypeFilter(filtering_options, ROItag):
    """ Applys a filter for ROI types.

    Parameters
    ==========
    filtering_options : dict
        A dictionary containing the filtering options.
    
    Returns
    =======
    filterPass : bool
        
        A boolean if this ROI tag passes the filtering.
  
    """
    if not filtering_options['filter_in_string']: #If filter is not present
        filterPass_in = True
    else:
        filterPass_in = (filtering_options['filter_in_string'] in ROItag)
    if not filtering_options['filter_out_string']: #If filter is not present
        filterPass_out = False
    else:
        filterPass_out = (filtering_options['filter_out_string'] in ROItag)
    
    filterPass = filterPass_in & (not filterPass_out)
    return filterPass

def dataIntoPDFrame(selected_data, stimulus_information, filtering_options):
    """ Organizes the dataset into a pandas dataframe.

    Parameters
    ==========
    selected_data : dict
        A dictionary containing the selected variables of each dataset with 
        unique keys.
    
    stimulus_information : dict
        A dictionary containing the stimulus information of the current dataset.
        
    filtering_options : dict
        A dictionary containing the filtering options.
  
    Returns
    =======
    
    all_data_frame : Pandas dataframe
    
  
    """
    all_data_frame = pd.DataFrame()
    trace_length = []
    data_frame_location = 1
    total_data_point_counter = 1
    for datasetKey, value in selected_data.iteritems():
        current_dataset = selected_data[datasetKey]
        interpolation_rate = np.asfarray(current_dataset['interpolationRate'])
        all_responses = current_dataset['interpolatedAllRoi']
        flyID = np.asarray(current_dataset['flyID']) # get the flyID
        tags = current_dataset['tags'] # ROI labels
        roi_nums = current_dataset['header'] # ROI numbers
        toxin = current_dataset['toxin'] # Information about toxin
        for epoch, epoch_responses in all_responses.iteritems():
            if stimulus_information['baseline_exist']:
                # Determine start time as F0 calculation start
                startTime = int(interpolation_rate * \
                                (stimulus_information['baseline_duration']-\
                                 stimulus_information['baseline_duration_before_epoch']))
                # End at the end of the epoch
                endTime = int((stimulus_information['baseline_duration'] + \
                               stimulus_information['epochs_duration'][epoch]) * \
                interpolation_rate)
            else:
                # If there's no baseline epoch then take all the trace
                startTime = 0
                endTime = -1
            for iROI, response in enumerate(epoch_responses):
                total_data_point_counter += 1
                curr_ROI_data = {}
                if (tags[iROI] == 'bg'): 
                    continue      
                elif np.isnan(response).any(0):
                    print('Fly ID: %d might have a label order problem, or an ROI with nans'\
                          % flyID)
                    continue
                
                curr_response = response[startTime:endTime]
                    
                if filtering_options['threshold_filter']:
                    t_type = filtering_options['threshold_method']
                    t_multiplier = filtering_options['threshold_std_multiplier']
                    if stimulus_information['baseline_exist']:
                       thresholdPass = applyThreshold(response[:startTime], 
                                                   curr_response, 
                                                   thresholdType = t_type,
                                                   stdMultiplier = t_multiplier)
                    else:
                       thresholdPass = applyThreshold(curr_response, 
                                                   curr_response, 
                                                   thresholdType = t_type,
                                                   stdMultiplier = t_multiplier)
                    if not thresholdPass:
                        curr_ROI_data['ThresholdPass'] = 0
                    else:
                        curr_ROI_data['ThresholdPass'] = 1
                    
                # ROI type name filtering
                if filtering_options['filter_string']:
                   stringFilterPass = applyROITypeFilter(filtering_options, tags[iROI])
                   
                   if not stringFilterPass:
                       continue
                    
                trace_length.append(np.shape(curr_response))
                curr_ROI_data['Max_resp'] = np.max(curr_response)
                curr_ROI_data['flyID'] = flyID 
                curr_ROI_data['Direction']  = stimulus_information['epoch_dir'][epoch]
                curr_ROI_data['Freq']  = stimulus_information['epoch_frequency'][epoch]
                curr_ROI_data['ROI_type']  = tags[iROI] 
                # Assigning a unique ID to the ROI which is consistent for every 
                # epoch and possibly stimulus
                curr_ROI_data['ROI_ID'] = 'Fly_%d_ROI_%s' % (flyID, roi_nums[iROI])
                curr_ROI_data['toxin'] = toxin
                curr_ROI_data['trace'] = [curr_response] 
                currDF = pd.DataFrame(curr_ROI_data,index=[iROI])
                
                all_data_frame = all_data_frame.append(currDF,ignore_index=True)
                data_frame_location += 1
    
    print('Total data points: %d\n' %  (total_data_point_counter-1))
    print('Selected data points: %d' % data_frame_location)             
    return all_data_frame

def dataIntoPDFrame_cluster(selected_data):
    """ Organizes the dataset into a pandas dataframe.

    Parameters
    ==========
    selected_data : dict
        A dictionary containing the selected variables of each dataset with 
        unique keys.
    
    stimulus_information : dict
        A dictionary containing the stimulus information of the current dataset.
        
    filtering_options : dict
        A dictionary containing the filtering options.
  
    Returns
    =======
    
    all_data_frame : Pandas dataframe
    
  
    """
    all_data_frame = pd.DataFrame()
    tf_tuning_data_frame = pd.DataFrame()
    data_frame_location = 1
    total_data_point_counter = 1
    for datasetKey, value in selected_data.iteritems():
        current_dataset = selected_data[datasetKey]
        flyID = np.asarray(current_dataset['flyID']) # get the flyID
        cluster_layers = current_dataset['cluster_layer_information'] # Cluster layers
        cluster_nums = current_dataset['cluster_numbers'] # ROI numbers
        toxin = current_dataset['toxin'] # Information about toxin              
        stim_info = current_dataset['stimulus_information']
        for iCluster in range(len(cluster_nums)):
            total_data_point_counter = total_data_point_counter+1
            curr_ROI_data = {}
             
            curr_ROI_data['Z_depth'] = current_dataset['layerPosition'][2]
            curr_ROI_data['flyID'] = flyID 
            curr_ROI_data['DSI']  = current_dataset['selected_clusters_DSI'][iCluster]
            curr_ROI_data['BestFreq']  = current_dataset['selected_clusters_TF'][iCluster]
            curr_ROI_data['SNR']  = current_dataset['selected_clusters_SNR'][iCluster]
            curr_ROI_data['Cluster_layer']  = cluster_layers[iCluster] 
            curr_ROI_data['ROI_ID'] = 'Fly_%d_ROI_%s' % (flyID, cluster_nums[iCluster])
            curr_ROI_data['toxin'] = toxin
            curr_ROI_data['TF_tuning'] = [current_dataset['selected_cluster_TF_tuning_no_edge'][iCluster,:]]
            currDF = pd.DataFrame(curr_ROI_data,index=[iCluster])
            
            uniq_freqs = np.unique(stim_info['epoch_frequency'])[1:]
            tf_tunings = current_dataset['selected_cluster_TF_tuning_no_edge']
            for iEpoch in range(np.shape(tf_tunings)[1]):
                curr_epoch_data = {}
                curr_epoch_data['Z_depth'] = current_dataset['layerPosition'][2]
                curr_epoch_data['flyID'] = flyID 
                curr_epoch_data['ROI_ID'] = 'Fly_%d_ROI_%s' % (flyID, cluster_nums[iCluster])
                curr_epoch_data['toxin'] = toxin
                curr_epoch_data['Cluster_layer']  = cluster_layers[iCluster] 
                curr_epoch_data['Response'] = tf_tunings[iCluster,iEpoch]
                curr_epoch_data['Freq'] = uniq_freqs[iEpoch]
                
                curr_epochDF = pd.DataFrame(curr_epoch_data,index=[iEpoch])
                tf_tuning_data_frame=tf_tuning_data_frame.append(curr_epochDF,ignore_index=True)
#                
            all_data_frame = all_data_frame.append(currDF,ignore_index=True)
            data_frame_location += 1
    
    print('Total data points: %d\n' %  (total_data_point_counter-1))
    print('Selected data points: %d' % data_frame_location)             
    return all_data_frame, tf_tuning_data_frame

def findDSI(data_frame):
    """ Finds the DSI according to (PD-ND)/(PD+ND)

    Parameters
    ==========
    data_frame : pandas df
        Pandas dataframe that contains the max responses and directions together 
        with ROI IDs.
    
    Returns
    =======
    dsi_df : pandas df
        A df with ROI IDs and the corresponding DSI.
  
    """
    dsi_dict = {'ROI_ID': [], 'DSI': []}

    unique_control_rois = np.unique(data_frame['ROI_ID']).tolist()
    for roi_id in unique_control_rois:
        curr_df = data_frame[data_frame['ROI_ID']==roi_id]
        max_idx = curr_df['Max_resp'].idxmax()
        max_dir = curr_df['Direction'].loc[max_idx]
        opposite_dir = (max_dir + 180) % 360
        max_resp =  curr_df['Max_resp'].max()
        opposite_resp = curr_df[curr_df['Direction']==opposite_dir]['Max_resp'].max()
        curr_DSI = (max_resp-opposite_resp)/(max_resp+opposite_resp)
        dsi_dict['ROI_ID'].append(roi_id)
        dsi_dict['DSI'].append(curr_DSI)
    
    dsi_df = pd.DataFrame.from_dict(dsi_dict)  
    return dsi_df

    

def plot_byCategory(big_dataframe, signal_variable_name, filter_cat, subcategories_to_plot, 
                    conditions, stim_start,title_string, save_dir = False,
                    plot_end_frame = -1):
    """ Plots ROIs

    Parameters
    ==========
    big_dataframe : Pandas dataframe
        Data in the format of pandas dataframe.
        
    signal_variable_name : str
        Within the dataframe, the name of the column where signals are stored.
        
    filter_cat : str
        Within the dataframe, the name of the column where categories that will be 
        seperated into different figures are located.
        
    subcategories_to_plot : str
        Within the dataframe, the name of the column where categories of interest
        are stored.
        
    conditions : dict
        A dictionary containing the genotype name and name of the stimulus.
    
    stim_start : float
        The point where stimulus starts (in seconds)
        
    title_string : str
        The title string containing the information about the dataset.
        
    save_dir : str | bool
        Default: False
        
        Name of the figure save directory. If nothing is given its value is False
        and nothing is saved.
        
    plot_end_frame : int
        Default: -1 (end of the trace)
        
        Determines how long the plots will be
    Returns
    =======
  
    """
    
    unique_filter_categories = np.unique(big_dataframe[filter_cat]).tolist()
    
    for iCat, filter_category_name in enumerate(unique_filter_categories):
        # Filtering the data frame by the category name
        data_frame = big_dataframe[big_dataframe[filter_cat] == filter_category_name]
        
        # Finding the unique subcategories to plot
        unique_categories = np.unique(data_frame[subcategories_to_plot]).tolist()
        
        # Defining figure properties
        subPlotNumbers = len(unique_categories)
        nrows = int(round(float(subPlotNumbers)/float(2)))
        if nrows == 1:
            ncols = 1
        else:
            ncols = 2
            
        # Creating a figure with desired subplots
        fig, ax = plt.subplots(nrows, ncols, sharey=True,figsize=(7, 7), 
                               facecolor='w', edgecolor='k')
        fig.suptitle('%s -- %s  %s, \n stim: %s' % (title_string,'Responses',filter_category_name,
                                          conditions['stimName']),
                     fontsize=8)
        ax = ax.flatten()
        
        # Some colors
        cmaps = OrderedDict()
        cmaps['Sequential'] = [
                    'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        
        for iSubplot, category_name in enumerate(unique_categories):
            category_mask = data_frame[subcategories_to_plot] == category_name
            cat_signals = data_frame[category_mask]
            cat_signals.reset_index
            current_cmap = plt.get_cmap(cmaps['Sequential'][iSubplot])
            if plot_end_frame != -1:
                all_resp = np.empty((0,plot_end_frame), float)
            else:
                all_resp = np.empty((0,len(cat_signals.iloc[0][signal_variable_name])), float)
            
            for index, row in cat_signals.iterrows():
                curr_response = row[signal_variable_name]
                curr_response = curr_response[:plot_end_frame]
                ax[iSubplot].plot(curr_response,color = current_cmap(index),
                                  linewidth = 0.5,alpha=0.2) 
                all_resp = np.vstack((all_resp, curr_response))
            
            mean_response = all_resp.mean(axis=0)
            std_resp  = all_resp.std(axis=0)
            sample_size = np.shape(all_resp)[0]
            
            ax[iSubplot].plot(mean_response,color = 'k',label='Mean',
                                  linewidth = 2,alpha=1) 
            line = ax[iSubplot].lines[-1]
            ax[iSubplot].fill_between(line.get_xdata(), mean_response-std_resp/2, 
                                      mean_response + std_resp/2,
                                      alpha= 0.8,color = 'k',label='std')
            ax[iSubplot].set_title('%s N: %d' % (category_name,sample_size))
            ax[iSubplot].set_ylabel('dF/F')
            ax[iSubplot].set_xlabel('Frames (10Hz)')
            ax[iSubplot].axvline(x=round(stim_start*10),color = 'r',linestyle='--',
                                  label='stimulus start')
            if stim_start == 50: # Means that it is 5sFFF
                ax[iSubplot].axvspan(0, round(stim_start*10), facecolor='k', alpha=0.4)
            ax[iSubplot].legend(prop={'size': 6})
            plt.tight_layout()
            sns.despine()
            
        if not isinstance(save_dir, (bool)):
            # Saving figure
            save_name = 'Trace-%s-%s-%s' % (conditions['genotype'], conditions['stimName'],
                                    filter_category_name)
            os.chdir(save_dir)
            plt.savefig('%s.png'% save_name, bbox_inches='tight')
            print('Figure saved')
            plt.close('all')
    
    return  None
                
        
    

def plot_by_cat_single_ROI(big_dataframe, signal_variable_name, filter_cat, 
                           subcategories_to_plot, subsub_cat_to_plot,
                    conditions, stim_start,title_string, save_dir = False,
                    toxin='No_toxin'):
    """ Plots single ROIs with subcategories on top

    Parameters
    ==========
    big_dataframe : Pandas dataframe
        Data in the format of pandas dataframe.
        
    signal_variable_name : str
        Within the dataframe, the name of the column where signals are stored.
        
    filter_cat : str
        Within the dataframe, the name of the column where categories that will be 
        seperated into different figures are located.
        
    subcategories_to_plot : str
        Within the dataframe, the name of the column where subcategories of interest
        are stored.
        
    subsub_cat_to_plot : str
        Within the dataframe, the name of the column where subsubcategories of interest
        are stored.
        
    conditions : dict
        A dictionary containing the genotype name and name of the stimulus.
    
    stim_start : float
        The point where stimulus starts (in seconds)
        
    title_string : str
        The title string containing the information about the dataset.
        
    save_dir : str | bool
        Default: False
        
        Name of the figure save directory. If nothing is given its value is False
        and nothing is saved.
    Returns
    =======
  
    """
    
    unique_filter_categories = np.unique(big_dataframe[filter_cat]).tolist()
    
    for iCat, filter_category_name in enumerate(unique_filter_categories):
        # Filtering the data frame by the category name
        data_frame = big_dataframe[big_dataframe[filter_cat] == filter_category_name]
        
        # Finding the unique subcategories to plot
        uniquesub_categories = np.unique(data_frame[subcategories_to_plot]).tolist()
        
        # Defining figure properties
        subPlotNumbers = len(uniquesub_categories)
        sp = math.sqrt(subPlotNumbers)
        if math.floor(sp) < subPlotNumbers/sp:
            nrows = int(round(subPlotNumbers/sp)+1)
        else:
            nrows = int(math.floor(sp))
        ncols = int(math.floor(sp))
            
        # Creating a figure with desired subplots
        fig, ax = plt.subplots(nrows, ncols, sharey=True,figsize=(14, 14), 
                               facecolor='w', edgecolor='k')
        fig.suptitle('%s -- %s  %s, \n stim: %s' % (title_string,'Responses',filter_category_name,
                                          conditions['stimName']),
                     fontsize=8)
        ax = ax.flatten()
        
        
        for iSubplot, subcategory_name in enumerate(uniquesub_categories):
            subcategory_mask = data_frame[subcategories_to_plot] == subcategory_name
            subcat_frame = data_frame[subcategory_mask]
            subcat_frame.reset_index
            
            unique_subsubcategories = np.unique(data_frame[subsub_cat_to_plot]).tolist()
            cmap = plt.matplotlib.cm.get_cmap('Set1')
            
            for iColor, subsubcategory_name in enumerate(unique_subsubcategories):
                
                
                subsubcategory_mask = subcat_frame[subsub_cat_to_plot] == subsubcategory_name
                cat_signals = subcat_frame[subsubcategory_mask]
                curr_response = cat_signals.iloc[0][signal_variable_name]
                curr_color = cmap(iColor)
                ax[iSubplot].plot(curr_response,linewidth = 1,alpha=0.5,
                                  label = subsubcategory_name, color=curr_color) 
                ax[iSubplot].scatter(curr_response.argmax(),np.max(curr_response),
                                  color=curr_color,marker='x') 
              
            ax[iSubplot].set_ylabel('dF/F')
            ax[iSubplot].set_xlabel('Frames (10Hz)')
            ax[iSubplot].set_title(subcategory_name)
            ax[iSubplot].axvline(x=round(stim_start*10),color = 'r',linestyle='--',
                                  label='stimulus start')
            ax[iSubplot].legend(prop={'size': 6})
            plt.tight_layout()
            sns.despine()
            
        if not isinstance(save_dir, (bool)):
            # Saving figure
            save_name = 'STrace-%s-%s-%s-%s-%s' % (conditions['genotype'], conditions['stimName'],
                                    title_string,filter_category_name,toxin)
            os.chdir(save_dir)
            plt.savefig('%s.png'% save_name, bbox_inches='tight')
            print('Figure saved')
            plt.close('all')
    
    return  None

def plotRawData(selected_data, filtering_options, conditions, title_string,
                stimulus_information,frames_to_plot = 4000, save_dir = False, 
                plot_randomizer_ROInum = 'max'):
    """ Plots raw data and stimulus considering the user defined parameters

    Parameters
    ==========
    selected_data : dict
        A dictionary containing the selected variables of each dataset with 
        unique keys.
        
    filtering_options : dict
        A dictionary containing the filtering options.
    
    conditions : dict
        A dictionary containing the genotype name and name of the stimulus.
        
    title_string : str
        Title for the figure informing about the filters used.
    
    stimulus_information : dict
        A dictionary containing the stimulus information of the current dataset.
        
    frames_to_plot : int
        Default: 4000
        
        Defines the frame to stop the raw trace plots.
        
    save_dir : str | bool
        Default: False
        
        Name of the figure save directory. If nothing is given its value is False
        and nothing is saved.
        
    plot_randomizer_ROInum : str | int
        Default: 'max'
        
        Determines random plotting of ROIs in case the user wants to view a few
        traces. When 'max', plots all ROIs.
        
  
    Returns
    =======

    
  
    """
   
    for datasetKey, value in selected_data.iteritems():
        current_dataset = selected_data[datasetKey]
        
        # Process the raw signals
        raw_responses = current_dataset['bgSub']
        raw_mean = np.mean(raw_responses,axis=0)
        dF_resp = (raw_responses - raw_mean) / raw_mean
        
        
        # Raw stimulus data
        rawStimData = current_dataset['rawStimData']
        stim_frames = rawStimData[:,7]  # Frame information
        stim_vals = rawStimData[:,3] # Stimulus value
        uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
        stim_vals = stim_vals[uniq_frame_id]
        stim_vals = stim_vals[:dF_resp.shape[0]]
        stim_df_unscaled = stim_vals
        # Make normalized values of stimulus values for plotting
        stim_df = (stim_df_unscaled/np.max(np.unique(stim_vals)))*5 
        # Text array to make stimulus labels
        epoch_dirs = stimulus_information['epoch_dir']
        epoch_freqs = stimulus_information['epoch_frequency']
        stim_epoch_names = list()
        for iEpoch in np.unique(stim_vals):
            epoch_name = u'%d°\n%.2fHz' % (epoch_dirs[int(iEpoch)],
                                               epoch_freqs[int(iEpoch)])
            stim_epoch_names.append(epoch_name)
        epoch_name_x_vals = np.unique(stim_vals,return_index=True)[1]
        
        flyID = np.asarray(current_dataset['flyID']) # get the flyID
        tags = current_dataset['tags'] # ROI labels
        toxin = current_dataset['toxin'] # Information about toxin
        
        plot_array = np.empty((0,frames_to_plot),float)
        if plot_randomizer_ROInum != 'max':
            N = len(tags)
            p = float(plot_randomizer_ROInum*1.1)/float(N)
            plot_choice = np.random.choice(a=[False, True], size=( N), p=[p, 1-p])  
        else:
            plot_choice = np.ones(len(tags), dtype=bool)
        
        # Selection of ROIs for plotting
        for iROI, ROI_type in enumerate(tags):
            if not plot_choice[iROI]:
                continue
            curr_response = dF_resp[:frames_to_plot,iROI]
            
            if (tags[iROI] == 'bg'): 
                continue      
            elif np.isnan(curr_response).any(0):
                print('Fly ID: %d might have a label order problem, or an ROI with nans'\
                      % flyID)
                continue
            
            
                
            if filtering_options['threshold_filter']:
                t_type = filtering_options['threshold_method']
                t_multiplier = filtering_options['threshold_std_multiplier']
                thresholdPass = applyThreshold(curr_response, 
                                               curr_response, 
                                               thresholdType = t_type,
                                               stdMultiplier = t_multiplier)
                if not thresholdPass:
                    continue
               
                
            # ROI type name filtering
            if filtering_options['filter_string']:
               stringFilterPass = applyROITypeFilter(filtering_options, tags[iROI])
               
               if not stringFilterPass:
                   continue
                
            plot_array = np.vstack((plot_array, curr_response))
            
        if plot_array.size == 0:
            continue
            
        plot_array_shifted = np.transpose(plot_array) + np.linspace(np.shape(plot_array)[0]*2,0,
                                                num=np.shape(plot_array)[0],dtype=int)
        fig = plt.figure(datasetKey,figsize=(14, 12), facecolor='w', edgecolor='k')
        flyName = 'Fly %d, %s, %s' % (flyID,toxin,title_string)
        fig.suptitle(flyName)
     
        plt.plot(plot_array_shifted,figure = fig)
        plt.plot(stim_df[:frames_to_plot], dashes=[6, 2],color='k',figure=fig)
        uniq_scaled_stim = np.unique(stim_df) + 0.5
        for iEpoch , epoch_name in enumerate(stim_epoch_names):
            if frames_to_plot > epoch_name_x_vals[iEpoch]:
                plt.text(epoch_name_x_vals[iEpoch],uniq_scaled_stim[iEpoch],
                         epoch_name,figure=fig, fontweight='bold')
            
        if not isinstance(save_dir, (bool)):
            # Saving figure
            save_name = 'RawTrace-%s-%s-%s-%s' % (conditions['genotype'], conditions['stimName'],
                                    title_string,flyName)
            os.chdir(save_dir)
            plt.savefig('%s.png'% save_name, bbox_inches='tight')
            print('Figure saved')
            plt.close('all')
        
       
    
       
    return None
