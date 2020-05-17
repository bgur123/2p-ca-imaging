"""
Created on Fri Nov 15 15:09:38 2019

@author: burakgur
"""

import cPickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

import ROI_mod
from core_functions import saveWorkspace
#%% Setting the directories
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/selected')
stimInputDir = os.path.join(initialDirectory,'stimulus_types')
saveOutputDir = os.path.join(initialDirectory,'analyzed_data',
                             '191113_T4T5_all_re_analyzed_1')
summary_save_dir = os.path.join(initialDirectory,
                              'selected_experiments/results')

# Plotting parameters
plt.style.use("dark_background")
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams["axes.titlesize"] = 'small'
plt.rcParams["legend.fontsize"] = 'small'

#%% Load datasets and desired variables

exp_t = '1st_exp'
datasets_to_load = ['181011bg_fly1-TSeries-003_sima_STICA.pickle',
                    '181019bg_fly2-TSeries-003_sima_STICA.pickle',
                    '181022bg_fly1-T-Series-003_sima_STICA.pickle',
                    '181022bg_fly2-T-Series-003_sima_STICA.pickle',
                    '181023bg_fly1-TSeries-10232018-1454-003_sima_STICA.pickle']
final_rois_all =[]
flyIDs = []
tunings =[]
properties = ['DSI', 'SNR', 'reliability', 'BF','z_depth','exp_ID','uniq_id','distance']
rois_dict = {}
#combined_df = pd.DataFrame(columns = properties)
for dataset in datasets_to_load:
    
    load_path = os.path.join(saveOutputDir,dataset)
    load_path = open(load_path, 'rb')
    #
    workspace = cPickle.load(load_path)
    final_rois = workspace['final_rois']
    cluster_analysis_params = workspace['cluster_analysis_params']
    current_exp_ID = workspace['current_exp_ID']
    current_movie_ID = workspace['current_movie_ID']
    stimType = workspace['stimType']
    extraction_type = workspace['extraction_type']
#    Genotype = workspace['Genotype']
#    Age = workspace['Age']

    #
    final_rois_all.append(workspace['final_rois'])
    flyIDs.append(list(map(lambda roi : roi.exp_ID, workspace['final_rois'])))
    tunings.append(np.squeeze(list(map(lambda roi : roi.TF_curve_resp, workspace['final_rois']))))
    a ,b = ROI_mod.calculate_distance_from_region(final_rois)
#    rois_dict = ROI_mod.data_to_list(workspace['final_rois'], properties)
#    curr_df = pd.DataFrame.from_dict(rois_dict)
#    combined_df = combined_df.append(curr_df, ignore_index=True,sort=False)

    os.chdir(functionPath)
    varDict = locals()
    saveWorkspace(outDir= saveOutputDir,
                  baseName=dataset, varDict= varDict, varFile='varSave_cluster_v2.txt',
                      extension='')

    print('%s saved...' % dataset)

#%% Tuning curves
tuning_curves = np.concatenate((tunings[:]))
all_mean_data = np.mean(tuning_curves,axis=0)
all_yerr = np.std(tuning_curves,axis=0)
epoch_freqs = final_rois[0][0].TF_curve_stim
fly_IDs = np.concatenate((flyIDs[:]))
unique_flies = np.unique(fly_IDs)
norm_tuning_cur_flies = {}
norm_tf_tuning = normalize(tuning_curves,axis=1,norm='max')
for fly_num in unique_flies:
    curr_ROI_mask = (fly_IDs == fly_num)
    norm_tuning_cur_flies[fly_num]= norm_tf_tuning[curr_ROI_mask,:]

#%% Clustering the tuning curves
plt.close('all')
plt.figure(1,figsize=(10, 7))  
plt.title("Cluster TF tuning dendrograms")  
dend = shc.dendrogram(shc.linkage(norm_tf_tuning, method='ward'))  
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
prediction = cluster.fit_predict(norm_tf_tuning)  
#plt.close('all')

plt.figure(2)
plt.subplot(122)
for cl_type in np.unique(prediction):
    curr_data =norm_tf_tuning[np.where(prediction==cl_type)[0],:]
    yerr = np.std(curr_data,axis=0)
    mean_data =np.mean(curr_data,axis=0)
    plt.errorbar(epoch_freqs,mean_data,yerr=yerr, lw =2,
                 label=('Cluster %d, N: %d ROIs' % (cl_type,len(np.where(prediction==cl_type)[0]))))
    
plt.legend()
plt.title('Predicted clusters')
plt.xscale('log')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')

plt.subplot(121)
plt.errorbar(epoch_freqs,all_mean_data,all_yerr,color=plt.cm.Dark2(3),
             label=('Mean +- std, N: %d' % np.shape(tuning_curves)[0]),lw=2)
plt.legend()
plt.xscale('log')
plt.title('Mean responses')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')
plt.show()
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
    plt.title('Fly ID: %s' % flyID)
    plt.xscale('log')
    plt.ylabel('Normalized dF/F')
    plt.xlabel('Temporal Frequency (Hz)')
    
    