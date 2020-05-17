import os
import cPickle


import ROI_mod
from core_functions import saveWorkspace
from cluster_analysis_functions_v2 import get_stim_xml_params
#%% Setting the directories
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/selected')
stimInputDir = os.path.join(initialDirectory,'stimulus_types')
saveOutputDir = os.path.join(initialDirectory,'analyzed_data',
                             '191113_T4T5_all_re_analyzed_1')
summary_save_dir = os.path.join(initialDirectory,
                              'selected_experiments/results')
stimInputDir = os.path.join(initialDirectory,'stimulus_types')

#%% Load datasets and desired variables

datasets_to_load = ['190912bg_fly2-TSeries-09122019-1024-004_sima_STICA.pickle']

# combined_df = pd.DataFrame(columns = properties)
for dataset in datasets_to_load:
    # Load dataset
    load_path = os.path.join(saveOutputDir, dataset)
    load_path = open(load_path, 'rb')

    # Load all the necessary variables for saving them later on again
    workspace = cPickle.load(load_path)
    final_rois = workspace['final_rois']
    cluster_analysis_params = workspace['cluster_analysis_params']
    current_exp_ID = workspace['current_exp_ID']
    current_movie_ID = workspace['current_movie_ID']
    stimType = workspace['stimType']
    extraction_type = workspace['extraction_type']

    # Add a property
    dataDir = os.path.join(alignedDataDir, current_exp_ID, current_movie_ID[current_movie_ID.find('-')+1:],'motCorr.sima')
    t_series_path = os.path.dirname(dataDir)
    (stimulus_information, trialCoor, frameRate, depth, x_size, y_size, pixelArea,
     stimType, stimOutFile, epochCount, stimName, layerPosition,
     stimInputFile, xmlFile, trialCount, isRandom, stimInputData,
     rawStimData) = get_stim_xml_params(t_series_path, stimInputDir)

    for roi in final_rois:
        roi.imaging_info = {'frame_rate' : frameRate, 'pix_size' : x_size}


    os.chdir('/Users/burakgur/Documents/GitHub/python_lab/pyCharm_2p_analysis')
    varDict = locals()
    saveWorkspace(outDir=saveOutputDir,
                  baseName=dataset, varDict=varDict, varFile='varSave_cluster_v2.txt',
                  extension='')

    print('%s saved...' % dataset)