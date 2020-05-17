#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:49:14 2019

@author: burakgur
"""

import os
import cv2
import numpy as np
import glob
import math
import copy
import seaborn as sns
from skimage import io
from skimage import filters
from skimage import exposure
from skimage import img_as_ubyte, img_as_float
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

functionPath = '/Users/burakgur/Documents/GitHub/python_lab/image_processing'
os.chdir(functionPath)
from core_functions import readStimOut, readStimInformation, getEpochCount
from core_functions import divideEpochs
from xmlUtilities import getFramePeriod, getLayerPosition


#%%
alignedDataDir = '/Users/burakgur/2p/Python_data/motion_corrected/chosen_alignment'
stimInputDir = "/Users/burakgur/2p/Python_data/stimulus_types"

# Directory where the processed data will be saved
saveOutputDir = "/Users/burakgur/2p/Python_data/analyzed_data"

# Database directory
dataBaseDir = "/Users/burakgur/2p/Python_data/database"
metaDataBaseFile = os.path.join(dataBaseDir, 'pixel_wise_DataBase.txt')

# Directory where the processed data will be stored for further analysis
processedDataStoreDir= \
"/Users/burakgur/2p/Python_data/database/processed_data"


#%% User defined variables
use_otsu = True
crop_movie = True



