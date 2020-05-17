#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:22:12 2020

@author: burakgur
"""

import os
import numpy as np
import re
import glob
import time
from shutil import copyfile
#%%
def combine_movies(raw_path,folder_name,series_to_combine):
    start1 = time.time()
    folder_path = os.path.join(raw_path,folder_name)
    combined_folder_n = 'combined%s' % '_'.join(series_to_combine.astype(str))
    combined_f_path = os.path.join(folder_path, combined_folder_n)
    
    try:
        os.mkdir(combined_f_path)
    except OSError:
        print('--Folder exists... skipping'.format(f=combined_f_path))
        return []
    
    print('--Folder {f} generated...'.format(f=combined_f_path))
    
    t_series_names = [file_n for file_n in os.listdir(folder_path) \
                                      if 'tseries' in file_n.lower() or \
                                      't-series' in file_n.lower()]
    
    
    series_numbers = \
        np.array(map(lambda im_n : np.array(re.findall(r'\d+',im_n)).astype(int)[-1], 
                     t_series_names))
    if not np.all(np.isin(series_to_combine,series_numbers)):
        os.rmdir(combined_f_path)
        raise NameError('--Some of the t-series do not exist')
        
    for iSeries, serie_name in enumerate(t_series_names):
            image_t_num = np.array(re.findall(r'\d+',serie_name)).astype(int)[-1]
            if image_t_num in series_to_combine:
                t_series_path = os.path.join(folder_path, serie_name)
                t_series_files = t_series_path + '/' + '*.tif'
                all_images = (glob.glob(t_series_files))
                for image in all_images:
                    copyfile(image, os.path.join(combined_f_path,
                                                 os.path.basename(image)))
                print('\n--{t} tiffs copied...'.format(t=serie_name))
    end1 = time.time()
    time_passed = end1 - start1
    print('\n--Files combined in %d minutes\n' % \
          round(time_passed / 60))
#%%
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
# initialDirectory = '/Users/burakgur/Documents/data'
raw_path = os.path.join(initialDirectory, 'raw_data')


#%%
data_dict = {}
data_dict['200311bg_fly2'] = np.array([1,2,3,4,5])
data_dict['200311bg_fly3'] = np.array([1,2,3,4,5])
data_dict['200311bg_fly4'] = [np.array([1,2,3,4,5]) , np.array([6,7,8])]
data_dict['200306bg_fly1'] = np.array([1,2,3,4,5])
data_dict['200306bg_fly2'] = np.array([1,2,3,4,5])

data_dict['200305bg_fly2'] = np.array([1,2,3,4,5])
data_dict['200305bg_fly3'] = np.array([1,2,3,4,5])
data_dict['200305bg_fly4'] = np.array([1,2,3,4,5])

data_dict['200303bg_fly1'] = np.array([1,2,3,4,5])
data_dict['200303bg_fly2'] = np.array([1,2,3,4,5])
data_dict['200303bg_fly3'] = np.array([1,2,3,4,5])
data_dict['200303bg_fly4'] = np.array([1,2,5,6])
data_dict['200303bg_fly5'] = np.array([1,2,3,4,5,6])
data_dict['200303bg_fly6'] = np.array([1,2,3,4,5,7,8])
data_dict['200303bg_fly7'] = np.array([1,2,3,4,5,6,7])

data_dict['200228bg_fly1'] = np.array([3,4,5,6,7])
data_dict['200228bg_fly2'] = np.array([2,3,4,5,6])
data_dict['200228bg_fly3'] = np.array([1,2,3,4,5])
data_dict['200228bg_fly4'] = np.array([1,2,3,4,5,6,7])
data_dict['200228bg_fly5'] = np.array([1,2,4,5,6])
data_dict['200228bg_fly6'] = np.array([1,2,3,4,5])

data_dict['200226bg_fly1'] = np.array([1,4,5,6,7])
data_dict['200226bg_fly2'] = np.array([3,4,5,6,7,8])
data_dict['200226bg_fly4'] = np.array([1,2,3,4,5])

data_dict['200225bg_fly3'] = np.array([1,2,3,4,5])
data_dict['200225bg_fly4'] = np.array([1,2,3,4,5])

data_dict['200224bg_fly1'] = np.array([1,2,3,4,5,6])
data_dict['200224bg_fly2'] = np.array([1,2,3,4,5])
data_dict['200224bg_fly3'] = np.array([1,2,3,4,5,6,7])
data_dict['200224bg_fly4'] = np.array([1,2,3])
data_dict['200224bg_fly5'] = np.array([1,2,3,6,7])
               
data_dict['200221bg_fly1'] = np.array([1,2,3,4,5])
data_dict['200221bg_fly2'] = np.array([1,2,3,4,5])
data_dict['200221bg_fly3'] = [np.array([3,6]),  np.array([1,2]), 
                              np.array([7,8,9,10])]


data_dict['200220bg_fly1'] = np.array([1,2,4,5,6])
data_dict['200220bg_fly2'] = np.array([1,2,5,6,7])
data_dict['200220bg_fly3'] = np.array([1,2,3,4,5])
data_dict['200220bg_fly4'] = np.array([1,2,3,4,5])

data_dict['200219bg_fly1'] = np.array([1,2,3,4,5])
data_dict['200219bg_fly2'] = np.array([1,2,3,4,5])
data_dict['200219bg_fly3'] = np.array([1,2,3,4,5])
data_dict['200219bg_fly4'] = np.array([1,2,3,4,5])
data_dict['200219bg_fly5'] = np.array([1,2,3,6,7])

data_dict['200218bg_fly1'] = np.array([1,2,3,4,5])
data_dict['200218bg_fly2'] = np.array([1,2,3,4,5])
data_dict['200218bg_fly3'] = np.array([1,2])
data_dict['200218bg_fly4'] = np.array([1,2,3,4,5])

data_dict['200217bg_fly1'] = np.array([1,2,3,4,5])

data_dict['200213bg_fly1'] = np.array([1,2,3,5,7])
data_dict['200213bg_fly2'] = np.array([2,3,4,5,6])
data_dict['200213bg_fly3'] = np.array([1,2,3,4,5])
data_dict['200213bg_fly4'] = np.array([1,2,3,4,5])



#%% Combined the T series in a single folder
start1 = time.time()
for folder_name, series in data_dict.iteritems():
    print('-{f} started...'.format(f=folder_name))
    if type(series) is list:
        for series_to_combine in series:
            combine_movies(raw_path,folder_name,series_to_combine)
    else:
        series_to_combine = series.copy()
        combine_movies(raw_path,folder_name,series_to_combine)
end1 = time.time()
time_passed = end1 - start1
print('\All files combined in %d minutes\n' % \
      round(time_passed / 60))
        
        
        
        
        