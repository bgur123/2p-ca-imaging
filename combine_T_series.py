#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:30:42 2020

@author: burakgur
"""

import os
import numpy as np
import re
import glob
import time
from shutil import copyfile
#%%
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
# initialDirectory = '/Users/burakgur/Documents/data'
raw_path = os.path.join(initialDirectory, 'raw_data')


#%%
folder_name = '200228bg_fly5'
series_to_combine = np.array([1,3,4,5,6])


#%% Combined the T series in a single folder
start1 = time.time()

folder_path = os.path.join(raw_path,folder_name)
combined_folder_n = 'combined%s' % '_'.join(series_to_combine.astype(str))
combined_f_path = os.path.join(folder_path, combined_folder_n)
os.mkdir(combined_f_path)
print('Folder {f} generated...'.format(f=combined_f_path))

t_series_names = [file_n for file_n in os.listdir(folder_path) \
                                  if 'tseries' in file_n.lower() or \
                                  't-series' in file_n.lower()]


series_numbers = \
    np.array(map(lambda im_n : np.array(re.findall(r'\d+',im_n)).astype(int)[-1], 
                 t_series_names))
if not np.all(np.isin(series_to_combine,series_numbers)):
    os.rmdir(combined_f_path)
    raise NameError('Some of the t-series do not exist')
    
for iSeries, serie_name in enumerate(t_series_names):
        image_t_num = np.array(re.findall(r'\d+',serie_name)).astype(int)[-1]
        if image_t_num in series_to_combine:
            t_series_path = os.path.join(folder_path, serie_name)
            t_series_files = t_series_path + '/' + '*.tif'
            all_images = (glob.glob(t_series_files))
            for image in all_images:
                copyfile(image, os.path.join(combined_f_path,
                                             os.path.basename(image)))
            print('\n{t} tiffs copied...'.format(t=serie_name))
end1 = time.time()
time_passed = end1 - start1
print('\nFiles combined in %d minutes\n' % \
      round(time_passed / 60))