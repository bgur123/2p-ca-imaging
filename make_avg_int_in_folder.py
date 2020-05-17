#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:53:33 2020

@author: burakgur
"""
#%%
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, measure, filters
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')

# %% User dependent directories, modify them according to your folder structure
# Directory where all the other directories located
initial_dir = '/Volumes/HD-SP1/Burak_data/Python_data'

# Directory where the folders are located
master_dir = os.path.join(initial_dir, 'selected_experiments/200213_luminance')

folder_name = '200313bg_fly1'
master_video_n = [1]

#%%
plt.rcParams["axes.titlesize"] = 'x-small'
experiment_path = os.path.join(master_dir,folder_name)

combined_names = [file_n for file_n in os.listdir(experiment_path) \
                                  if 'combined' in file_n.lower()]
    
combined_names.sort(key=lambda combined_n : \
                    np.max(np.array(re.findall(r'\d+',combined_n),dtype=int)))
for i_combined, combined_n in enumerate(combined_names):
    
    master_vid_n = master_video_n[i_combined]
    # Only the image containing folders starting with "combined"should be taken
    if combined_n[0] != 'c':
        continue
    videos_dir = os.path.join(experiment_path,combined_n,'motCorr.sima')
    t_series_names = [file_n for file_n in os.listdir(videos_dir) \
                                      if 'tseries' in file_n.lower() or \
                                      't-series' in file_n.lower()]
        
    
    taken_ts = []    
    non_master_ims = {}
    for im_n in t_series_names:
        t_num = np.array(re.findall(r'\d+',im_n)).astype(int)[-1] 
        movie_path = os.path.join(videos_dir,im_n)
        raw_movie = io.imread(movie_path)
        mean_image = raw_movie.mean(0)
        scaled_mean_image = mean_image/mean_image.max()
        # scaled_mean_image= scaled_mean_image>scaled_mean_image.mean()
        # # otsu_threshold_Value = filters.threshold_otsu(scaled_mean_image)
        # # otsu_thresholded_mask = scaled_mean_image > otsu_threshold_Value
        if t_num == master_vid_n:
            master_im = scaled_mean_image.copy()
            master_im_n = im_n
        else:
            non_master_ims[im_n] = scaled_mean_image.copy()
            
        move_path = os.path.join(experiment_path, im_n.split('_motCorr')[0])
        os.rename(movie_path, os.path.join(move_path,im_n))
    
    idx =0
    # fig, axes = plt.subplots(ncols=3, nrows=len(non_master_ims),figsize=())
    
    for im_n, image in non_master_ims.iteritems():
        t_num = np.array(re.findall(r'\d+',im_n)).astype(int)[-1] 
        # master_ax = axes[idx][0]
        # im_ax = axes[idx][1]
        # corr_ax = axes[idx][2]
        plt.close('all')
        fig = plt.figure(figsize=(16,10))
        master_ax = plt.subplot(221)
        im_ax = plt.subplot(222)
        corr_ax = plt.subplot(223)
        mix_ax = plt.subplot(224)
        
        sns.heatmap(master_im,cmap='binary',ax=master_ax,cbar=False)
        master_ax.axis('off')
        master_ax.set_title(master_im_n)
        
        sns.heatmap(image,cmap='binary',ax=im_ax,cbar=False)
        im_ax.axis('off')
        im_ax.set_title(im_n)
        
        diff = measure.compare_ssim(master_im, image, full=True,
                                    win_size=5)
        diff_im = diff[1]
        # diff_im[diff_im<diff[0]] = 0
        diff_im = 1-diff_im
        # diff_im = diff_im/diff_im.max()
        
        
        sns.heatmap(diff_im,cmap='Reds',ax=corr_ax,cbar=False)
        corr_ax.axis('off')
        corr_ax.set_title('Image difference')
        
        sns.heatmap(diff_im,cmap='Reds',ax=mix_ax,cbar=False)
        sns.heatmap(master_im,cmap='binary',ax=mix_ax,alpha=0.6,cbar=False)
        mix_ax.axis('off')
        mix_ax.set_title('Image difference')
        
        
        plt.waitforbuttonpress()
        take_im = raw_input("\nTake image (y/n)")
        if take_im == 'y':
            taken_ts.append(t_num)
        
        idx += 1
    
    if len(taken_ts) == len(non_master_ims):
        os.mkdir(os.path.join(experiment_path, 
                              'all_combined_good_{m}_{s}'.format(s=combined_n,
                                                                 m=master_vid_n)))
    elif len(taken_ts)>0:
        os.mkdir(os.path.join(experiment_path, 
                              'only_{m}_{s}_combined'.format(s='_'.join(map(str, 
                                                                            taken_ts)),
                                                             m=master_vid_n)))
    else:
        os.mkdir(os.path.join(experiment_path, 'none_{s}'.format(s=combined_n)))
   

