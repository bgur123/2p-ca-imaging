#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:00:59 2018

@author: deniz
"""

import tkinter
import tkinter.filedialog as filedialog
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import rf_tools as rt
#%%

# Ask RF file location

title='Please select RF file to draw'
root= tkinter.Tk()
root.withdraw()
rffile = filedialog.askopenfilename(initialdir=initialdir,title=title)

# Create needed folders
rfdir = os.path.dirname(rffile)
pardir = os.path.dirname(rfdir)
immaster = '3-RF_Figures'
if immaster not in os.listdir(pardir):
    os.mkdir(os.path.join(pardir,immaster))
immaster = os.path.join(pardir,immaster)
date = os.path.split(rffile)[1][:6]
if date not in os.listdir(immaster):
    os.mkdir(os.path.join(immaster,date))
immaster=os.path.join(immaster,date)
rfname = os.path.split(rffile)[1]
ind = re.search('RFdata',rfname).span()[0]
#flyid = os.path.split(rffile)[1][6:ind].strip('_')
flyid = rfname[6:-4].replace('_RFdata_','_').strip('_')
if flyid not in os.listdir(immaster):
    os.mkdir(os.path.join(immaster,flyid))
immaster = os.path.join(immaster,flyid)
datafile = os.path.join(pardir,'1-Analyzed_Data',
                        rfname[:ind].strip('_')+'.npz')

clusterAnalysis = False
with np.load(rffile) as rf:
    stas = rf['stas']
    roi_ind = rf['roi_ind']
    corrcoefs = rf['corrcoefs']
    epochind = rf['epochind']
if os.path.exist(datafile):
    with np.load(datafile) as data:
    	ch1amax = data['ch1amax']
    	masks = data['masks']
    	stimtype = data['stimtype']
    	if 'layer' in data.keys():
            clusterAnalysis= True
            t4t5 = data['t4t5']
            layer = data['layer']
else:
    title = "Please select mat file of whitenoise stimulus"
    datafile = rf.get_fileloc(title,'')
    data = rt.mattopython(datafile)
# ClusterAnalysis, stimtype, t4t5, layer,masks,ch1amax
# WRITE THE REST -------------------------------------------------------------------------------------
#%%
global_colorbar = True

if '23_El' in stimtype[0]:
    s1='Left'
    s2='Right'
if '23_Az' in stimtype[0]:
    s1='Up'
    s2='Down'
    stas = np.flip(stas,2)
    # @TODO: You have to flip the stas in 2nd dim when stim is azimuth

if global_colorbar:
# @TODO divide all the stas with their own standard deviations to have a similar scale globally
    stas = (stas[:,:,:].T)/np.std(stas[:,:,:],axis=(1,2))
    uplimit=np.max(np.abs(stas))
    downlimit=-uplimit
fig, ax0 = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
for e,i in enumerate(roi_ind):
    if not global_colorbar: # For the local scales, get local maximum and the negative of it
        uplimit=np.max(np.abs(stas[e,:,:]))
        downlimit=-uplimit
    pxlsize = np.sum(masks[i])
    if clusterAnalysis:
        plt.suptitle('Layer:%d T%d ROI:%d Pxl%d'%(layer[i],t4t5[i],i,pxlsize),fontsize=8)
    else:
        plt.suptitle(' ROI:%d Pxl%d'%(i,pxlsize),fontsize=8)

#Plot masks
    ax1 = plt.subplot2grid((1,2),(0,0))
    ax1.imshow(ch1amax,cmap = 'gray')
    ax1.imshow(np.ma.masked_where(masks[i]<1,masks[i]), cmap = 'autumn')
    ax1.set_aspect('auto')
    ax1.set_axis_off()
#Plot RF
    ax = plt.subplot2grid((1,2),(0,1))
    # To show rfs you have to plot transpose!
    im = ax.imshow(np.transpose((stas[e,:,:]), cmap='RdBu_r', vmin=downlimit,
                   vmax=uplimit,interpolation='nearest')
    ax.set_aspect('auto')
    ax.set_xlabel('Time (s)')
    ax.tick_params(axis='y',which='minor',left='on')
    ax.xaxis.set_ticks([0,stas[e,:,:].shape[0]/2-1,stas[e,:,:].shape[0]-1])
    ax.xaxis.set_ticklabels([-2,-1,0])
    ax.yaxis.set_ticks([0,stas[e,:,:].shape[1]-1])
    ax.yaxis.set_ticklabels([s1,s2])
    cb = fig.colorbar(im,ax=ax, orientation='vertical')
    cb.set_ticks([downlimit,uplimit])
    cb.set_ticklabels(['OFF','ON'])
#    plt.show()


    if clusterAnalysis:
        plt.savefig(immaster + '/t%d_layer%d_Cluster%d.png'%(layer[i],i),
                    format='pdf',transparent=True,bbox_inches='tight')
    else:
        plt.savefig(immaster + '/cluster%d.svg'%(i),format='svg',
                    transparent=True,bbox_inches='tight')
        plt.savefig(immaster + '/cluster%d.pdf'%(i),format='pdf',
                    transparent=True,bbox_inches='tight')
    plt.clf()
