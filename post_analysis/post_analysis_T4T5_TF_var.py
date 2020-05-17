#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:27:02 2020

@author: burakgur
"""

import cPickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.cluster.hierarchy as shc
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

import ROI_mod
import post_analysis_core as pac

# %% Setting the directories
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
alignedDataDir = os.path.join(initialDirectory,
                              'selected_experiments/selected')
stimInputDir = os.path.join(initialDirectory, 'stimulus_types')
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data',
                             '200130_T4T5_goodData_TFVar')
summary_save_dir = os.path.join(initialDirectory,
                                'results/200116_T4T5_delay_experiments/TF_Var2/mixed')

# Plotting parameters
colors = pac.run_matplotlib_params()
color = colors[5]
color_pair = colors[2:4]
roi_plots=False
# %% Load datasets and desired variables
exp_t = '5th_exp_R64G09_mixed_CSI0p4_REL_0p4_all'
datasets_to_load = ['191206bg_fly1-TSeries-12062019-0951-002_transfer_sima_STICA.pickle',
                    '191206bg_fly4-TSeries-12062019-0951-002_transfer_sima_STICA.pickle',
                    '191210bg_fly2-TSeries-12102019-0944-007_transfer_sima_STICA.pickle',
                    '191212bg_fly3-TSeries-002_transfer_sima_STICA.pickle',
                    '191213bg_fly1-TSeries-12132019-0909-004_transfer.pickle',
                    '191216bg_fly2-TSeries-002_transfer_sima_STICA.pickle',
                    '191216bg_fly3-TSeries-002_transfer.pickle',
                    '200124bg_fly3-TSeries-01242020-0901-004_transfer.pickle',
                    '200127bg_fly3-TSeries-002_transfer.pickle',
                    '200127bg_fly4-TSeries-002_transfer_sima_STICA.pickle',
                    '200129bg_fly1-TSeries-004_transfer_sima_STICA.pickle',
                    '200129bg_fly3-TSeries-001_transfer_sima_STICA.pickle',
                    '200129bg_fly4-TSeries-001_transfer.pickle',
                    '200129bg_fly6-TSeries-001_transfer.pickle',
                    '200130bg_fly2-TSeries-004_transfer.pickle',
                    '200130bg_fly3-TSeries-001_transfer.pickle',
                    '200131bg_fly1-TSeries-01312020-0853-002_transfer.pickle',
                    '200131bg_fly1-TSeries-01312020-0853-004_transfer.pickle',
                    '200131bg_fly3-TSeries-01312020-0853-001_transfer.pickle' ]
                    
properties = ['BF', 'PD', 'SNR', 'CSI','Reliab','CS','depth','DSI',
              'RF_map_center']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
tunings = []
# Initialize variables
flyNum = 0
for idataset, dataset in enumerate(datasets_to_load):
    load_path = os.path.join(saveOutputDir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'CSI':0.4,'reliability':0.4})
    if len(curr_rois) < 2:
        continue
    # if not('Sine' in curr_rois[1].stim_name):
    #     continue
    # if not(('Sine' in curr_rois[0].stim_name) and ('30deg' in curr_rois[0].stim_name)):
    #     continue
    three_hz_indx = [idx for idx,roi in enumerate(curr_rois) if roi.BF ==3 ]
    if three_hz_indx:
        curr_rois.pop(three_hz_indx[0])
    all_rois.append(curr_rois)
    data_to_extract = ['SNR', 'CSI','reliability','CS','BF','PD','DSI']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    
    # There is just one ROI with 3Hz so discard that one
    
    tunings.append(np.squeeze\
                   (list(map(lambda roi: roi.TF_curve_resp,curr_rois))))
    depths = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))

    df_c = {}
    df_c['depth'] = depths
    df_c['RF_map_center'] = list(map(lambda roi : (roi.RF_map_norm>0.95).astype(int)
                                     , curr_rois))
    df_c['BF'] = roi_data['BF']
    df_c['PD'] = roi_data['PD']
    df_c['SNR'] = roi_data['SNR']
    df_c['CSI'] = roi_data['CSI']
    df_c['Reliab'] = roi_data['reliability']
    df_c['CS'] = roi_data['CS']
    df_c['DSI'] = roi_data['DSI']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['flyNum'] = np.tile(flyNum,len(curr_rois))
    flyNum = flyNum +1
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))

# %% CS vs BF
fig = plt.figure(5, figsize=(3, 3))
a = sns.countplot('BF', hue='CS', data=combined_df, 
                  palette=[colors[-1],colors[-2]], orient='h', hue_order=['OFF', 'ON'])

# ax1.set_xlabel('BF (Hz)')
# ax1.set_ylabel('Depth ($\mu m$)')
a.set_title('{exp_t}_BFvsCS'.format(exp_t=exp_t))

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsCS'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')


# %% BF vs z depth

fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
sns.violinplot(x="BF", y="depth", data=combined_df, scale="count", 
               inner="stick", palette='plasma', ax=ax2[0])

ax2[0].set_xlabel('BF (Hz)')
# ax1.set_xscale("log")
# ax1.set_xlim((0,5))
ax2[0].set_ylim((10, 60))
ax2[0].set_ylabel('Depth ($\mu m$)')
ax2[0].set_title('{exp_t}_BFvsDepth'.format(exp_t=exp_t))

sns.countplot(x="depth", hue='BF', data=combined_df, palette='plasma', 
              orient='h', ax=ax2[1])

# ax1.set_xlabel('BF (Hz)')
# ax1.set_ylabel('Depth ($\mu m$)')
ax2[1].set_title('{exp_t}_BFvsDepth'.format(exp_t=exp_t))

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsDepth'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

# %% BF vs CSI
fig2, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
sns.boxplot(x='BF',y='CSI',width=0.5,ax=ax3,linewidth=1.5,data=combined_df,
               palette='plasma',notch=True)
ax3.set_ylim((0, 1.5))
ax3.set_ylabel('CSI')
ax3.set_xlabel('BF (Hz)')

fig.tight_layout()

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsCSI'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

# %% BF vs SNR
fig2, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
sns.boxplot(x='BF',y='SNR',width=0.5,ax=ax3,linewidth=1.5,data=combined_df,
               palette='plasma',notch=True)
# ax3.set_ylim((0, 1.5))
ax3.set_ylabel('SNR')
ax3.set_xlabel('BF (Hz)')

fig.tight_layout()

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsSNR'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

# %% BF vs Rel
fig2, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
sns.boxplot(x='BF',y='Reliab',width=0.5,ax=ax3,linewidth=1.5,data=combined_df,
               palette='plasma',notch=True)
# ax3.set_ylim((0, 1.5))
ax3.set_ylabel('Reliability')
ax3.set_xlabel('BF (Hz)')

fig.tight_layout()

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsRel'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')
# %% BF vs DSI
fig2, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
sns.violinplot(x='BF',y='DSI',width=0.5,ax=ax3,linewidth=1.5,data=combined_df,
               palette='plasma',inner="quartile")
# ax3.set_ylim((0, 25))
ax3.set_ylabel('PD vector length')
ax3.set_xlabel('BF (Hz)')

fig.tight_layout()

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsDSI'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

#%% BF vs vectors
plt.close('all')
fig = plt.figure(figsize=(15, 8))
fig.suptitle('DSI Vectors')

cmap = matplotlib.cm.get_cmap('inferno')
norm = matplotlib.colors.Normalize(vmin=0, vmax=1.5)

unique_bf_num = len(np.unique(combined_df['BF']))
for idx, bf in enumerate(np.unique(combined_df['BF'])):
    ax = fig.add_subplot(2, 3, idx+1,projection='polar',alpha=.8)
    curr_df = combined_df[combined_df['BF']==bf]
    num=len(curr_df)
    for index, row in curr_df.iterrows():
        curr_bf = row['BF']
        curr_dsi = row['DSI']
        # Angles will be adjusted for: 0-180 horizontal 90-270 vertical
        curr_pd = np.mod(90-row['PD'],360)
        # ax.plot(curr_dsi, curr_pd,color=cmap(norm(curr_bf)))
        
        x_vec = curr_dsi*np.cos(np.radians(curr_pd))
        y_vec = curr_dsi*np.sin(np.radians(curr_pd))
        ax.quiver(0,0, np.radians(curr_pd),curr_dsi,color=cmap(norm(curr_bf)),
                    scale=1,angles="xy",scale_units='xy',alpha=1.)
    ax.set_rmin(0)
    ax.set_rmax(3.0)
    ax.set_title('{bf}Hz | {s} ROIs'.format(bf= bf,s=num))
    ax.grid(alpha=.3)
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_BF_DSI_Vector'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
# plt.close('all')




# %% BF vs Z depth with stacked bars
df_plot = \
    combined_df.groupby(['BF', 'depth']).size().reset_index().pivot(columns='BF',
                                                                            index='depth', values=0)
df_plot.plot(kind='bar', stacked=True, cmap='plasma')

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsDepth_v2'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

# %% Tuning curves

tuning_curves = np.concatenate((tunings[:]))
all_mean_data = np.mean(tuning_curves, axis=0)
all_yerr = np.std(tuning_curves, axis=0)
epoch_freqs = curr_rois[0].TF_curve_stim
fly_IDs = combined_df['flyID']
unique_flies = np.unique(fly_IDs)
norm_tuning_cur_flies = {}
norm_tf_tuning = normalize(tuning_curves, axis=1, norm='max')
for fly_num in unique_flies:
    curr_ROI_mask = (fly_IDs == fly_num)
    norm_tuning_cur_flies[fly_num] = norm_tf_tuning[curr_ROI_mask, :]
## Plots
fig = plt.figure(figsize=(5, 7))
grid = plt.GridSpec(3, 1, wspace=0, hspace=0.5)
ax = plt.subplot(grid[0])
chart = sns.countplot('BF', data=combined_df, palette='plasma')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, fontweight='light')
leg = chart.legend()
leg.remove()
ax.set_xlabel('BF (Hz)')
ax.set_title('{exp_t}_BF_tuning'.format(exp_t=exp_t))
ax.set_ylabel('ROI count')

## Tuning curve
ax2 = plt.subplot(grid[1:])
mean_t = all_mean_data
std_t = all_yerr
ub = mean_t + std_t
lb = mean_t - std_t
# Tuning curve
ax2.fill_between(epoch_freqs, ub, lb, color=color, alpha=.2)
ax2.plot(epoch_freqs, mean_t, '-o', lw=4, color=color, markersize=10)

ax2.set_xscale('log')
ax2.set_title('')
ax2.set_xlabel('Temporal Frequency (Hz)')
ax2.set_ylabel('$\Delta F/F$')
ax2.set_xlim((ax2.get_xlim()[0], 10))
if 1:
    # Saving figure
    save_name = '{exp_t}_BF_tuning'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

#%% PCA analysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=8)
# Scale the input 
pca.fit(norm_tf_tuning)
pca_data = pca.transform(norm_tf_tuning)
pca_data_inv = pca.inverse_transform(pca_data)

combined_df['PC1'] = pca_data[:,0]*10
combined_df['PC2'] = pca_data[:,1]*10
combined_df['PC3']= pca_data[:,2]*10

explained_variance = pca.explained_variance_ratio_
explained_variance_cum = np.zeros(explained_variance.shape)
cum_so_far=0
for idx, val in enumerate(explained_variance):
    cum_so_far = explained_variance[idx]+cum_so_far
    explained_variance_cum[idx]= cum_so_far


fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(2, 3, wspace=0.5, hspace=0.5)

plt.subplot(grid[0,0])
plt.plot(explained_variance_cum)
plt.xlabel('# of components')
plt.ylabel('Explained $\sigma$')
plt.title('PCA components')
plt.ylim((0,1))

# plt.subplot(grid[0,1:],projection='3d')
plt.subplot(grid[:,1:])
sns.scatterplot(x='PC2',y='PC1',s=30 , linewidth=0,alpha=.99,
              data=combined_df,hue='BF',palette='inferno')
ax = plt.gca()
plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='10') # for legend title
fig.tight_layout()

plt.subplot(grid[1,0])
plt.plot(epoch_freqs,np.transpose(pca.components_[:2]))
plt.legend(['PC1','PC2'])
plt.xscale('log')
plt.xlabel('Hz')
plt.ylabel('Weight')



if 1:
    # Saving figure
    save_name = '{exp_t}_PCA'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

#%%
from collections import OrderedDict
from functools import partial
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.
Axes3D

n_neighbors = 10
n_components = 2

# Set-up manifold methods
LLE = partial(manifold.LocallyLinearEmbedding,
              n_neighbors, n_components, eigen_solver='auto')

methods = OrderedDict()
methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                 random_state=0)
methods['LLE'] = LLE(method='standard')
methods['Modified LLE'] = LLE(method='modified')
methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                           n_neighbors=n_neighbors)


X = norm_tf_tuning
fig = plt.figure(figsize=(15, 8))
fig.suptitle("Manifold Learning with %i neighbors"
             % (n_neighbors), fontsize=14)
ax = fig.add_subplot(2, 3, 1)
ax.scatter(pca_data[:, 0], pca_data[:, 1], c=combined_df['BF'], 
           s=25,alpha=.8,cmap=plt.cm.inferno)
ax.set_title("PCA")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')
# Plot results
for i, (label, method) in enumerate(methods.items()):
    t0 = time()
    Y = method.fit_transform(X)
    m1 = '{m}1'.format(m=label)
    m2 = '{m}2'.format(m=label)
    combined_df[m1] = Y[:,0]*10
    combined_df[m2] = Y[:,1]*10
    t1 = time()
    print("%s: %.2g sec" % (label, t1 - t0))
    ax = fig.add_subplot(2, 3, i+2)
    ax.scatter(Y[:, 0], Y[:, 1], c=combined_df['BF'], cmap=plt.cm.inferno,
               s=25,alpha=.8)
    ax.set_title("%s" % (label))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')

if 1:
    # Saving figure
    save_name = '{exp_t}_Manifold_learning_{n}neighbors_BF'.format(exp_t=exp_t,
                                                                n=n_neighbors)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

# %% Clustering the tuning curves
plt.close('all')
plt.figure(1)
plt.title("Cluster TF tuning dendrograms")
dend = shc.dendrogram(shc.linkage(norm_tf_tuning, method='ward'))
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
prediction = cluster.fit_predict(norm_tf_tuning) 
combined_df['Hierarchical_labels'] = prediction
# plt.close('all')

plt.figure(2, figsize=(16, 3))
plt.subplot(132)
for i, cl_type in enumerate(np.unique(prediction)):
    curr_data = norm_tf_tuning[np.where(prediction == cl_type)[0], :]
    yerr = np.std(curr_data, axis=0)
    mean_data = np.mean(curr_data, axis=0)
    plt.fill_between(epoch_freqs, mean_data + yerr, mean_data - yerr, color=colors[i+2], alpha=.2)
    plt.plot(epoch_freqs, mean_data, '-o', lw=2, color=colors[i+2], markersize=7,
             label=('Cluster %d, N: %d ROIs' % (cl_type, len(np.where(prediction == cl_type)[0]))))

plt.legend()
plt.title('Predicted clusters Hierarchical')
plt.xscale('log')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')

plt.subplot(131)
plt.fill_between(epoch_freqs, ub, lb, color=color, alpha=.2)
plt.plot(epoch_freqs, mean_t, '-o', lw=2, color=color, markersize=7)
plt.legend()
plt.xscale('log')
plt.title('Mean responses')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')

plt.subplot(133)
plt.title('t-SNE on tunings')
sns.scatterplot(x='t-SNE1',y='t-SNE2',s=20 , linewidth=0,alpha=.7,
              data=combined_df,hue='Hierarchical_labels',palette=color_pair)
# for curr_bf in np.unique(combined_df['BF']):
#     curr_mask = combined_df['BF']==curr_bf
#     plt.scatter(combined_df['PC1'][curr_mask], 
#                 combined_df['PC2'][curr_mask], 
#                 combined_df['PC3'][curr_mask], 
#                 color=cmap(norm(curr_bf)),
#                 linewidth=4,facecolor=cmap(norm(curr_bf)))


ax = plt.gca()
plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='10') # for legend title
fig.tight_layout()
if 1:
    # Saving figure
    save_name = '{exp_t}_Tuning_hierarchical_cluster'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

#%% K means clustering
plt.close('all')
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(norm_tf_tuning)
prediction = (kmeans.labels_-1) * -1
combined_df['K_means_labels'] = prediction
plt.figure(2, figsize=(16, 3))
plt.subplot(132)
for i, cl_type in enumerate(np.unique(prediction)):
    curr_data = norm_tf_tuning[np.where(prediction == cl_type)[0], :]
    yerr = np.std(curr_data, axis=0)
    mean_data = np.mean(curr_data, axis=0)
    plt.fill_between(epoch_freqs, mean_data + yerr, mean_data - yerr, color=colors[i+2], alpha=.2)
    plt.plot(epoch_freqs, mean_data, '-o', lw=2, color=colors[i+2], markersize=7,
             label=('Cluster %d, N: %d ROIs' % (cl_type, len(np.where(prediction == cl_type)[0]))))

plt.legend()
plt.title('Predicted clusters Kmeans')
plt.xscale('log')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')

plt.subplot(131)
plt.fill_between(epoch_freqs, ub, lb, color=color, alpha=.2)
plt.plot(epoch_freqs, mean_t, '-o', lw=2, color=color, markersize=7)
plt.legend()
plt.xscale('log')
plt.title('Mean responses')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')


plt.subplot(133)
plt.title('t-SNE on tunings')
sns.scatterplot(x='t-SNE1',y='t-SNE2',s=20 , linewidth=0,alpha=.7,
              data=combined_df,hue='K_means_labels',palette=color_pair)
# sns.scatterplot(x='t-SNE1',y='t-SNE2',s=20 , linewidth=0,alpha=.7,
              # data=combined_df,hue='flyNum',palette='Dark2',legend=False)
# for curr_bf in np.unique(combined_df['BF']):
#     curr_mask = combined_df['BF']==curr_bf
#     plt.scatter(combined_df['PC1'][curr_mask], 
#                 combined_df['PC2'][curr_mask], 
#                 combined_df['PC3'][curr_mask], 
#                 color=cmap(norm(curr_bf)),
#                 linewidth=4,facecolor=cmap(norm(curr_bf)))


ax = plt.gca()
plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='10') # for legend title
fig.tight_layout()

if 1:
    # Saving figure
    save_name = '{exp_t}_Tuning_Kmeans_cluster'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')
#%% Cluster evaluation
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(norm_tf_tuning)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
#%% RF analysis
plt.close('all')
fig = plt.figure(figsize=(15, 8))
fig.suptitle("RF distributions")


screen = np.zeros(np.shape(combined_df['RF_map_center'][0]))
screen[np.isnan(curr_rois[0].RF_map)] = -0.1


unique_bf_num = len(np.unique(combined_df['BF']))
for idx, bf in enumerate(np.unique(combined_df['BF'])):
    ax = fig.add_subplot(2, 3, idx+1)
    curr_maps = combined_df['RF_map_center'][combined_df['BF']==bf]
    num=len(curr_maps)
    plt.imshow(screen, cmap='binary', alpha=.5)
    plt_map = (curr_maps.sum().astype(float) / num) * 100
    plt_map[plt_map==0] = np.nan
    plt.imshow(plt_map,alpha=.9,cmap='inferno',vmin=0,vmax=25)

    cbar = plt.colorbar(shrink=.5)
    cbar.set_clim((0,25))
    cbar.set_label('% of ROIs')
    ax.set_xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
    ax.set_ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
    ax.axis('off')
    ax.set_title('{bf}Hz | {s} ROIs'.format(bf= bf,s=num))
    
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_of_diff_BF'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.close('all')

#%%
plt.close('all')
fig = plt.figure(figsize=(15, 8))
fig.suptitle("RF distributions K means")


screen = np.zeros(np.shape(combined_df['RF_map_center'][0]))
screen[np.isnan(curr_rois[0].RF_map)] = -0.1


unique_cluster_num = len(np.unique(combined_df['K_means_labels']))
for idx, cl in enumerate(np.unique(combined_df['K_means_labels'])):
    ax = fig.add_subplot(2, 3, idx+1)
    curr_maps = combined_df['RF_map_center'][combined_df['K_means_labels']==cl]
    num=len(curr_maps)
    plt.imshow(screen, cmap='binary', alpha=.5)
    plt_map = (curr_maps.sum().astype(float) / num) * 100
    plt_map[plt_map==0] = np.nan
    plt.imshow(plt_map,alpha=.9,cmap='inferno',vmin=0,vmax=25)

    cbar = plt.colorbar(shrink=.5)
    cbar.set_clim((0,25))
    cbar.set_label('% of ROIs')
    ax.set_xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
    ax.set_ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
    ax.axis('off')
    ax.set_title('Cluster {bf} | {s} ROIs'.format(bf= cl,s=num))
    
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_of_diff_K-means'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.close('all')

#%%
plt.close('all')
fig = plt.figure(figsize=(15, 8))
fig.suptitle("RF distributions Hierarchical")


screen = np.zeros(np.shape(combined_df['RF_map_center'][0]))
screen[np.isnan(curr_rois[0].RF_map)] = -0.1


unique_cluster_num = len(np.unique(combined_df['Hierarchical_labels']))
for idx, cl in enumerate(np.unique(combined_df['Hierarchical_labels'])):
    ax = fig.add_subplot(2, 3, idx+1)
    curr_maps = combined_df['RF_map_center'][combined_df['Hierarchical_labels']==cl]
    num=len(curr_maps)
    plt.imshow(screen, cmap='binary', alpha=.5)
    plt_map = (curr_maps.sum().astype(float) / num) * 100
    plt_map[plt_map==0] = np.nan
    plt.imshow(plt_map,alpha=.9,cmap='inferno',vmin=0,vmax=25)

    cbar = plt.colorbar(shrink=.5)
    cbar.set_clim((0,25))
    cbar.set_label('% of ROIs')
    ax.set_xlim(((np.shape(screen)[0]-60)/2,(np.shape(screen)[0]-60)/2+60))
    ax.set_ylim(((np.shape(screen)[0]-60)/2+60,(np.shape(screen)[0]-60)/2))
    ax.axis('off')
    ax.set_title('Cluster {bf} | {s} ROIs'.format(bf= cl,s=num))
    
if 1:
    # Saving figure
    save_name = '{exp_t}_RF_of_diff_hierarchical'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.close('all')

# %% cluster vs z depth

fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
sns.violinplot(x="K_means_labels", y="depth", data=combined_df, 
               inner="quartile", palette=color_pair, ax=ax2[0])

ax2[0].set_xlabel('Cluster')
# ax1.set_xscale("log")
# ax1.set_xlim((0,5))
# ax2[0].set_ylim((10, 60))
ax2[0].set_ylabel('Depth ($\mu m$)')
ax2[0].set_title('{exp_t}_BFvsDepth'.format(exp_t=exp_t))

sns.countplot(x="depth", hue='K_means_labels', data=combined_df, palette=color_pair, 
              orient='h', ax=ax2[1])

# ax1.set_xlabel('BF (Hz)')
# ax1.set_ylabel('Depth ($\mu m$)')
ax2[1].set_title('{exp_t}_KmeansvsDepth'.format(exp_t=exp_t))

if 1:
    # Saving figure
    save_name = '{exp_t}_KmeansvsDepth'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

#%% Cluster vs indiv flies
fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))

sns.countplot(x="flyNum", hue='K_means_labels', data=combined_df, palette=color_pair, 
              orient='h', ax=ax2)

# ax1.set_xlabel('BF (Hz)')
# ax1.set_ylabel('Depth ($\mu m$)')
ax2.set_title('{exp_t}_KmeansvsFlyNum'.format(exp_t=exp_t))

if 1:
    # Saving figure
    save_name = '{exp_t}_KmeansvsFlyNum'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

# %% BF vs CSI
fig2, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))
sns.boxplot(x='K_means_labels',y='CSI',width=0.5,ax=ax3,linewidth=1.5,data=combined_df,
               palette=color_pair,notch=True)
ax3.set_ylim((0, 1.5))
ax3.set_ylabel('CSI')
ax3.set_xlabel('Cluster')

fig.tight_layout()

if 1:
    # Saving figure
    save_name = '{exp_t}_ClustervsCSI'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')
#%%
if roi_plots:
    import random
    import copy
    plt.close('all')
    data_to_extract = ['DSI', 'BF', 'SNR', 'reliability', 'uniq_id','CSI',
                           'PD', 'exp_ID', 'stim_name']
    
    
    roi_figure_save_dir = os.path.join(summary_save_dir, 'ROI_ex0')
    if not os.path.exists(roi_figure_save_dir):
        os.mkdir(roi_figure_save_dir)
        
    # conditions = (combined_df['K_means_labels'] == 0) & (combined_df['flyNum'] == 5)
    conditions = (combined_df['K_means_labels'] == 0) & (combined_df['CS'] == 'ON') 
    mask = np.where(conditions)[0]
    copy_rois = copy.deepcopy(np.concatenate(all_rois)[mask.astype(int)])
    random.shuffle(copy_rois)
    roi_d = ROI_mod.data_to_list(copy_rois, data_to_extract)
    rois_df = pd.DataFrame.from_dict(roi_d)
    for n,roi in enumerate(copy_rois):
        if n>30:
            break
        fig = ROI_mod.make_ROI_tuning_summary(rois_df, roi,cmap='coolwarm')
        save_name = '%s_ROI_summary_%d' % ('a', roi.uniq_id)
        os.chdir(roi_figure_save_dir)
        fig.savefig('%s.png' % save_name,bbox_inches='tight',
                           transparent=False,dpi=300)
            
        plt.close('all')
        
    
    roi_figure_save_dir = os.path.join(summary_save_dir, 'ROI_ex1')
    if not os.path.exists(roi_figure_save_dir):
        os.mkdir(roi_figure_save_dir)
        
    # conditions = (combined_df['K_means_labels'] == 1) & (combined_df['flyNum'] == 5)
    conditions = (combined_df['K_means_labels'] == 1) & (combined_df['CS'] == 'ON') 
    mask = np.where(conditions)[0]
    copy_rois = copy.deepcopy(np.concatenate(all_rois)[mask.astype(int)])
    random.shuffle(copy_rois)
    roi_d = ROI_mod.data_to_list(copy_rois, data_to_extract)
    rois_df = pd.DataFrame.from_dict(roi_d)
    for n,roi in enumerate(copy_rois):
        if n>30:
            break
        fig = ROI_mod.make_ROI_tuning_summary(rois_df, roi,cmap='coolwarm')
        save_name = '%s_ROI_summary_%d' % ('a', roi.uniq_id)
        os.chdir(roi_figure_save_dir)
        fig.savefig('%s.png' % save_name,bbox_inches='tight',
                           transparent=False,dpi=300)
            
        plt.close('all')
                
            