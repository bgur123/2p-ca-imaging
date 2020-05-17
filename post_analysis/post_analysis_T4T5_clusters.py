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
                             '191113_T4T5_all_re_analyzed_1')
summary_save_dir = os.path.join(initialDirectory,
                                'results/200110_T4T5_re')

# Plotting parameters
colors = pac.run_matplotlib_params()
color = colors[5]

# %% Load datasets and desired variables
exp_t = '3rd_exp_R42F06x2_sine_gratings_30'
datasets_to_load = ['190912bg_fly3-TSeries-09122019-1024-002_sima_STICA.pickle',
                    '190912bg_fly3-TSeries-09122019-1024-004_sima_STICA.pickle',
                    '190912bg_fly3-TSeries-09122019-1024-006_sima_STICA.pickle',
                    '190912bg_fly4-TSeries-09122019-1024-002_sima_STICA.pickle',
                    '190912bg_fly4-TSeries-09122019-1024-008_sima_STICA.pickle',
                    '190912bg_fly4-TSeries-09122019-1024-010_sima_STICA.pickle',
                    '190913bg_fly2-TSeries-09132019-1009-002_sima_STICA.pickle',
                    '190913bg_fly2-TSeries-09132019-1009-004_sima_STICA.pickle',
                    '190913bg_fly4-TSeries-09132019-1009-004_sima_STICA.pickle',
                    '190913bg_fly4-TSeries-09132019-1009-007_sima_STICA.pickle']
edge_exists = 0
final_rois_all = []
flyIDs = []
expIDs = []
tunings = []
dist_data = []
uniq_bfs = []
spread_factors = []
properties = ['DSI', 'SNR', 'reliability', 'BF', 'z_depth', 'exp_ID', 'uniq_id', 'distance']
rois_dict = {}
all_bfs = []
combined_df = pd.DataFrame(columns=properties)
for idataset, dataset in enumerate(datasets_to_load):
    load_path = os.path.join(saveOutputDir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    expIDs.append(workspace['current_exp_ID'])
    final_rois_all.append(workspace['final_rois'])
    flyIDs.append(list(map(lambda roi: roi.exp_ID, workspace['final_rois'])))
    tunings.append(np.squeeze(list(map(lambda roi: roi.TF_curve_resp, workspace['final_rois']))))

    if edge_exists:
        ROI_mod.calculate_edge_timing(workspace['final_rois'])

    rois_dict = ROI_mod.data_to_list(workspace['final_rois'], properties)
    if edge_exists:
        rois_dict['distance'] = np.array(rois_dict['distance']) * workspace['final_rois'][0].imaging_info['pix_size']

    roi_dist_matrix = ROI_mod.calculate_distance_between_rois(workspace['final_rois'])
    roi_dist_matrix[roi_dist_matrix == 0] = np.nan
    # get rid of diagonal where the distance between the same roi is present
    # roi_dist_matrix = roi_dist_matrix[~np.eye(roi_dist_matrix.shape[0], dtype=bool)].reshape(roi_dist_matrix.shape[0], -1)
    roi_bfs = np.array(rois_dict['BF'])
    uniq_bf = []
    for freq in np.unique(roi_bfs):
        if len(roi_bfs[roi_bfs == freq]) == 1:
            continue
        else:
            uniq_bf.append(freq)

    dist_matrix = np.zeros((len(uniq_bf), len(uniq_bf)))

    for i, freq1 in enumerate(uniq_bf):
        temp_dist = roi_dist_matrix[roi_bfs == freq1, :]
        for j, freq2 in enumerate(uniq_bf):
            dist_matrix[i, j] = np.nanmean(temp_dist[:, roi_bfs == freq2])
    dist_data.append(dist_matrix)
    uniq_bfs.append(uniq_bf)
    all_bfs = np.concatenate((all_bfs, uniq_bf))
    other_freqs = dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)].reshape(dist_matrix.shape[0], -1)
    spread_factor = np.diag(dist_matrix) / other_freqs.mean(axis=1)
    spread_factors.append(spread_factor)

    curr_df = pd.DataFrame.from_dict(rois_dict)
    combined_df = combined_df.append(curr_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded'.format(ds=dataset))

# %% CS vs BF
fig = plt.figure(5, figsize=(3, 3))
a = sns.countplot('BF', hue='CS', data=combined_df, palette='plasma', orient='h', hue_order=['OFF', 'ON'])

# ax1.set_xlabel('BF (Hz)')
# ax1.set_ylabel('Depth ($\mu m$)')
a.set_title('{exp_t}_BFvsCS'.format(exp_t=exp_t))

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsCS'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
plt.show()
plt.close('all')

# %% BF vs edge properties
fig = plt.figure(6, figsize=(5, 5))
ax = sns.relplot(x='edge_peak_t', y='BF', hue='CS', data=combined_df, palette='plasma', alpha=.7,
                 hue_order=['OFF', 'ON'],
                 size='edge_response', sizes=(1, 50))

ax.axes[0][0].set_xlabel('Edge timing')
ax.axes[0][0].set_ylabel('BF')
ax.axes[0][0].set_yscale("log")
# fig.set_title('{exp_t}_BFvsEdgeProps'.format(exp_t=exp_t))

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsEdgeProps'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
    print('{sn} saved...'.format(sn=save_name))
plt.show()
# plt.close('all')

# %% Distance between same BF

# Plot matrices of single flies
for i in range(len(dist_data)):
    plt.figure()
    sns.heatmap(dist_data[i], vmin=0, cmap=sns.cm.rocket_r, xticklabels=uniq_bfs[i],
                yticklabels=uniq_bfs[i], cbar_kws={'label': 'Distance $\mu m$'})
    plt.yticks(rotation=0)
    plt.title(expIDs[i])
    # Saving figure
    save_name = '{exp_t}_ROI_spread_factor_matrix_{i}'.format(exp_t=exp_t,i=i)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=100)
    plt.close('all')

#%%
# Plot spread factor
uniq_all_bfs = np.unique(all_bfs)
mean_distances = np.zeros((len(uniq_all_bfs), 1))
std_distances = np.zeros((len(uniq_all_bfs), 1))

mean_sf = np.zeros((len(uniq_all_bfs), 1))
std_sf = np.zeros((len(uniq_all_bfs), 1))

for i, freq in enumerate(uniq_all_bfs):
    curr_dist = []
    curr_spread = []
    for k in range(len(dist_data)):
        curr_dist = np.concatenate((curr_dist, np.diag(dist_data[k])[uniq_bfs[k] == freq]))
        curr_spread = np.concatenate((curr_spread, spread_factors[k][uniq_bfs[k] == freq]))
    mean_distances[i] = curr_dist.mean()
    std_distances[i] = curr_dist.std()

    mean_sf[i] = np.nanmean(curr_spread)
    std_sf[i] = np.nanstd(curr_spread)

plt.errorbar(uniq_all_bfs, np.transpose(mean_sf)[0], np.transpose(std_sf)[0], elinewidth=3, c=color,
             ms=10, lw=4, marker='o')
plt.ylim((0,1.3))
plt.plot([0, 5], [1, 1], '--k')
plt.xlim((0,3.5))
plt.ylabel('Spread factor')
plt.xlabel('BF (Hz)')
plt.title('{exp_t}_ROI_sf'.format(exp_t=exp_t))

if 1:
    # Saving figure
    save_name = '{exp_t}_ROI_spread_factors'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
# plt.close('all')
# %% BF vs distance plots
plt.close('all')
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
sns.violinplot(x="BF", y="distance", data=combined_df, scale="count", 
               inner="stick", palette='plasma', ax=ax[0])

ax[0].set_xlabel('BF (Hz)')
# ax1.set_xscale("log")
# ax1.set_xlim((0,5))
# ax1.set_ylim((20,60))
ax[0].set_ylabel('Distance ($\mu m$)')
ax[0].set_title('{exp_t}_BFvsDist'.format(exp_t=exp_t))

sns.regplot(x="BF", y="distance", data=combined_df, x_jitter=.03, ax=ax[1],
            color=color, fit_reg=False, x_estimator=np.mean, scatter_kws={'s': 40}, line_kws={'lw': 3})

ax[1].set_xlabel('BF (Hz)')
ax[1].set_xscale("log")
ax[1].set_xlim((0, 5))
ax[1].set_ylabel('Distance ($\mu m$)')
ax[1].set_title('{exp_t}_BFvsDist'.format(exp_t=exp_t))

if 1:
    # Saving figure
    save_name = '{exp_t}_BFvsDist'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')

# %% BF vs z depth

fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
sns.violinplot(x="BF", y="z_depth", data=combined_df, scale="count", 
               inner="stick", palette='plasma', ax=ax2[0])

ax2[0].set_xlabel('BF (Hz)')
# ax1.set_xscale("log")
# ax1.set_xlim((0,5))
ax2[0].set_ylim((10, 80))
ax2[0].set_ylabel('Depth ($\mu m$)')
ax2[0].set_title('{exp_t}_BFvsDepth'.format(exp_t=exp_t))

sns.countplot(x="z_depth", hue='BF', data=combined_df, palette='plasma', orient='h', ax=ax2[1])

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

# %% BF vs Z depth with stacked bars
df_plot = combined_df.groupby(['BF', 'z_depth']).size().reset_index().pivot(columns='BF',
                                                                            index='z_depth', values=0)
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
epoch_freqs = final_rois_all[0][0].TF_curve_stim
fly_IDs = np.concatenate((flyIDs[:]))
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

# %% Clustering the tuning curves
plt.close('all')
plt.figure(1)
plt.title("Cluster TF tuning dendrograms")
dend = shc.dendrogram(shc.linkage(norm_tf_tuning, method='ward'))
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
prediction = cluster.fit_predict(norm_tf_tuning)
# plt.close('all')

plt.figure(2, figsize=(12, 5))
plt.subplot(122)
for i, cl_type in enumerate(np.unique(prediction)):
    curr_data = norm_tf_tuning[np.where(prediction == cl_type)[0], :]
    yerr = np.std(curr_data, axis=0)
    mean_data = np.mean(curr_data, axis=0)
    plt.fill_between(epoch_freqs, mean_data + yerr, mean_data - yerr, color=colors[i+2], alpha=.2)
    plt.plot(epoch_freqs, mean_data, '-o', lw=2, color=colors[i+2], markersize=7,
             label=('Cluster %d, N: %d ROIs' % (cl_type, len(np.where(prediction == cl_type)[0]))))

plt.legend()
plt.title('Predicted clusters')
plt.xscale('log')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')

plt.subplot(121)
plt.fill_between(epoch_freqs, ub, lb, color=color, alpha=.2)
plt.plot(epoch_freqs, mean_t, '-o', lw=2, color=color, markersize=7)
plt.legend()
plt.xscale('log')
plt.title('Mean responses')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')
if 1:
    # Saving figure
    save_name = '{exp_t}_Tuning_hierarchical_cluster'.format(exp_t=exp_t)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
plt.show()
plt.close('all')
# %% Single fly cluster fit
fly_data = {}
for flyID, tunings in norm_tuning_cur_flies.items():
    prediction = cluster.fit_predict(tunings)
    plt.figure()

    for i, cl_type in enumerate(np.unique(prediction)):
        curr_data = tunings[np.where(prediction == cl_type)[0], :]
        fly_data[flyID] = curr_data
        yerr = np.std(curr_data, axis=0)
        mean_data = np.mean(curr_data, axis=0)
        plt.fill_between(epoch_freqs, mean_data + yerr, mean_data - yerr, color=colors[i+2], alpha=.2)
        plt.plot(epoch_freqs, mean_data, '-o', lw=2, color=colors[i+2], markersize=7,
                 label=('Cluster %d, N: %d ROIs' % (cl_type, len(np.where(prediction == cl_type)[0]))))
    plt.legend()
    plt.title('Fly ID: %s' % flyID)
    plt.xscale('log')
    plt.ylabel('Normalized dF/F')
    plt.xlabel('Temporal Frequency (Hz)')

    if 1:
        # Saving figure
        save_name = '{exp_t}_Fly_{flyID}_tuning_clusters'.format(exp_t=exp_t, flyID=flyID)
        os.chdir(summary_save_dir)
        plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
    plt.show()
    plt.close('all')

# %% Single fly separate clustering
flyID = '190116bg_fly2'
plt.close('all')
plt.figure(3)
plt.title("Cluster TF tuning dendrograms")
dend = shc.dendrogram(shc.linkage(fly_data[flyID], method='ward'))
plt.show()
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
prediction = cluster.fit_predict(fly_data[flyID])

plt.figure(2, figsize=(12, 5))
plt.subplot(122)
for i, cl_type in enumerate(np.unique(prediction)):
    curr_data = fly_data[flyID][np.where(prediction == cl_type)[0], :]
    yerr = np.std(curr_data, axis=0)
    mean_data = np.mean(curr_data, axis=0)
    plt.fill_between(epoch_freqs, mean_data + yerr, mean_data - yerr, color=colors(i), alpha=.2)
    plt.plot(epoch_freqs, mean_data, '-o', lw=2, color=colors(i), markersize=7,
             label=('Cluster %d, N: %d ROIs' % (cl_type, len(np.where(prediction == cl_type)[0]))))

plt.title('Predicted clusters {fn}'.format(fn=flyID))
plt.xscale('log')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')
plt.legend()
plt.subplot(121)
mean_t = fly_data[flyID].mean(axis=0)
std_t = fly_data[flyID].std(axis=0)
ub = mean_t + std_t
lb = mean_t - std_t

plt.fill_between(epoch_freqs, ub, lb, color=color, alpha=.2)
plt.plot(epoch_freqs, mean_t, '-o', lw=2, color=color, markersize=7)
plt.legend()
plt.xscale('log')
plt.title('Mean responses')
plt.ylabel('Normalized dF/F')
plt.xlabel('Temporal Frequency (Hz)')
if 1:
    # Saving figure
    save_name = '{exp_t}_Tuning_hierarchical_cluster_{fn}'.format(exp_t=exp_t, fn=flyID)
    os.chdir(summary_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight')
plt.show()
