#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:52:37 2019

@author: burakgur
Continuation for analysis and plotting of full field flash stimulus
"""

#%% Process data


# Getting fly means
fly_data = big_df.groupby(['ROI_type','flyID','toxin'],as_index=False).mean()

control_roi_data = big_df[big_df['toxin'] == 'No_toxin']
toxin_roi_data = big_df[big_df['toxin'] == 'Agi_200nM']

roi_num_control = len(np.unique(control_roi_data['ROI_ID']).tolist())
roi_num_toxin = len(np.unique(toxin_roi_data['ROI_ID']).tolist())

fly_num_control = len(np.unique(control_roi_data['flyID']).tolist())
fly_num_toxin = len(np.unique(toxin_roi_data['flyID']).tolist())

#%%White plots
sns.set(style="ticks", context="talk",rc={"font.size":14,"axes.titlesize":12,
                                          "axes.labelsize":14,'xtick.labelsize':12,
                                          'ytick.labelsize' : 12,
                                          'legend.fontsize':12})
plt.style.use("default")
sns.set_palette("Set2")
colors = sns.color_palette(palette='Set2', n_colors=2)

#%% Plot according to subcategories
plot_byCategory(big_df, 'trace', 'toxin', 'ROI_type', 
                    conditions, 5,title_string, save_dir = False,
                    plot_end_frame =100)



#%% Fly means plot
fig1, ax = plt.subplots(ncols=1,figsize=(7, 7), facecolor='w', edgecolor='k')
fig1.suptitle(title_string,fontsize=12)


sns.barplot(x="ROI_type", y="Max_resp",hue='toxin',data=fly_data,
             ax=ax,dodge=True,saturation=.7,ci='sd',hue_order = ['No_toxin','Agi_200nM'])

ax.set(ylabel=r'Max $\Delta$F/F', xlabel='ROI type')
ax.set_title('Fly means')
ax.set_title('No toxin N: %d(%d) -- Agitoxin N: %d(%d)'% (fly_num_control,
                                                roi_num_control,fly_num_toxin,
                                                roi_num_toxin))