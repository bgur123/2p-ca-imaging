#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:26:53 2020

@author: burakgur
"""

fig1=ROI_mod.plot_RFs(rois_plot[40:], number=40, f_w =8,cmap='coolwarm',
                                center_plot = True, center_val = 0.95)
os.chdir(figure_save_dir)
fig1.savefig('RF_maps_center', bbox_inches='tight',
               transparent=False,dpi=300)


fig2=ROI_mod.plot_RFs(rois_plot[40:], number=40, f_w =8,cmap='coolwarm',
                                center_plot = False, center_val = 0.95)
os.chdir(figure_save_dir)
fig2.savefig('RF_maps_no_center', bbox_inches='tight',
               transparent=False,dpi=300)


fig3 = ROI_mod.plot_RF(rois_plot[np.random.randint(len(rois_plot))],
                       cmap1='coolwarm',cmap2='inferno',
                       center_plot = False, center_val = 0.95)
os.chdir(figure_save_dir)
fig3.savefig('ROI_example3', bbox_inches='tight',
               transparent=False,dpi=300)

fig3 = ROI_mod.plot_RF(rois_plot[40],
                       cmap1='coolwarm',cmap2='inferno',
                       center_plot = True, center_val = 0.95)
os.chdir(figure_save_dir)
fig3.savefig('ROI_example1', bbox_inches='tight',
               transparent=False,dpi=300)



fig4 = plot_RF_centers_on_screen(rois,prop='PD')
os.chdir(figure_save_dir)
fig4.savefig('ROI_center_screen', bbox_inches='tight',
               transparent=False,dpi=300)