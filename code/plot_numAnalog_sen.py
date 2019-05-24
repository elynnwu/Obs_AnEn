#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:06:03 2019

@author: elynn
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="ticks",font_scale=1.1)

numAna = [3,5,10,15,20,25]
ind = np.arange(len(numAna))
width = 0.25  # the width of the bars
blue = (0.0, 0.4470588235294118, 0.6980392156862745)
orange = (0.8352941176470589, 0.3686274509803922, 0.0)
with plt.rc_context({'axes.autolimit_mode': 'round_numbers'}):
    fig, axes = plt.subplots(3, 3, figsize=(10,7), sharex = True)
data = np.zeros((len(numAna),3,9))
for i in range(len(numAna)):
    path = '/mnt/lab_48tb1/database/Sc_group/AnEn/Results/zi_x2_dthetaL_x2_oceanLCC_x2_others_x1_'+str(numAna[i])+'analogs/stats/'
    for cc in range(3):
        f = pd.read_csv(path+'Cloudy_test_'+str(cc)+'.csv').mean()
        data[i,cc,:] = f.values[1:]
        # axes[cc,0].bar(i-width,f.values[1],width,color=blue)
        # axes[cc,0].bar(i,f.values[4],width,color='#3FC2A5')
        # axes[cc,0].bar(i+width,f.values[7],width,color=orange)
        # axes[cc,1].bar(i-width,f.values[2],width,color=blue)
        # axes[cc,1].bar(i,f.values[5],width,color='#3FC2A5')
        # axes[cc,1].bar(i+width,f.values[8],width,color=orange)
        # axes[cc,2].bar(i-width,f.values[3],width,color=blue)
        # axes[cc,2].bar(i,f.values[6],width,color='#3FC2A5')
        # axes[cc,2].bar(i+width,f.values[9],width,color=orange)        
for cc in range(3):
    axes[cc,0].plot(ind,data[:,cc,0],'-s',color=blue,markersize=7, markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor=blue,label='AnEn mean')
    axes[cc,0].plot(ind,data[:,cc,3],'-o',color='#3FC2A5',markersize=7, markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor='#3FC2A5',label='AnEn median')
    axes[cc,0].plot(ind,data[:,cc,6],'-^',color=orange,markersize=7,markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor=orange,label='Persistance')
    
    axes[cc,1].plot(ind,data[:,cc,1],'-s',color=blue,markersize=7, markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor=blue,label='AnEn mean')
    axes[cc,1].plot(ind,data[:,cc,4],'-o',color='#3FC2A5',markersize=7, markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor='#3FC2A5',label='AnEn median')
    axes[cc,1].plot(ind,data[:,cc,7],'-^',color=orange,markersize=7,markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor=orange,label='Persistance')
    
    axes[cc,2].plot(ind,data[:,cc,2],'-s',color=blue,markersize=7, markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor=blue)
    axes[cc,2].plot(ind,data[:,cc,5],'-o',color='#3FC2A5',markersize=7, markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor='#3FC2A5')
    axes[cc,2].plot(ind,data[:,cc,8],'-^',color=orange,markersize=7,markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor=orange)
    
for i in range(3):
    axes[i,0].set_ylabel(r'RMSE [$Wm^{-2}$]')
    axes[i,0].set_xlim([-0.5,5.5])
    axes[i,0].set_xticks([0,1,2,3,4,5])
    axes[i,0].set_xticklabels(('5','10'))
    axes[i,1].set_ylabel(r'CRMSE [$Wm^{-2}$]')
    axes[i,1].set_xlim([-0.5,5.5])
    axes[i,1].set_xticks([0,1,2,3,4,5])
    axes[i,1].set_xticklabels(('5','10'))
    axes[i,2].set_ylabel(r'Bias [$Wm^{-2}$]')
    axes[i,2].set_xlim([-0.5,5.5])
    axes[2,i].set_xlabel('Number of analogs')
    axes[i,2].set_xticks([0,1,2,3,4,5])
    axes[i,2].set_xticklabels(('3','5','10','15','20','25'))
axes[0,0].set_ylabel('Clear\nRMSE [$Wm^{-2}$]')
axes[1,0].set_ylabel('Cloudy decoupled\nRMSE [$Wm^{-2}$]')
axes[2,0].set_ylabel('Cloudy well-mixed\nRMSE [$Wm^{-2}$]')
axes[1,1].legend(loc=0)
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig('/mnt/lab_48tb1/database/Sc_group/github/Obs_AnEn/fig/numberOfAnalog_stats.pdf',bbox_inches='tight')
