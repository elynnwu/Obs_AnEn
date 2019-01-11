#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:42:13 2019

@author: elynn
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks",font_scale=1.2)
def plot_results(test_data_4am,i_test,NKX,current,top5):
    plt.figure(figsize=(7,5))
    persistence = test_data_4am.index[i_test-1].strftime('%Y-%m-%d')
    current = test_data_4am.index[i_test].strftime('%Y-%m-%d')
    tindex = NKX[current].index.hour+NKX[current].index.minute/60.
    plt.plot(tindex,NKX[current].GHI,marker='o',\
             markersize=7,color='k',markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor='k',label='Obs')
    plt.plot(tindex,NKX[current].CS_GHI,'--',color='k',label='Clearsky')
    orange = (0.8352941176470589, 0.3686274509803922, 0.0)
    plt.plot(tindex,NKX[persistence].GHI,marker='^',\
             markersize=7,color=orange,markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor=orange, label='Persistence')
    obs = NKX[current].GHI.values
    AnEn = np.zeros((len(top5),len(obs)))
    flag = True
    for i in range(len(top5)):
        current_top5 = top5[i].strftime('%Y-%m-%d')
        tindex = NKX[current_top5].index.hour+NKX[current_top5].index.minute/60.
        kt = NKX[current_top5].GHI/NKX[current_top5].CS_GHI
        forecast_GHI = kt.values*NKX[current].CS_GHI.values
        forecast_GHI = np.nan_to_num(forecast_GHI)
        if flag:
            plt.plot(tindex,forecast_GHI,color='gray',alpha=0.8, label='AnEn individual')
            flag = False
        else:
            plt.plot(tindex,forecast_GHI,color='gray',alpha=0.8)
        AnEn[i,:] = forecast_GHI
    blue = (0.0, 0.4470588235294118, 0.6980392156862745)
    plt.plot(tindex,np.average(AnEn,axis=0),color=blue, \
             marker='s',markersize=7, markerfacecolor='none', \
             markeredgewidth=1, markeredgecolor=blue, label='AnEn mean')
    plt.xlim([4,20])
    plt.ylim([0,1100])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, columnspacing = 1)
    plt.xticks(np.arange(4,21,2))
    plt.xlabel('Time [hr]')
    plt.ylabel(r'GHI [$W/m^2$]')
    plt.tight_layout()
    return obs, AnEn