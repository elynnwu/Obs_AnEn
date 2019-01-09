#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:26:14 2018

@author: monica
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import error_calc as error_m
import plot_methods as AnEnPlot
import scipy.stats
# Read 4am stats table
#data_source='../data/DataForAnEn_20142017.csv'
data_source='../data/DataForAnEn_20142017_v3.csv'
all_data_4am = pd.read_csv(data_source,index_col=0,parse_dates=True)
train_data_4am = all_data_4am.loc['2014':'2016']
test_data_4am = all_data_4am.loc['2017']

###############################################################################
#                 Create arrays for cloudy and non cloudy days
###############################################################################
cloudy_vars = ['z_inv_base','thetaL_BL','thetaL_jump','qT_3km','LCL_srf','instant_u','instant_v','SST_instant','Ocean_lcc','Zone1_lcc','Zone2_lcc','Zone3_lcc']
clear_vars = ['z_inv_base','Tsrf','Tdew','precipitable_water','DZ','instant_u','instant_v','SST_instant']
weights = np.ones(len(cloudy_vars))*0.04
weights[0] = 0.3; weights[2] = 0.3; weights[8] = 0.5
weights = weights/np.sum(weights)

cloudy_days=(~np.isnan(train_data_4am.n_clouds))
train_cloudy = train_data_4am[cloudy_days][cloudy_vars].dropna()
n_train_cloudy=len(train_cloudy)

# EW: we should keep these in the dictionary, fast and easy access
cloudy = {}
for i in range(len(cloudy_vars)):
    current = train_cloudy[cloudy_vars[i]]
    cloudy[cloudy_vars[i]+'_mean'] = np.mean(current)
    cloudy[cloudy_vars[i]+'_std'] = np.std(current)
    cloudy[cloudy_vars[i]] = (current-cloudy[cloudy_vars[i]+'_mean'])/cloudy[cloudy_vars[i]+'_std']
    cloudy[cloudy_vars[i]+'_weight'] = weights[i]#1./len(cloudy_vars) # equal weights for now


# Clear days
clear_days=(np.isnan(train_data_4am.n_clouds))
train_clear = train_data_4am[clear_days][clear_vars].dropna()
n_train_clear=len(train_clear)

clear = {}
for i in range(len(clear_vars)):
    current = train_clear[clear_vars[i]]
    clear[clear_vars[i]+'_mean'] = np.mean(current)
    clear[clear_vars[i]+'_std'] = np.std(current)
    clear[clear_vars[i]] = (current-clear[clear_vars[i]+'_mean'])/clear[clear_vars[i]+'_std']
    clear[clear_vars[i]+'_weight'] = 1./len(clear_vars) # equal weights for now



###############################################################################
#                      Set case and compute distance
###############################################################################

# Randomly pick a day in 2017
n_test=test_data_4am.shape[0]; #number of available days
i_test=int(np.floor(np.random.random()*n_test))

# Is that day cloudy or clear?
cloudy_test=~np.isnan(test_data_4am.n_clouds[i_test])


# Compute L2 distance
if cloudy_test: # Cloudy days
    # EW: using dictionary
    d2 = {}
    for i in range(len(cloudy_vars)):
        d2[cloudy_vars[i]+'_d2'] = np.empty((n_train_cloudy,1))
        d2[cloudy_vars[i]+'_d2'][:] = np.nan
    d2['d2_total'] = np.zeros((n_train_cloudy,1))
    for dc in range(n_train_cloudy):
        for i in range(len(cloudy_vars)):
            current = (test_data_4am[cloudy_vars[i]].iloc[i_test]-cloudy[cloudy_vars[i]+'_mean'])/cloudy[cloudy_vars[i]+'_std']
            d2[cloudy_vars[i]+'_d2'][dc] = (current - cloudy[cloudy_vars[i]].iloc[dc])**2
            d2['d2_total'][dc] = d2['d2_total'][dc]+d2[cloudy_vars[i]+'_d2'][dc]
        
else: # Clear days
    # EW: using dictionary
    d2 = {}
    for i in range(len(clear_vars)):
        d2[clear_vars[i]+'_d2'] = np.empty((n_train_clear,1))
        d2[clear_vars[i]+'_d2'][:] = np.nan
    d2['d2_total'] = np.zeros((n_train_clear,1))
    for dc in range(n_train_clear):
        for i in range(len(clear_vars)):
            current = (test_data_4am[clear_vars[i]].iloc[i_test]-clear[clear_vars[i]+'_mean'])/clear[clear_vars[i]+'_std']
            d2[clear_vars[i]+'_d2'][dc] = (current - clear[clear_vars[i]].iloc[dc])**2
            d2['d2_total'][dc] = d2['d2_total'][dc]+d2[clear_vars[i]+'_d2'][dc]

print 'Current test day: ', test_data_4am.index[i_test]
print 'Cloudy day: ', cloudy_test
non_nan = ~np.isnan(d2['d2_total'])
top5 = np.argsort(d2['d2_total'][:,0])[0:5]
if cloudy_test:
    print 'The 5 most similar days: ', cloudy[cloudy_vars[0]].index[top5]
    top5 = cloudy[cloudy_vars[0]].index[top5]
else:
    print 'The 5 most similar days: ', clear[clear_vars[0]].index[top5]
    top5 = clear[clear_vars[0]].index[top5]

#NKX = pd.read_csv('../data/NKX_GHI.csv',skiprows=1,index_col=0,parse_dates=True)
#NKX = NKX[[NKX.columns[0],NKX.columns[5]]]
#NKX.columns = ['GHI','CS_GHI']
obs, AnEn = AnEnPlot.plot_results(test_data_4am,i_test,NKX,current,top5)
print 'RMSE: ', error_m.calc_RMSE(np.average(AnEn,axis=0),obs)
print 'CRMSE, BIAS: ', error_m.calc_CRMSE_BIAS(np.average(AnEn,axis=0),obs)
print 'Rs: ', scipy.stats.spearmanr(np.average(AnEn,axis=0),obs)