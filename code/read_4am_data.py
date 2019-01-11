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
data_source='../data/DataForAnEn_20142017_v3.csv'
all_data_4am = pd.read_csv(data_source,index_col=0,parse_dates=True)
all_data_4am = all_data_4am[all_data_4am.index.hour==12]
train_data_4am = all_data_4am.loc['2014':'2017']
test_data_4am = all_data_4am.loc['2017']

# Randomly pick a day in the testing dataset
n_test=test_data_4am.shape[0]; #number of available days
i_test=int(np.floor(np.random.random()*n_test))
#i_test = 87 #using this day for example

# Is that day cloudy or clear?
if np.isnan(test_data_4am.n_clouds[i_test]):
    cloudy_test = 0 # clear day
elif test_data_4am.decoupled_dtv[i_test]==1:
    cloudy_test = 1 # cloudy and decoupled
else:
    cloudy_test = 2 # cloudy and well-mixed

# leave one out approach: drop the selected day
train_data_4am = train_data_4am.drop(test_data_4am.index[i_test]) 

###############################################################################
#                 Create arrays for cloudy-wellMixed, cloudy-decoupled, and non cloudy days
###############################################################################
cloudy_vars = ['z_inv_base','thetaL_BL','thetaL_jump','qT_3km','LCL_srf','instant_u','instant_v','SST_instant','Ocean_lcc','Zone1_lcc','Zone2_lcc','Zone3_lcc']
clear_vars = ['z_inv_base','Tsrf','Tdew','precipitable_water','DZ','instant_u','instant_v','SST_instant']
weights = np.ones(len(cloudy_vars))*25
weights[0] = 50; weights[2] = 50; weights[8] = 50
cloudy_weights = weights/np.sum(weights)

'''Criteria for each class'''
cloudy_decoupled_days = (train_data_4am['decoupled_dtv']==1)
clear_days = (np.isnan(train_data_4am.n_clouds))
cloudy_wellMixed_days = (train_data_4am['decoupled_dtv']!=1) & (~np.isnan(train_data_4am.n_clouds))

'''Training days for each class'''
train_cloudy_wellMixed = train_data_4am[cloudy_wellMixed_days][cloudy_vars]
train_cloudy_decoupled = train_data_4am[cloudy_decoupled_days][cloudy_vars]
train_clear = train_data_4am[clear_days][clear_vars]

'''Number of days for each class'''
n_train_cloudy_wellMixed = len(train_cloudy_wellMixed)
n_train_cloudy_decoupled = len(train_cloudy_decoupled)
n_train_clear = len(train_clear)

# cloudy and well-mixed days
cloudy_wellMixed = {}
for i in range(len(cloudy_vars)):
    current = train_cloudy_wellMixed[cloudy_vars[i]]
    cloudy_wellMixed[cloudy_vars[i]+'_mean'] = np.mean(current)
    cloudy_wellMixed[cloudy_vars[i]+'_std'] = np.std(current)
    cloudy_wellMixed[cloudy_vars[i]] = (current-cloudy_wellMixed[cloudy_vars[i]+'_mean'])/cloudy_wellMixed[cloudy_vars[i]+'_std']
    cloudy_wellMixed[cloudy_vars[i]+'_weight'] = weights[i]

# cloudy and decoupled days
cloudy_decoupled = {}
for i in range(len(cloudy_vars)):
    current = train_cloudy_decoupled[cloudy_vars[i]]
    cloudy_decoupled[cloudy_vars[i]+'_mean'] = np.mean(current)
    cloudy_decoupled[cloudy_vars[i]+'_std'] = np.std(current)
    cloudy_decoupled[cloudy_vars[i]] = (current-cloudy_decoupled[cloudy_vars[i]+'_mean'])/cloudy_decoupled[cloudy_vars[i]+'_std']
    cloudy_decoupled[cloudy_vars[i]+'_weight'] = weights[i]


# clear days - when n_clouds is nan
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

# Compute L2 distance
if cloudy_test == 2: # Cloudy and well-mixed
    d2 = {}
    for i in range(len(cloudy_vars)):
        d2[cloudy_vars[i]+'_d2'] = np.empty((n_train_cloudy_wellMixed,1))
        d2[cloudy_vars[i]+'_d2'][:] = np.nan
    d2['d2_total'] = np.zeros((n_train_cloudy_wellMixed,1))
    for dc in range(n_train_cloudy_wellMixed):
        for i in range(len(cloudy_vars)):
            current = (test_data_4am[cloudy_vars[i]].iloc[i_test]-cloudy_wellMixed[cloudy_vars[i]+'_mean'])/cloudy_wellMixed[cloudy_vars[i]+'_std']
            d2[cloudy_vars[i]+'_d2'][dc] = cloudy_wellMixed[cloudy_vars[i]+'_weight'] * (current - cloudy_wellMixed[cloudy_vars[i]].iloc[dc])**2
            d2['d2_total'][dc] = d2['d2_total'][dc]+d2[cloudy_vars[i]+'_d2'][dc]
elif cloudy_test == 1: # Cloudy and decoupled
    d2 = {}
    for i in range(len(cloudy_vars)):
        d2[cloudy_vars[i]+'_d2'] = np.empty((n_train_cloudy_decoupled,1))
        d2[cloudy_vars[i]+'_d2'][:] = np.nan
    d2['d2_total'] = np.zeros((n_train_cloudy_decoupled,1))
    for dc in range(n_train_cloudy_wellMixed):
        for i in range(len(cloudy_vars)):
            current = (test_data_4am[cloudy_vars[i]].iloc[i_test]-cloudy_decoupled[cloudy_vars[i]+'_mean'])/cloudy_decoupled[cloudy_vars[i]+'_std']
            d2[cloudy_vars[i]+'_d2'][dc] = cloudy_decoupled[cloudy_vars[i]+'_weight'] * (current - cloudy_decoupled[cloudy_vars[i]].iloc[dc])**2
            d2['d2_total'][dc] = d2['d2_total'][dc]+d2[cloudy_vars[i]+'_d2'][dc]    
else: # Clear days
    d2 = {}
    for i in range(len(clear_vars)):
        d2[clear_vars[i]+'_d2'] = np.empty((n_train_clear,1))
        d2[clear_vars[i]+'_d2'][:] = np.nan
    d2['d2_total'] = np.zeros((n_train_clear,1))
    for dc in range(n_train_clear):
        for i in range(len(clear_vars)):
            current = (test_data_4am[clear_vars[i]].iloc[i_test]-clear[clear_vars[i]+'_mean'])/clear[clear_vars[i]+'_std']
            d2[clear_vars[i]+'_d2'][dc] = clear[clear_vars[i]+'_weight'] * (current - clear[clear_vars[i]].iloc[dc])**2
            d2['d2_total'][dc] = d2['d2_total'][dc]+d2[clear_vars[i]+'_d2'][dc]

print 'Current test day: ', test_data_4am.index[i_test]
print 'Cloudy day: ', cloudy_test
non_nan = ~np.isnan(d2['d2_total'])
top5 = np.argsort(d2['d2_total'][:,0])[0:5]
if cloudy_test == 2:
    print 'The 5 most similar days: ', cloudy_wellMixed[cloudy_vars[0]].index[top5]
    top5 = cloudy_wellMixed[cloudy_vars[0]].index[top5]
elif cloudy_test == 1:
    print 'The 5 most similar days: ', cloudy_decoupled[cloudy_vars[0]].index[top5]
    top5 = cloudy_decoupled[cloudy_vars[0]].index[top5]
else:
    print 'The 5 most similar days: ', clear[clear_vars[0]].index[top5]
    top5 = clear[clear_vars[0]].index[top5]

#NKX = pd.read_csv('../data/NKX_GHI.csv',skiprows=1,index_col=0,parse_dates=True)
#NKX = NKX[[NKX.columns[0],NKX.columns[5]]]
#NKX.columns = ['GHI','CS_GHI']
obs, AnEn = AnEnPlot.plot_results(test_data_4am,i_test,NKX,current,top5)
plt.savefig('/mnt/lab_45d1/database/Sc_group/AnEn/Results/zi_x2_dthetaL_x2_oceanLCC_x2_others_x1/'+test_data_4am.index[i_test].strftime('%Y%m%d')+'.png',dpi=200,bbox_inches='tight')
plt.close()
print 'RMSE: ', error_m.calc_RMSE(np.average(AnEn,axis=0),obs)
print 'CRMSE, BIAS: ', error_m.calc_CRMSE_BIAS(np.average(AnEn,axis=0),obs)
print 'Rs: ', scipy.stats.spearmanr(np.average(AnEn,axis=0),obs)