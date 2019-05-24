#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script includes functions to calculate different error metrics.
@author: elynn
"""

import numpy as np
import pandas as pd
def calc_RMSE(forecast,observation):
    '''
    Parameters
    ----------
    forecast: np.array
        Forecasted values
    observation: np.array
        Observed values (i.e., truth)
    Returns
    -------
    rmse: float
        Root mean square error
    '''
    rmse = 0.
    if len(forecast) == len(observation):
        n = len(forecast)
        for i in range(n):
            rmse = rmse + 1./float(n) * (forecast[i]-observation[i])**2
    else:
        print('Error: forecast and observation do not have the same length')
    return np.sqrt(rmse)

def calc_CRMSE_BIAS(forecast,observation):
    '''
    Parameters
    ----------
    forecast: np.array
        Forecasted values
    observation: np.array
        Observed values (i.e., truth)
    Returns
    -------
    crmse: float
        Centered root mean square error
    bias: float
        Bias
    '''
    crmse = 0.
    if len(forecast) == len(observation):
        n = len(forecast)
        forecast_avg = np.average(forecast)
        obs_avg = np.average(observation)
        for i in range(n):
            crmse = crmse + 1./float(n) * ((forecast[i]-forecast_avg)-(observation[i]-obs_avg))**2
        bias = forecast_avg - obs_avg
    else:
        print('Error: forecast and observation do not have the same length')
    return np.sqrt(crmse), bias  

def post_process_stats(path):
    numOfAnalog = 10
    log = pd.read_csv('/mnt/lab_48tb1/database/Sc_group/github/Obs_AnEn/code/AnEn_log_'+str(numOfAnalog)+'days.csv',index_col=0,parse_dates=True)
    for i in range(1,3):
        output = open(path+'stats/Cloudy_test_'+str(i)+'_kt.csv','w')
        output.write('Date,AnEn mean RMSE,AnEn mean CRMSE,AnEn mean Bias,AnEn median RMSE,AnEn median CRMSE,AnEn median Bias,Persistence RMSE,Persistence CRMSE,Persistence Bias\n')
        selector = log['Cloudy test']==i
        current = log[selector]
        for j in range(len(current)):
            f = np.genfromtxt(path+'ts_kt/'+current.index[j].strftime('%Y%m%d')+'_kt.csv',delimiter=',')
            AnEn_mean_crmse, AnEn_mean_bias = calc_CRMSE_BIAS(f[:,1],f[:,0])
            AnEn_mean_rmse = calc_RMSE(f[:,1],f[:,0])
            AnEn_med_crmse, AnEn_med_bias = calc_CRMSE_BIAS(f[:,2],f[:,0])
            AnEn_med_rmse = calc_RMSE(f[:,2],f[:,0])
            persis_crmse, persis_bias = calc_CRMSE_BIAS(f[:,3],f[:,0])
            persis_rmse = calc_RMSE(f[:,3],f[:,0])
            output.write(current.index[j].strftime('%Y%m%d')+','+str(AnEn_mean_rmse)+','+str(AnEn_mean_crmse)+','+str(AnEn_mean_bias)+','+\
                        str(AnEn_med_rmse)+','+str(AnEn_med_crmse)+','+str(AnEn_med_bias)+','+\
                        str(persis_rmse)+','+str(persis_crmse)+','+str(persis_bias)+'\n')
        output.close()
    