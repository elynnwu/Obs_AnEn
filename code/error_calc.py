#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script includes functions to calculate different error metrics.
@author: elynn
"""

import numpy as np

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
        print 'Error: forecast and observation do not have the same length'
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
        print 'Error: forecast and observation do not have the same length'
    return np.sqrt(crmse), bias  

