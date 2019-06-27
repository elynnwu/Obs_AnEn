#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:26:14 2018

@author: monica
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import error_calc as error_m
import plot_methods as AnEnPlot
import scipy.stats
# Read 4am stats table
data_source='../data/DataForAnEn_20142017_v5_withSat.csv'
all_data_4am = pd.read_csv(data_source,index_col=0,parse_dates=True)
all_data_4am = all_data_4am[all_data_4am.index.hour==12] #only 12Z
'''Remove two failed days'''
all_data_4am = all_data_4am.drop(pd.Timestamp('2015-08-22 12:00:00'))
all_data_4am = all_data_4am.drop(pd.Timestamp('2016-05-06 12:00:00'))

test_data_4am = all_data_4am.loc['2014':'2017']
numAnalogs = [10] # confirmed

'''Read NKX GHI data for saving data later'''
NKX = pd.read_csv('../data/NKX_GHI.csv',index_col=0,parse_dates=True)
NKX.columns = ['GHI','CS_GHI']

'''Read cloudy weights sensitivity test'''
weights_table = pd.read_csv('../data/cloudy_weights_table.csv')

for wc in range(1,22):
    '''First, make folders needed to save data'''
    try:
        # os.makedirs('/mnt/lab_48tb1/database/Sc_group/AnEn/Results/weights_sens/Sens'+str(wc))
        os.makedirs('/mnt/lab_48tb1/database/Sc_group/AnEn/Results/weights_sens/Sens'+str(wc)+'/ts_kt')
        # os.makedirs('/mnt/lab_48tb1/database/Sc_group/AnEn/Results/weights_sens/Sens'+str(wc)+'/stats')
    except OSError:
        if not os.path.isdir('/mnt/lab_48tb1/database/Sc_group/AnEn/Results/weights_sens/Sens'+str(wc)):
            raise
    current_path = '/mnt/lab_48tb1/database/Sc_group/AnEn/Results/weights_sens/Sens'+str(wc)+'/'
    # log_file = open('AnEn_log_'+str(numAnalogs[na])+'days.txt','w')
    # Randomly pick a day in the testing dataset
    n_test=test_data_4am.shape[0]; #number of available days
    # i_test=int(np.floor(np.random.random()*n_test))
    for i_test in range(n_test):
        train_data_4am = all_data_4am.loc['2014':'2017']
        # Is that day cloudy or clear?
        if np.isnan(test_data_4am.n_clouds[i_test]):
            cloudy_test = 0 # clear day
        elif test_data_4am.decoupled_dtv[i_test]==1:
            cloudy_test = 1 # cloudy and decoupled
        else:
            cloudy_test = 2 # cloudy and well-mixed
        if cloudy_test > 0: # doing sensitivity for cloudy cases ONLY
            # leave one out approach: drop the selected day
            train_data_4am = train_data_4am.drop(test_data_4am.index[i_test])

            ###############################################################################
            #                 Create arrays for cloudy-wellMixed, cloudy-decoupled, and non cloudy days
            ###############################################################################
            cloudy_vars = ['z_inv_base','thetaL_BL','thetaL_jump','qT_3km','LCL_srf','instant_u','instant_v','SST_instant','Ocean_lcc','Zone1_lcc','Zone2_lcc','Zone3_lcc']
            clear_vars = ['z_inv_base','Tsrf','Tdew','precipitable_water','DZ','instant_u','instant_v','SST_instant']
            # weights = np.ones(len(cloudy_vars))*25
            # weights[0] = 50; weights[2] = 50; weights[8] = 50
            # weights = weights/np.sum(weights)
            weights = weights_table['Sens'+str(wc)].values

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
                for dc in range(n_train_cloudy_decoupled):
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

            print ('Sens'+str(wc)+' Current test day: ', test_data_4am.index[i_test])
        #    print 'Cloudy day: ', cloudy_test
            # log_file.write(str(test_data_4am.index[i_test])+','+str(cloudy_test)+',')
            non_nan = ~np.isnan(d2['d2_total'])
            top5 = np.argsort(d2['d2_total'][:,0])[0:10]

            if cloudy_test == 2:
        #        print 'The 5 most similar days: ', cloudy_wellMixed[cloudy_vars[0]].index[top5]
                top5 = cloudy_wellMixed[cloudy_vars[0]].index[top5]
            elif cloudy_test == 1:
        #        print 'The 5 most similar days: ', cloudy_decoupled[cloudy_vars[0]].index[top5]
                top5 = cloudy_decoupled[cloudy_vars[0]].index[top5]
            else:
        #        print 'The 5 most similar days: ', clear[clear_vars[0]].index[top5]
                top5 = clear[clear_vars[0]].index[top5]
            # for i in range(len(top5)):
            #     log_file.write(str(top5[i])+',')

            obs, AnEn, persistence = AnEnPlot.plot_results(test_data_4am,i_test,NKX,current,top5)
            # plt.savefig(current_path+test_data_4am.index[i_test].strftime('%Y%m%d')+'.png',dpi=200,bbox_inches='tight')
            # plt.close()
            RMSE =  error_m.calc_RMSE(np.average(AnEn,axis=0),obs)
            CRMSE, BIAS = error_m.calc_CRMSE_BIAS(np.average(AnEn,axis=0),obs)
            Rs, pval = scipy.stats.spearmanr(np.average(AnEn,axis=0),obs)
            # log_file.write(str(RMSE)+','+str(CRMSE)+','+str(BIAS)+','+str(Rs)+','+str(pval)+'\n')
            output = np.zeros((48,4))
            output[:,0] = obs
            output[:,1] = np.average(AnEn,axis=0)
            output[:,2] = np.median(AnEn,axis=0)
            output[:,3] = persistence
            np.savetxt(current_path+'ts_kt/'+test_data_4am.index[i_test].strftime('%Y%m%d')+'_kt.csv',output,delimiter=',',fmt='%s')
    # log_file.close()
    error_m.post_process_stats(current_path)
