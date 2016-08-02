# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:16:29 2016

@author: anna
"""


'''The function is desighned to analyse the resourse 
    utilization based on clusterization and linear models of trend.
    Use the R function to detect last period of homogeneous structure
'''

import rpy2.interactive as r
import rpy2.interactive.packages # this can take few seconds
rlib = r.packages.packages
r.packages.importr("utils")
#rlib.utils.install_packages("devtools")
from rpy2.robjects.packages import importr
devtools=importr('devtools')
devtools.install_github(repo="BreakoutDetection", username="twitter")
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import IPython
from IPython.display import display, HTML

""" median filtering """
def get_median_filtered(signal, threshold=3):
    """
    signal: is numpy array-like
    returns: signal, numpy array
    """
    difference = np.abs(signal - np.median(signal))
    #print(np.median(signal))
    #print(np.unique(difference))
    median_difference = np.median(difference)
    #s = [0 if median_difference == 0 else dd / float(median_difference) for dd in difference]
    s = [dd / float(median_difference) for dd in difference]

    mask = [ss > threshold for ss in s]
    #print(np.unique(mask))
    if np.median(signal)==0:
        signal = [np.average(signal) if mask[i] else signal[i] for i in range(len(signal))]
    else:
        signal = [np.median(signal) if mask[i] else signal[i] for i in range(len(signal))]
    return signal

def get_avg_filtered(signal, threshold=3):
    """
    signal: is numpy array-like
    returns: signal, numpy array
    """
    difference = np.abs(signal - np.median(signal))
    median_difference = np.average(difference)
    s = [0 if median_difference == 0 else dd / float(median_difference) for dd in difference]
    mask = [ss > threshold for ss in s]
    signal = [np.average(signal) if mask[i] else signal[i] for i in range(len(signal))]
    return signal


def get_util(test, index, plot_clust, min_clust_size=0.1):
    #print(min_clust_size)
    clust_n=7
    if len(set(test)) < clust_n:
        print('Len of unique data less than number of clusters')
        test_h = test
        inds = range(len(test_h))
    else:
        km = KMeans(clust_n)
        #print("Get utilized")
        x=StandardScaler().fit_transform(np.array(test).reshape(-1,1))
        km.fit(np.array(x).reshape(len(x), 1), clust_n)
        clust = km.predict(np.array(x).reshape(len(test), 1))
        #print("Sihoette distance for "+str(clust_n)+" clusters is")
        #print(sklearn.metrics.silhouette_score(np.array(x).reshape(len(x), 1), clust))
        val = [np.argmax(km.cluster_centers_)]
        inds = [int(x) for x in range(len(clust)) if clust[x] in val]
        test_h = [test[i] for i in inds]#test[inds]
        clust_cent = km.cluster_centers_

        # while len(test_h) < min_clust_size * len(test):
        #    if len(np.unique(clust_cent[0])) == 1  and (np.unique(clust_cent[0])[0] == 0 or np.unique(clust_cent[0])[0] < 0):
        #        test_h = test
        #        inds = range(len(test_h))
        #        break
        #    clust_cent[val] = 0
        #    val.append(np.argmax(clust_cent))
        #    inds = [int(x) for x in range(len(clust)) if clust[x] in val]
        #    test_h = [test[i] for i in inds] 
        while len(test_h) < min_clust_size * len(test):
            #print('Length of last cluster is '+str(len(test_h)))
            #print("High load data length "+str(len(test_h)))
            #print(type(np.ravel(clust_cent)))
            #print(np.ravel(clust_cent))
            #print(((np.ravel(clust_cent))<=0).all())
            #print(len(test_h))
            if ((np.ravel(clust_cent))<=0).all():
                test_h = test
                inds = range(len(test_h))
                break
            clust_cent[val] = 0
            val.append(np.argmax(clust_cent))
            inds = [int(x) for x in range(len(clust)) if clust[x] in val]
            test_h = [test[i] for i in inds] 
            # print(len(test_h))
    if plot_clust:
        plt.figure()
        plt.scatter(index,test)
        #plt.plot(test)
        plt.scatter(x=[index[i] for i in inds], y=test_h, color='red')
        plt.show()
        plt.clf()
    
    test_h = get_median_filtered(test_h)
    #print(test_h)
    #print("Max is")
    #print(max(test_h))
    return max(test_h)
 
def last_period_forecast(data, min_size, min_clust_size=0.1):
    BreakoutDetection = importr('BreakoutDetection')
    base = importr('base')
    data2 = [x for x in data]
    data3 = data2
    points = []
    #print("Length of input data "+str(len(data3)))
    #print("min size"+str(min_size))
    if len(data)<=min_size:
        test=data
    else:
        while len(data3) > min_size:
            breaks = BreakoutDetection.breakout(base.as_numeric(data3), min_size)
            #print("Length of data is "+str(len(data3)))
            #print(breaks[0])
            if int(np.asarray(breaks[0])[0]) == 0:
                if len(points)==0:
                    points=[0]
                break
            if len(points) > 0:
                points.append(int(np.asarray(breaks[0])[0]) + points[len(points) - 1])
            else:
                points = [int(np.asarray(breaks[0])[0])]
            data3 = data3[int(np.asarray(breaks[0])[0]):]

        #print(points)
        #print(len(data))
        test = data[points[len(points) - 1]:]
        i = 1
        while len(test) < min_size:
            test = data[points[len(points) - i]:]
            i=i+1
        
    clust_n=6

    if len(set(test))<clust_n:
        test_h=test
        inds=range(len(test_h))
    else:
        km = KMeans(clust_n)
        #print("Forecast")
        x=StandardScaler().fit_transform(test.reshape(-1,1))
        km.fit(np.array(x).reshape(len(x), 1), clust_n)
        clust = km.predict(np.array(x).reshape(len(x), 1))
        #print(km.cluster_centers_)
        val = [np.argmax(km.cluster_centers_)]
        # print(val)
        inds = [int(x) for x in range(len(clust)) if clust[x] in val]
        test_h = [test[i] for i in inds]  # test[inds]
        clust_cent = km.cluster_centers_
        
        #print(len(test_h))

        while len(test_h) < min_clust_size * len(test):
            #if any(np.ravel(clust_cent))<=0:
            if ((np.ravel(clust_cent))<=0).all():
                test_h = test
                inds = range(len(test_h))
                break
            clust_cent[val] = 0
            val.append(np.argmax(clust_cent))
            inds = [int(x) for x in range(len(clust)) if clust[x] in val]
            test_h = test[inds]
            # print(len(test_h))
        #print(len(test_h))
    test_h = get_median_filtered(test_h)
    #test_h=get_avg_filtered(test_h)
    #print("Test H is calculated")

    lr1 = LinearRegression()
    # if param == 'used_cpu':
    #    test_h = [np.float16(x) for x in test_h]
    #print('Last index of high cluster '+str(inds[len(inds)-1]))
    #print('Test length '+str(len(test)))
    if(inds[len(inds)-1]<0.5*len(test)):
        test_h=test
        inds=range(len(test))
        #print('Test_h is changes to Test')
    
    #print(inds)
    #print(test_h)
    #print(len(test))
    lr1.fit(np.array(inds).reshape(len(inds), 1), np.array(test_h).reshape(len(test_h), 1))
    # predict for 1 week L =2016
    X = np.array(range(max(inds), (max(inds) + 1008)))
    res = lr1.predict(X.reshape(len(X), 1))
    res = [x for x in res.ravel()]
    
    lr2 = LinearRegression()
    data_ind=range(len(test))
    lr2.fit(np.array(data_ind).reshape(len(data_ind), 1), np.array(test).reshape(len(test), 1))
    X = np.array(range(max(data_ind), (max(data_ind) + 1008)))
    pred_data = lr2.predict(X.reshape(len(X), 1))
    pred_data = [x for x in pred_data.ravel()]
    
    res=[max(res[i],pred_data[i]) for i in range(len(res))]
    if len(points)>0:
        return res, np.std(test_h), points[len(points) - 1]
    else:
        return res, np.std(test_h), len(data)


def forecast_on_sar(host, user, min_size=566, path=os.getcwd(), start=0, end=0, plot_clust=True):
    #ID = pd.DataFrame({'kbmemused': "mem_data", "used_cpu": "cpu_data", "bread/s": "io_data",
    #                   "bwrtn/s": "io_data"}, index=range(1))
    ID = pd.DataFrame({'kbmemused': "mem_data", "used_cpu": "cpu_data", "bread/s": "io_data",
                       "bwrtn/s": "io_data"}, index=range(1))
    npl = 221
    my_dpi = 80
    #fig1 = plt.figure()
    fig1 = plt.figure(0, figsize=(2000 / my_dpi, 900 / my_dpi), dpi=my_dpi)
    ncores=0
    ram=0
    D=pd.DataFrame()
    # for input_data in ID:
    for param in ID.columns:
        #print(param)
        input_data = ID[param][0]
        mem = pd.read_csv(path+host+"/" + input_data + ".csv", index_col=False)
        if start!="":
            mem=mem[mem.time>start]
        if end!="":
            mem=mem[mem.time<end]
        if len(mem)<10:
            break
        #print(mem.head())
        #print(input_data)
        if input_data=="cpu_data":
            ncores=(len(pd.unique(mem['CPU']))-1)
            mem2=mem[mem['CPU']!="all"]
            mem2.index = pd.DatetimeIndex(mem2.time)
            mem2 = mem2.sort_index()
            dm=mem2.groupby(mem2.index)[param].sum()
            data = dm.values
            min_clust_size=0.01
            util = get_util([x for x in data], dm.index, plot_clust, min_clust_size)
            mem=mem[mem['CPU']=="all"]

            #print("CPU Cores " + str(ncores))
        elif input_data=="mem_data":
            ram=np.float32(mem['kbmemused'].values[0]+mem['kbmemfree'].values[0])
            
            mem['real_kbfree']=mem['kbmemfree']+mem['kbcached']
            mem['real_used']=[(ram-x) for x in mem['real_kbfree']]
            mem.to_csv(path+host+"/memory_calculated.csv")
            param='real_used'
            mem.index = pd.DatetimeIndex(mem.time)
            mem = mem.sort_index()
            dm = mem[param]
            data = dm.values
            util = get_util([x for x in data], dm.index, plot_clust)
            ram=np.ceil(ram/1024/1024)
            D = D.append(pd.DataFrame({"Resource": input_data, "Capacity": ram, "Param":param, "Used": np.ceil(util/1024/1024), "Details":"All data in GB"}, index=[host]))
            #param='kbmemused'
            param='real_used'
            dm = mem[param]
            #dm.index = pd.DatetimeIndex(dm.time)
            dm = dm.sort_index()
            data = dm.values
            min_clust_size=0.1
            util = get_util([x for x in data], dm.index, plot_clust, min_clust_size)

            #print("RAM " + str(ram))
        else:
            mem.index = pd.DatetimeIndex(mem.time)
            mem = mem.sort_index()
            dm = mem[param]
            #dm = dm.sort_index()
            data = dm.values
            min_clust_size=0.01
            util = get_util([x for x in data], dm.index, plot_clust, min_clust_size)
            
        res, stand_dev, point = last_period_forecast(data, min_size, min_clust_size=min_clust_size)
        #print("Utilized "+ str(util))
        #print("forecast_done")
        #print(dm.index)
        
        

    
        res = [np.median(data) if res[i]<min(data) else res[i] for i in range(len(res))]
        
        pred = (pd.Series(res, index=pd.date_range(str(dm.index[dm.shape[0] - 1]), periods=1008, freq='10min')))
        pred.index = pd.to_datetime(pred.index, unit='s')
        RES = pd.concat([dm, pred], axis=0)
        #Lin_model = (pd.Series(pred_data, index=pd.date_range(str(dm.index[dm.shape[0] - 1]), periods=1008, freq='10min')))

        #print("plot_forecast")
        fig1.add_subplot(npl)
        plt.plot(dm.index, dm.values, lw=1, label="Input data", color='blue')
        plt.plot(pred.index, pred.values, lw=2, label="Forecast", color='red', ls='--')
        plt.plot(pred.index, pred.values + 3 * stand_dev, lw=2, label="Forecast with confidence interval",
                 color='orange', ls='--')
        plt.plot(dm.index, [util]*len(dm.index), lw=3, label='Margin', color='red')
        #plt.plot(Lin_model.index, Lin_model.values, lw=2, label="Model of all TS data",
        #         color='green', ls='--')
        plt.fill_between(RES.index[point:len(dm.index) - 1], max(RES), min(RES), \
                         facecolor='yellow', alpha=0.5, \
                         label="last period with uniform structure")
        resource=param
        if resource=='real_used':
            resource='used memory'
        plt.title("Forecast of " + resource + " based on sar", fontsize=18)
        plt.legend(loc='upper left',prop={'size':10})
        #print("done")
        npl = npl + 1
        if input_data == "cpu_data":
            #print("Cores %")
            #print(np.float64(ncores)/100)
            D=D.append(pd.DataFrame({"Resource":input_data, "Capacity":ncores,"Param":param,"Used":(str(util)+" %"), "Details":"Used in %. All data ncores*100%"}, index=[host]))
        elif input_data=="mem_data":
            D = D.append(pd.DataFrame({"Resource": input_data, "Capacity": ram, "Param":param,"Used": util/1024/1024, "Details":"All data in GB"}, index=[host]))

    #plt.show(block=False)
    plt.show()
    fig1.show()
    fig1.savefig(path+host+"/Forecast_"+host.replace(".","_")+"_min_period_"+str(min_size), dpi=300)
    #''/home/'+user+'/tmp/'+host + "/Forecast_"+host.replace(".","_")+"_min_period_"+str(min_size), dpi=300)
    fig1.clf()
    return (D)
    
def get_resource_data(group_serv, DFR_RAM, DFR_CPU):
    D=pd.DataFrame()
    for G in DFR_RAM.Group.unique():
        if group_serv in G:
            print(G)
            df=pd.DataFrame()
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
            ax0, ax1= axes.flat

            d=DFR_RAM[DFR_RAM.Group==G]
            #print(d)
            D=D.append(d)
            X, Y=d.shape
            ind = np.arange(X)
            width = 0.5
            #d=d[['server','Used', 'Available']]
            p1 = ax0.bar(ind, d.Used.values,  color='red', label='Used_memory_Gb')
            p2 = ax0.bar(ind, d.Available.values, color='green', bottom=d.Used.values, label='Available')
            ax0.set_xticks(range(d.shape[0]))
            ax0.set_xticklabels([str(x) for x in d.server.values], rotation=90)
            ax0.set_title(G+" Memory usage Gb")
            ax0.get_legend()
            ax0.set_xlabel("KB")
            df=df.append(pd.DataFrame({"Group":G, "Number":d.shape[0],"Resource":"Memory","Sum_used":d.Used.sum(),"Sum_capacity":d.Capacity.sum() }, index=['RAM']))


            d=DFR_CPU[DFR_CPU.Group==G]
            D=D.append(d)
            X, Y=d.shape
            ind = np.arange(X)
            width = 0.5
            #d=d[['server','Used', 'Available']]
            d['Used']=np.float16(d['Used'].str.replace(" %", ""))
            p1 = ax1.bar(ind, d.Used.values,  color='red', label='Used_CPU_%')
            p2 = ax1.bar(ind, d.Available.values, color='green', bottom=d.Used.values, label='Available')
            ax1.set_xticks(range(d.shape[0]))
            ax1.set_xticklabels([str(x) for x in d.server.values], rotation=90)
            ax1.set_title(G+" CPU usage %")
            ax1.set_xlabel("%")


            #fig.text(G)
            fig.set_label(G)
            fig.show

            df=df.append(pd.DataFrame({"Group":G, "Number":d.shape[0], "Resource":"CPU","Sum_used":d.Used.sum(),"Sum_capacity":100*d.Capacity.sum() }, index=['CPU']), ignore_index=True)
            display(df)
            print("Summary CPU ")
            print("Used "+str(d.Used.sum()/d.Capacity.sum())+" %")
            print("Summary RAM ")
            print("Used "+str(100*np.float16(df['Sum_used'][df.Resource=='Memory'].values/df.Sum_capacity[df.Resource=='Memory'].values)[0])+" %")
    return D
