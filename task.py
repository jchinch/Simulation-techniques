#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing necessary libraries
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Setting a random seed generator to provide randomness in the experiment
random.seed(42)

# Calculating mean for the respective batch using stated formula
def get_mean(sample):
    return sum(sample)/len(sample)

# Calculating 95th percentile value.
def get_percentile(sample):
    sample.sort()
    return sample[int(math.ceil(len(sample)*0.95))]

# Calculating Standard deviation for the respective batch using stated formulae    
def get_standard_deviation(samples, mean):
    temp = 0
    for sample in samples:
        temp += (sample - mean)*(sample - mean)
    return math.sqrt(temp/(len(samples)-1))

# Calculating 95% Confidence interval for the respective batch using stated formulae.
def get_confidence_interval(batch_mean, std, n):
    t_95 = 1.68
    temp = (std*t_95)/math.sqrt(n)
    return (batch_mean - temp, batch_mean + temp)
    
def simulation(mean_time_RT, mean_time_nRT, serve_RT, serve_nRT, m, b):
    ## Setting initial condition as per the task 1 in the project
    MC, RTCL, nRTCL, nRT, nonRT, SCL, status, prempt_time = [0.0], [3.0], [5.0], [0], [0], [4.0], [2], [0.0]
    temp = [RTCL[-1], nRTCL[-1], SCL[-1]]
    
    ## Generating empty lists to store mean, percentile variable values.
    RT_mean, nRT_mean = [], []
    RT_percentile, nRT_percentile = [], []
    arrival_RT, arrival_nRT = [], []
    for i in range(m):
        response_RT, response_nRT = [], []
        for j in range(b):
            idx = temp.index(min(temp))
            # If the RT signal arrives at the scheduler prcess it and append necessary time.
            if idx == 0:
                r = random.uniform(0,1)
                random_RT = -1*mean_time_RT*math.log(r)
                random_nRT = -1*mean_time_nRT*math.log(r)
                random_st_RT = -1*serve_RT*math.log(r)
                random_st_nRT = -1*serve_nRT*math.log(r)
                MC.append(RTCL[-1])
                RTCL.append(RTCL[-1] + random_RT)
                nRTCL.append(nRTCL[-1])
                nRT.append(nRT[-1] + (1 if status[-1] == 1 and MC[-1] < SCL[-1] else 0))
                nonRT.append(nonRT[-1] + (1 if RTCL[-2] < SCL[-1] else 0))
                SCL.append(RTCL[-2] + random_st_RT)
                status.append(1)
                prempt_time.append(SCL[-2] - RTCL[-2] if RTCL[-2] < SCL[-2] else  0)
                temp = [RTCL[-1], nRTCL[-1], SCL[-1]]
                arrival_RT.append(MC[-1])
                if nRT[-1] == 0:
                    arrival_time = arrival_RT.pop(0)
                    response_RT.append(SCL[-1]-arrival_time)
                    
            # If Non Real Time signal arrives at scheduler.        
            elif idx == 1:
                r = random.uniform(0,1)
                random_RT = -1*mean_time_RT*math.log(r)
                random_nRT = -1*mean_time_nRT*math.log(r)
                random_st_RT = -1*serve_RT*math.log(r)
                random_st_nRT = -1*serve_nRT*math.log(r)
                arrival_nRT.append(nRTCL[-1])
                
                if nRTCL[-1] >= SCL[-1]:
                    MC.append(nRTCL[-1])
                    RTCL.append(RTCL[-1])
                    nRTCL.append(nRTCL[-1] + random_nRT)
                    nRT.append(nRT[-1])
                    nonRT.append(nonRT[-1] + (1 if MC[-1] < SCL[-1] else 0))
                    SCL.append(MC[-1] + random_st_nRT)
                    status.append(2)
                    prempt_time.append(prempt_time[-1])
                    temp = [RTCL[-1], nRTCL[-1], SCL[-1]]
                    if nonRT[-1] == nonRT[-2]:
                        arrival_time = arrival_nRT.pop(0)
                        response_nRT.append(MC[-1] + random_st_nRT - arrival_time)
                else:
                    MC.append(nRTCL[-1])
                    RTCL.append(RTCL[-1])
                    nRTCL.append(nRTCL[-1] + random_nRT)
                    nRT.append(nRT[-1])
                    nonRT.append(nonRT[-1] + 1)
                    SCL.append(SCL[-1])
                    status.append(status[-1])
                    prempt_time.append(prempt_time[-1])
                    temp = [RTCL[-1], nRTCL[-1], SCL[-1]]
            # If server status is idle and no next process in queue for the server.
            elif idx == 2 and nRT[-1] == 0 and nonRT[-1] == 0 and prempt_time[-1] == 0:
                MC.append(SCL[-1])
                RTCL.append(RTCL[-1])
                nRTCL.append(nRTCL[-1])
                nRT.append(nRT[-1])
                nonRT.append(nonRT[-1])
                SCL.append(SCL[-1])
                status.append(0)
                prempt_time.append(prempt_time[-1])
                temp.pop(idx)
                idx = temp.index(min(temp))
            # If status is idle but nRT signal are there or premptive time is there for non RT.
            else:
                r = random.uniform(0,1)
                random_RT = -1*mean_time_RT*math.log(r)
                random_nRT = -1*mean_time_nRT*math.log(r)
                random_st_RT = -1*serve_RT*math.log(r)
                random_st_nRT = -1*serve_nRT*math.log(r)
                
                if nRT[-1] > 0:
                    MC.append(SCL[-1])
                    RTCL.append(RTCL[-1])
                    nRTCL.append(nRTCL[-1])
                    nRT.append(nRT[-1] - 1)
                    nonRT.append(nonRT[-1])
                    SCL.append(RTCL[-2] + random_st_RT)
                    status.append(1)
                    prempt_time.append(SCL[-2] - RTCL[-2] if RTCL[-2] < SCL[-2] else  0)
                    temp = [RTCL[-1], nRTCL[-1], SCL[-1]]
                    arrival_time = arrival_RT.pop(0)
                    response_RT.append(SCL[-1] - arrival_time)
                    
                else:
                    if prempt_time[-1] > 0:
                        MC.append(SCL[-1])
                        RTCL.append(RTCL[-1])
                        nRTCL.append(nRTCL[-1])
                        nRT.append(nRT[-1])
                        nonRT.append(nonRT[-1] - 1)
                        SCL.append(SCL[-1] + prempt_time[-1])
                        status.append(2)
                        prempt_time.append(0)
                        temp = [RTCL[-1], nRTCL[-1], SCL[-1]]
                    
                    else:
                        MC.append(SCL[-1])
                        RTCL.append(RTCL[-1])
                        nRTCL.append(nRTCL[-1])
                        nRT.append(nRT[-1])
                        nonRT.append(nonRT[-1] - 1)
                        SCL.append(SCL[-1] + random_st_nRT)
                        status.append(2)
                        prempt_time.append(0)
                        temp = [RTCL[-1], nRTCL[-1], SCL[-1]]
                        arrival_time = arrival_nRT.pop(0)
                        response_nRT.append(MC[-1] + random_st_nRT - arrival_time)
        # Ignoring the first batch in order to ignore transient state values of schedulling.    
        if i == 0:
            continue
        RT_mean.append(get_mean(response_RT))
        nRT_mean.append(get_mean(response_nRT))
        RT_percentile.append(get_percentile(response_RT))
        nRT_percentile.append(get_percentile(response_nRT))
        
    return RT_mean, nRT_mean, RT_percentile, nRT_percentile
        
# Generating data to be used for fetching a graph.
def graph_data(mean_time_RT, serve_RT, serve_nRT, m, b):
    mean_time_nRT = [10, 15, 20, 25, 30, 35, 40]
    output_mean_data = []
    output_percentile_data = []
    for time in mean_time_nRT:
        temp1, temp2 = [], []
        RT_mean, nRT_mean, RT_percentile, nRT_percentile = simulation(mean_time_RT, time, serve_RT, serve_nRT, m, b)
        
        RT_super_mean = get_mean(RT_mean)
        nRT_super_mean = get_mean(nRT_mean)
    
        RT_standard_deviation = get_standard_deviation(RT_mean, RT_super_mean)
        nRT_standard_deviation = get_standard_deviation(nRT_mean, nRT_super_mean)
        
        RT_95percentile = get_confidence_interval(RT_super_mean, RT_standard_deviation, m - 1)
        nRT_95percentile = get_confidence_interval(nRT_super_mean, nRT_standard_deviation, m - 1)
        
        RT_batch_percentile = get_percentile(RT_percentile)
        nRT_batch_percentile = get_percentile(nRT_percentile)
        
        RT_percentile_std = get_standard_deviation(RT_percentile, RT_batch_percentile)
        nRT_percentile_std = get_standard_deviation(nRT_percentile, nRT_batch_percentile)
        
        RT_95th_conf = get_confidence_interval(RT_batch_percentile, RT_percentile_std, m-1)
        nRT_95th_conf = get_confidence_interval(nRT_batch_percentile, nRT_percentile_std, m-1)
        
        temp1 =  temp1 + [RT_super_mean, RT_95percentile[0], RT_95percentile[1], nRT_super_mean, nRT_95percentile[0], nRT_95percentile[1]]
        output_mean_data.append(temp1)
        temp2 = temp2 + [RT_batch_percentile, RT_95th_conf[0], RT_95th_conf[1], nRT_batch_percentile, nRT_95th_conf[0], nRT_95th_conf[1]]
        output_percentile_data.append(temp2)
    return output_mean_data, output_percentile_data

def plot_graph(output_mean_data, output_percentile_data):
    #Extracting mean values for RT and nRT messages
    RT_bar, nRT_bar, RT_mean_lower, nRT_mean_lower = [],[],[],[]
    for i in range(7):
        RT_bar.append(output_mean_data[i][0])
        nRT_bar.append(output_mean_data[i][3])
        RT_mean_lower.append(output_mean_data[i][1])
        nRT_mean_lower.append(output_mean_data[i][4])
        
    RT_error = np.array(RT_bar) - np.array(RT_mean_lower)
    nRT_error = np.array(nRT_bar) - np.array(nRT_mean_lower)
    r1 = np.arange(len(RT_bar))
    r2 = [x + 0.3 for x in r1]
     
    # Create blue bars
    plt.bar(r1, RT_bar, width = 0.3, color = 'blue', edgecolor = 'black', yerr=RT_error, capsize=7, label='RT mean')
    plt.bar(r2, nRT_bar, width = 0.3, color = 'brown', edgecolor = 'black', yerr=nRT_error, capsize=7, label='nonRT mean')
     
    # Plot mean graph
    plt.xticks([r + 0.3 for r in range(len(RT_bar))], ['10', '15', '20', '25', '30', '35', '40'])
    plt.ylabel('Mean')
    plt.xlabel('MIAT of nonRT')
    plt.plot(RT_bar)
    plt.plot(nRT_bar)
    plt.legend()
    plt.title("Mean Plot")
    plt.savefig('mean_plot.png')
    plt.close()
    #Extracting CI values values for RT and nRT messages
    RT_bar, nRT_bar, RT_mean_lower, nRT_mean_lower = [],[],[],[]
    for i in range(7):
        RT_bar.append(output_percentile_data[i][0])
        nRT_bar.append(output_percentile_data[i][3])
        RT_mean_lower.append(output_percentile_data[i][1])
        nRT_mean_lower.append(output_percentile_data[i][4])
        
    RT_error = np.array(RT_bar) - np.array(RT_mean_lower)
    nRT_error = np.array(nRT_bar) - np.array(nRT_mean_lower)
    r1 = np.arange(len(RT_bar))
    r2 = [x + 0.3 for x in r1]
     
    # Create blue bars
    plt.bar(r1, RT_bar, width = 0.3, color = 'blue', edgecolor = 'black', yerr=RT_error, capsize=7, label='RT mean')
    plt.bar(r2, nRT_bar, width = 0.3, color = 'brown', edgecolor = 'black', yerr=nRT_error, capsize=7, label='nonRT mean')
     
    # Plot mean graph
    plt.xticks([r + 0.3 for r in range(len(RT_bar))], ['10', '15', '20', '25', '30', '35', '40'])
    plt.ylabel('95th Percentile')
    plt.xlabel('MIAT of nonRT')
    plt.plot(RT_bar)
    plt.plot(nRT_bar)
    plt.legend()
    plt.title("95th Percentile Plot")
    plt.savefig('95percentile_plot.png')
    plt.close()
    
    

def main():
    mean_time_RT = float(input('mean inter arrival time of RT messages: '))
    mean_time_nRT = float(input('mean inter arrival time of non RT messages: '))
    serve_RT = float(input('mean service time of an RT message: '))
    serve_nRT = float(input('mean service time of an non RT message: '))
    m = int(input("enter the number of batches: "))
    b = int(input('enter the batch size: '))
    
    RT_mean, nRT_mean, RT_percentile, nRT_percentile = simulation(mean_time_RT, mean_time_nRT, serve_RT, serve_nRT, m, b)
    
    #Calculate super mean for the batch.
    RT_super_mean = get_mean(RT_mean)
    nRT_super_mean = get_mean(nRT_mean)
    #Calculate standar deviation which will be used for CI calculation.
    RT_standard_deviation = get_standard_deviation(RT_mean, RT_super_mean)
    nRT_standard_deviation = get_standard_deviation(nRT_mean, nRT_super_mean)
    #Calculating confidence Intervals for mean values.
    RT_95percentile = get_confidence_interval(RT_super_mean, RT_standard_deviation, m - 1)
    nRT_95percentile = get_confidence_interval(nRT_super_mean, nRT_standard_deviation, m - 1)
    
    RT_batch_percentile = get_percentile(RT_percentile)
    nRT_batch_percentile = get_percentile(nRT_percentile)
    
    RT_percentile_std = get_standard_deviation(RT_percentile, RT_batch_percentile)
    nRT_percentile_std = get_standard_deviation(nRT_percentile, nRT_batch_percentile)
    
    RT_95th_conf = get_confidence_interval(RT_batch_percentile, RT_percentile_std, m-1)
    nRT_95th_conf = get_confidence_interval(nRT_batch_percentile, nRT_percentile_std, m-1)
    
    print("-----------------------------Results----------------------------------")
    print("RT mean: {}".format(RT_super_mean))
    print("nonRT mean: {}".format(nRT_super_mean))
    print("RT mean CI: {}".format(RT_95percentile))
    print("nonRT mean CI: {}".format(nRT_95percentile))
    print("RT 95th percentile: {}".format(RT_batch_percentile))
    print("nonRT 95th percentile: {}".format(nRT_batch_percentile))
    print("RT 95th percentile CI: {}".format(RT_95th_conf))
    print("nonRT 95th percentile CI: {}".format(nRT_95th_conf))

    output_mean_data, output_percentile_data = graph_data(mean_time_RT, serve_RT, serve_nRT, m, b)
    plot_graph(output_mean_data, output_percentile_data)
    

if __name__ == "__main__":
    main()
    
