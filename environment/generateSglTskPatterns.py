# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:32:26 2017

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import copy
#from itertools import permutations as perm
#from itertools import combinations as comb
#from itertools import combinations_with_replacement as comb_r
from itertools import product as prod
import time


#%% netwwork dimensions

units_perdim = 3


# Input layer

input_dim = 3

input_units = units_perdim*input_dim


# Output layer

output_dim = 3

output_units = units_perdim*output_dim


# Task Layer

task_units = input_dim*output_dim


# Hidden layer

hidden_units = input_units*output_units


#%% Generate Input Patterns

def get_inputPatterns(units, dimensions):
    
    base = np.zeros(units)
    base[0] = 1.0
    
    base_perm = copy.copy(base)
        
    for i in range(1, len(base)):
        base_perm = np.vstack((base_perm, np.roll(base, i)))
    
    patterns = np.array(list(prod(base_perm, repeat = dimensions)))    
    
    patterns = patterns.reshape(patterns.shape[0], -1)
    
    return patterns


#%% Generate Task Patterns

def get_taskPatterns(units, in_dim, out_dim):
    
    base = np.zeros(units)
    base[0] = 1.0
    
    patterns = copy.copy(base)
        
    for i in range(1, len(base)):
        patterns = np.vstack((patterns, np.roll(base, i)))
        
    tasks = np.arange(units)
    
    task_map = np.reshape(tasks, (in_dim, out_dim))

    return patterns, task_map


#%% Generate Train Patterns

def get_trainPatterns(in_patterns, tsks, units_dim, dimensions, tsk_map):
    
    num_trials = len(in_patterns)*len(tsks)
    
    patterns = np.zeros((num_trials, units_dim*dimensions))
    input_patterns = np.zeros((num_trials, np.size(in_patterns, 1)))
    tsk_patterns = np.zeros((num_trials, np.size(tsks, 1)))
    in_out_map = np.zeros((num_trials, tsk_map.ndim))
    
    trial = 0
    
    for tsk_pattern in tsks:
        
        for in_pattern in in_patterns:
            
            input_patterns[trial] = in_pattern
            
            tsk_patterns[trial] = tsk_pattern
            
            tsk = np.where(tsk_pattern == 1)[0][0]
            
            mapping = np.array(np.where(tsk_map == tsk))
            
            in_out_map[trial] = mapping[:, 0]
            
            in_dim = in_out_map[trial][0]
            
            out_dim = in_out_map[trial][1]
            
            patterns[trial, int(units_dim*out_dim):int(units_dim*(out_dim + 1))] = (
                    in_pattern[int(units_dim*in_dim):int(units_dim*(in_dim + 1))])
            
            trial += 1

    return patterns, tsk_patterns, in_out_map, input_patterns

            

#%% Train Network

def init_weights(in_units, h_units, out_units, t_units):
    
    scale = 0.1
    
    weights_IH = scale*np.random.uniform(-1, 1, (in_units, h_units))
    weights_HO = scale*np.random.uniform(-1, 1, (h_units, out_units))
    weights_TH = scale*np.random.uniform(-1, 1, (t_units, h_units))
    weights_TO = scale*np.random.uniform(-1, 1, (t_units, out_units))
    bias_H = np.ones(h_units)
    bias_O = np.ones(out_units)
    
    return (weights_IH, weights_HO, weights_TH, weights_TO, bias_H, bias_O)
    
    
def activity(net_input, derive = False):
    
    f_act = 1/(1 + np.exp(-1*net_input))
    
    if derive:
        
        return np.reshape(f_act*(1 - f_act), (1, -1))
    else:
        
        return np.reshape(f_act, (1, -1))
    

# Settings

np.random.seed(28)

alpha = 0.1
bias_w = -1


# Get input and task patterns

input_patterns = get_inputPatterns(units_perdim, input_dim)

tasks, task_map = get_taskPatterns(task_units, input_dim, output_dim)


# Set training iterations

batches = 1600
    # setting trials per batch in some cases?

# Initialize Weights

(w_IH, w_HO, w_TH, w_TO, bias_H, bias_O) = init_weights(input_units, hidden_units, output_units, task_units)


MSE_log = []
input_log = []
hidden_log = []
output_log = []
train_log = []
io_map_log = []
tasks_log = []
w_IH_log = [w_IH]
w_HO_log = [w_HO]
w_TH_log = [w_TH]
w_TO_log = [w_TO]

start = time.process_time()

for batch in range(batches):
    
    in_patterns = input_patterns[np.random.permutation(len(input_patterns))]
    tasks_perm = tasks[np.random.permutation(len(tasks))].reshape((len(tasks), -1))
    
    train_patterns, task_patterns, input_output_map, patterns = get_trainPatterns(in_patterns, tasks_perm, units_perdim, output_dim, task_map)
    train_log += [train_patterns]
    tasks_log += [task_patterns]
    io_map_log += [input_output_map]
    input_log += [patterns]
    
    trials = len(train_patterns)
    
    MSE = np.zeros(trials)
    output_patterns = np.zeros((trials, output_units))
    hidden_patterns = np.zeros((trials, hidden_units))
    
    for trial in range(trials):

        # Compute activity for hidden and output layers
    
        hidden_net = np.reshape(patterns[trial], (1, -1)) @ w_IH + np.reshape(task_patterns[trial], (1, -1)) @ w_TH + bias_w*np.reshape(bias_H, (1, -1))
    
        hidden_act  = activity(hidden_net)
        
        hidden_patterns[trial] = hidden_act
        
        output_net = hidden_act @ w_HO + np.reshape(task_patterns[trial], (1, -1)) @ w_TO + bias_w*np.reshape(bias_O, (1, -1))
    
        output_act = activity(output_net)
        
        output_patterns[trial] = output_act
    
        
        # Compute error and deltas
        
        MSE[trial] = np.mean(0.5*(train_patterns[trial] - output_act)**2)
                     
        delta_out = activity(output_net, derive = True)*(output_act - train_patterns[trial])
        
        delta_hidden = activity(hidden_net, derive = True)*(delta_out @ np.transpose(w_HO))    
        
        
        # Compute gradient
        
        def gradient(activity, delta):
            return np.transpose(activity) @ delta
        
        dE_wHO = gradient(hidden_act, delta_out)
    
        dE_wIH = gradient(np.reshape(patterns[trial], (1, -1)), delta_hidden)
        
        dE_wCO = gradient(np.reshape(task_patterns[trial], (1, -1)), delta_out)
        
        dE_wCH = gradient(np.reshape(task_patterns[trial], (1, -1)), delta_hidden)
        
        
        # Change Weights
        
        def delta_W(rate, weights, e_gradient):
            return weights - rate*e_gradient
        
        w_HO = delta_W(alpha, w_HO, dE_wHO)
        
        w_IH = delta_W(alpha, w_IH, dE_wIH)
        
        w_TO = delta_W(alpha, w_TO, dE_wCO)
        
        w_TH = delta_W(alpha, w_TH, dE_wCH)
        
    hidden_log += [hidden_patterns]
    output_log += [output_patterns]
    MSE_log += [MSE]
    w_IH_log += [w_IH]
    w_HO_log += [w_HO]
    w_TH_log += [w_TH]
    w_TO_log += [w_TO]
        
MSE_log = np.mean(np.array(MSE_log), 1)

end = time.process_time()

time = end - start
    
#plt.plot(MSE_log)
#plt.ylabel('MSE')
#plt.xlabel('Epochs ('+str(trials)+' patterns/epoch)')


#%% Export variables

import pickle

with open('singleTasks.pickle', 'wb') as handle:
    pickle.dump([units_perdim, input_dim, output_dim, w_IH_log, w_HO_log, w_TH_log, w_TO_log,
                 input_log, tasks_log, output_log, train_log, time], handle, protocol = pickle.HIGHEST_PROTOCOL)
