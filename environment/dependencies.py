#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:42:02 2017

@author: norbert cruz
"""

import numpy as np
import copy
from itertools import combinations as comb
from itertools import product as prod

#%% Initialize Weights

def init_weights(in_units, h_units, out_units, t_units, bias):
    
    # make the weights small enough around 0, 
    # as to not bias the network in any direction
    
    weights_IH = np.random.uniform(-1/in_units**0.5, 1/in_units**0.5, (in_units, h_units))
    weights_HO = np.random.uniform(-1/h_units**0.5, 1/h_units**0.5, (h_units, out_units))
    weights_TH = np.random.uniform(-1/t_units**0.5, 1/t_units**0.5, (t_units, h_units))
    weights_TO = np.random.uniform(-1/t_units**0.5, 1/t_units**0.5, (t_units, out_units))
    bias_H = bias*np.ones(h_units)
    bias_O = bias*np.ones(out_units)
    
    return (weights_IH, weights_HO, weights_TH, weights_TO, bias_H, bias_O)

#%% Generate Input Patterns

def get_inputPatterns(units, dimensions):
    
    # generate a base vector with a single node active
    
    base = np.zeros((1, units))
    base[0, 0] = 1.0
    
    base_perm = copy.copy(base)   # make a copy as to not modify the base vector
    
    
    # roll the vector to get all possible nodes active in a single input dimension
    
    for i in range(1, units):
        base_perm = np.vstack((base_perm, np.roll(base, i, axis = 1)))
    
    
    # get all possible combinations of input dimensions
    
    patterns = np.array(list(prod(base_perm, repeat = dimensions)))  
    
    patterns = patterns.reshape(patterns.shape[0], -1)
    
    return patterns


#%% Generate Task Patterns

def get_taskPatterns(in_dim, out_dim):
    
    units = in_dim*out_dim
    
    # generate base vector
    
    base = np.zeros((1, units))
    base[0, 0] = 1.0
    
    patterns = copy.copy(base)
        
    
    # roll vector to get all possible nodes active
    
    for i in range(1, units):
        pattern s = np.vstack((patterns, np.roll(base, i, axis = 1)))
        
    
    # Generate a map of dimensions mapped by each task
        
    tasks = np.arange(units)
    
    task_map = np.reshape(tasks, (in_dim, out_dim))

    return patterns, task_map

#%% Generate Training Patterns

def get_trainPatterns(in_patterns, tsks, units_dim, dimensions, tsk_map):
    
    num_trials = len(in_patterns)*len(tsks)   # enough for all combinations of inputs and tasks
    
    patterns = np.zeros((num_trials, units_dim*dimensions))             # logs target per trial
    
    input_patterns = np.zeros((num_trials, np.size(in_patterns, 1)))    # logs input per trial
    
    tsk_patterns = np.zeros((num_trials, np.size(tsks, 1)))             # logs task per trial
        
    in_out_map = np.zeros((num_trials, tsk_map.ndim))                   # logs dim map per trial
    
    
    # generate input-task combination and target pattern per trial
    
    trial = 0
    
    for tsk_pattern in tsks:                # loops along task patterns
        
        for in_pattern in in_patterns:      # loops along input patterns
            
            # log selected patterns
        
            input_patterns[trial] = in_pattern
            
            tsk_patterns[trial] = tsk_pattern
            
            
            # find task to be implemented based on the task pattern
            
            tsk = np.where(tsk_pattern == np.max(tsk_pattern))[0][0]
            
            
            # find which dimension it maps
            
            mapping = np.array(np.where(tsk_map == tsk))
            
            in_out_map[trial] = mapping[:, 0]
            
            in_dim = in_out_map[trial][0]
            
            out_dim = in_out_map[trial][1]
            
            
            # Perform the dimension mapping from input to output
            
            patterns[trial, int(units_dim*out_dim):int(units_dim*(out_dim + 1))] = (
                    in_pattern[int(units_dim*in_dim):int(units_dim*(in_dim + 1))])
            
            trial += 1

    return patterns, tsk_patterns, in_out_map, input_patterns


#%% Activity and Derivative

def activity(net_input, derive = False):
    
    f_act = 1/(1 + np.exp(-1*net_input))    # sigmoid
    
    if derive:
        
        return np.reshape(f_act*(1 - f_act), (1, -1))    # derivative of the sigmoid
    else:
        
        return np.reshape(f_act, (1, -1))
    

#%% Error Gradient

def gradient(activity, delta):
    
    return np.transpose(activity) @ delta    # from backprop rule
    
    
#%% Modify Weights

def delta_W(rate, weights, e_gradient):
    
    return weights - rate*e_gradient        # from backprop rule