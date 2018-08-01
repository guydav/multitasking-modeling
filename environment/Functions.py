#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:58:04 2017

@author: norbert
"""

import numpy as np
import copy
#from itertools import permutations as perm
from itertools import combinations as comb
#from itertools import combinations_with_replacement as comb_wr
from itertools import product as prod

#%% Generate Input Patterns

def get_inputPatterns(units, dimensions, sc):
    
    base = np.zeros((1, units))
    base[0, 0] = 1.0
    
    if sc:
    
        scale = np.arange(0.25, 1, 0.25)
        
        base = np.reshape(scale, (-1, 1)) @ np.reshape(base, (1, -1))
    
    ## Add a second loop for scale dimension
    
    base_perm = copy.copy(base)
    
    for i in range(1, units):
        base_perm = np.vstack((base_perm, np.roll(base, i, axis = 1)))
    
    patterns = np.array(list(prod(base_perm, repeat = dimensions)))  
    
    patterns = patterns.reshape(patterns.shape[0], -1)
    
    return patterns

#%% Generate Task Patterns

def get_singleTasks(in_dim, out_dim):
    
    units = in_dim*out_dim
    
    base = np.zeros((1, units))
    base[0, 0] = 1.0
    
    patterns = copy.copy(base)
        
    for i in range(1, units):
        patterns = np.vstack((patterns, np.roll(base, i, axis = 1)))
        
    tasks = np.arange(units)
    
    task_map = np.reshape(tasks, (in_dim, out_dim))

    return patterns, task_map


#%% Generate Mult-Task Patterns

def get_multiTasks(in_dim, out_dim, simult):
    
    if (simult >= in_dim or simult >= out_dim):
        if in_dim <= out_dim:
            simult = in_dim
        else:
            simult = out_dim
    elif simult < 2:
        simult = 2
    
    units = in_dim*out_dim
    
    tasks = np.arange(units)
    
    task_map = np.reshape(tasks, (in_dim, out_dim))
    
    c = np.array(list(comb(tasks, simult)))
    
    num_combs = np.size(c, axis = 0)
    
    mult_able = np.zeros(num_combs) == 1
    
    dims = np.zeros((num_combs, simult, 2))
    
    for mult in range(num_combs):
        mult_tasks = c[mult, :]
        mult_dims = np.zeros((simult, 2))
        
        for task in range(simult):
            mult_dims[task, 0] = np.where(task_map == mult_tasks[task])[0][0]
            mult_dims[task, 1] = np.where(task_map == mult_tasks[task])[1][0]
            
        dims[mult, :, :] = mult_dims[:, :]
        
        mult_c = np.array(list(comb(np.arange(simult), 2)))
        
        num_mult_c = np.size(mult_c, axis = 0)
        
        able = np.zeros(num_mult_c) == 1
        
        for i in range(num_mult_c):
            able[i] = (dims[mult, mult_c[i, 0], 0] != dims[mult, mult_c[i, 1], 0])*(dims[mult, mult_c[i, 0], 1] != dims[mult, mult_c[i, 1], 1])
        
        mult_able[mult] = np.prod(able)
        
    c_able = c[mult_able]
    
    
    # generate patterns
    
    patterns = np.zeros((len(c_able),units))
    
    for i in range(len(patterns)):
        patterns[i, c_able[i]] = 1
        
    return patterns, task_map


#%% Generate Train Patterns

def get_trainPatterns(in_patterns, tsks, units_dim, dimensions, tsk_map, mult):
    
    num_trials = len(in_patterns)*len(tsks)
    patterns = np.zeros((num_trials, units_dim*dimensions))
    input_patterns = np.zeros((num_trials, np.size(in_patterns, 1)))
    tsk_patterns = np.zeros((num_trials, np.size(tsks, 1)))
    
    if mult:
        
        in_out_map = None    ## Have to fix this
        
        trial = 0
        
        for tsk_pattern in tsks:
            
            for in_pattern in in_patterns:
                
                input_patterns[trial] = in_pattern
                
                tsk_patterns[trial] = tsk_pattern
                
                tsk = np.where(tsk_pattern > 0.75)[0]
                
                mapping = []
                
                for i in range(len(tsk)):
                    
                    mapping += [np.array(np.where(tsk_map == tsk[i]))[:, 0]]
                    
                    in_dim = mapping[i][0]
                    
                    out_dim = mapping[i][1]
                    
                    patterns[trial, int(units_dim*out_dim):int(units_dim*(out_dim + 1))] = (
                            in_pattern[int(units_dim*in_dim):int(units_dim*(in_dim + 1))])
                    
                trial += 1
                    
    else:
    
        
        in_out_map = np.zeros((num_trials, tsk_map.ndim))
        
        trial = 0
        
        for tsk_pattern in tsks:
            
            for in_pattern in in_patterns:
                
                input_patterns[trial] = in_pattern
                
                tsk_patterns[trial] = tsk_pattern
                
                tsk = np.where(tsk_pattern == np.max(tsk_pattern))[0][0]
                
                mapping = np.array(np.where(tsk_map == tsk))
                
                in_out_map[trial] = mapping[:, 0]
                
                in_dim = in_out_map[trial][0]
                
                out_dim = in_out_map[trial][1]
                
                patterns[trial, int(units_dim*out_dim):int(units_dim*(out_dim + 1))] = (
                        in_pattern[int(units_dim*in_dim):int(units_dim*(in_dim + 1))])
                
                trial += 1

    return patterns, tsk_patterns, in_out_map, input_patterns


#%% Input Congruency

def getCongruency(patterns, task_patterns, task_map, units_dim):
    
    congruency_map = np.zeros(np.size(patterns, axis = 0)) == 1
    incongruency_map = np.zeros(np.shape(congruency_map)) == 1
            
    for p in range(len(patterns)):
        
        tasks = np.where(task_patterns[p] == 1)[0]
        
        in_dims = np.zeros(len(tasks))
        
        for i in range(len(tasks)):
            
            in_dims[i] = np.where(task_map == tasks[i])[0]
        
        congruency = True
        
        incongruency = True
        
        dim_combs = np.array(list(comb(in_dims, 2)))
        
        for dims in dim_combs:
            
            congruency = congruency*(patterns[p, int(units_dim*dims[0]):int(units_dim*(dims[0]+1))] == patterns[p, int(units_dim*dims[1]):int(units_dim*(dims[1]+1))]).all(axis = 0)
    
            incongruency = incongruency*~((patterns[p, int(units_dim*dims[0]):int(units_dim*(dims[0]+1))] == patterns[p, int(units_dim*dims[1]):int(units_dim*(dims[1]+1))]).all(axis = 0))
    
        congruency_map[p] = congruency
        incongruency_map[p] = incongruency
        
    return congruency_map, incongruency_map

#%% Intialize weights

def init_weights(in_units, h_units, out_units, t_units, bias):
    
    weights_IH = np.random.uniform(-1/in_units**0.5, 1/in_units**0.5, (in_units, h_units))
    weights_HO = np.random.uniform(-1/h_units**0.5, 1/h_units**0.5, (h_units, out_units))
    weights_TH = np.random.uniform(-1/t_units**0.5, 1/t_units**0.5, (t_units, h_units))
    weights_TO = np.random.uniform(-1/t_units**0.5, 1/t_units**0.5, (t_units, out_units))
    bias_H = bias*np.ones(h_units)
    bias_O = bias*np.ones(out_units)
    
    return (weights_IH, weights_HO, weights_TH, weights_TO, bias_H, bias_O)
    
  
#%% Activity and Derivative

def activity(net_input, derive = False):
    
    f_act = 1/(1 + np.exp(-1*net_input))
    
    if derive:
        
        return np.reshape(f_act*(1 - f_act), (1, -1))
    else:
        
        return np.reshape(f_act, (1, -1))
    
#%% Forward Pass

def forward_pass(inputs, tasks, weights):
    
    # weights are given as a list / includes all layer combinations and bias (IH, TH, HO, TO, BH, BO)
    
    IH = weights[0]
    TH = weights[1]
    HO = weights[2]
    TO = weights[3]
    BH = weights[4]
    BO = weights[5]
    
    output_units = np.size(HO, axis = 1)
    hidden_units = np.size(IH, axis = 1)
    
    trials = len(inputs)
    
    hidden = np.zeros((trials, hidden_units))
    
    output = np.zeros((trials, output_units))
    
    for trial in range(trials):
        
        hidden_net = np.reshape(inputs[trial], (1, -1)) @ IH + np.reshape(tasks[trial], (1, -1)) @ TH + np.reshape(BH, (1, -1))
    
        hidden_act  = activity(hidden_net)
        
        hidden[trial] = hidden_act
        
        output_net = hidden_act @ HO + np.reshape(tasks[trial], (1, -1)) @ TO + np.reshape(BO, (1, -1))
    
        output[trial] = activity(output_net)
        
    return hidden, output

#%% Test Performance

def performance(target_patterns, test_output, task_patterns, mult, simult):
    
    # Compare choices to target
        
    correct = np.zeros((np.size(target_patterns, 0), simult))
    
    for i in range(len(correct)):
        correct[i, :] = np.where(target_patterns[i] == np.max(target_patterns[i]))[0]
    
    choice = np.zeros(np.shape(correct))
    
    for i in range(len(choice)):
        choice[i, :] = np.sort(np.argsort(test_output[i])[-simult:])
        
    classification = np.prod(choice == correct, axis = 1)
    
    
    
     # Strat2 PGC
    
    e = target_patterns - test_output
    
    ae = np.abs(e)
    
    treshold = ae <= 0.20
    
    good = np.sum(treshold, axis = 1) == np.size(test_output, 1)
    
    
    # Combine measures to compute PGC
    
    combined = classification*good
    
    pgc = 100*np.sum(combined)/len(combined)
    
    
    # Other measures
    
    mae = np.mean(ae, 1)
    se = e**2
    rmse = np.sqrt(np.mean(se, 1))
    
    
    # By Task
    
    if not mult:
    
        task_units = np.size(task_patterns, axis = 1)

        mae_byTask = np.zeros((int(np.size(mae, axis = 0)/task_units), task_units))
        
        shape = np.shape(mae_byTask)
        
        rmse_byTask = np.zeros(shape)
        pgc_byTask = np.zeros(task_units)
        class_byTask = np.zeros(shape)
        good_byTask = np.zeros(shape)
        correct_byTask = np.zeros(shape)
        choice_byTask = np.zeros(shape)
        
        tasks = np.where(task_patterns == np.max(task_patterns))[1]
        
        for task in range(task_units):
            condition = np.where(tasks == task)[0]
            
            mae_byTask[:, task] = mae[condition]
            rmse_byTask[:, task] = rmse[condition]
            class_byTask[:, task] = classification[condition]
            good_byTask[:, task] = good[condition]
            
            combined_byTask  = class_byTask[:, task]*good_byTask[:, task]
            pgc_byTask[task] = 100*np.sum(combined_byTask)/len(combined_byTask)
            
            correct_byTask[:, task] = np.reshape(correct[condition], np.shape(correct_byTask[:, 0]))
            choice_byTask[:, task] = np.reshape(choice[condition], np.shape(choice_byTask[:, 0]))
        
    else:
        
        mae_byTask = None
        rmse_byTask = None
        pgc_byTask = None
        class_byTask = None
        good_byTask = None
        correct_byTask = None
        choice_byTask = None
        
    
    return (mae, rmse, pgc, classification, good, correct, choice, mae_byTask, 
            rmse_byTask, pgc_byTask, class_byTask, good_byTask, correct_byTask,
            choice_byTask)


    