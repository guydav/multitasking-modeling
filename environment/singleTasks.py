#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:47:28 2017

@author: norbert
"""

# Import dependencies

from Functions import *
from matplotlib import pyplot as plt

#%% Set seed

# Set the seed for random number generator (Important for weight initialization)

seed = 28

np.random.seed(seed)

#%% Parameters

units_perdim = 3    # features per dimension  (default: 3)

input_dim = 3       # input dimensions (default: 3)

output_dim = 3      # output dimensions (default: 3)


alpha = 0.3         # Learning rate

bias = -1           # bias weight   (shifts threshold for unit activation)

simult = 1          # number of task to be performed simultaneously (1 for single tasks)


#%% Netwwork structure

# Input layer

input_units = units_perdim*input_dim


# Output layer

output_units = units_perdim*output_dim


# Task Layer

task_units = input_dim*output_dim   # all possible input-output mappings


# Hidden layer

hidden_units = input_units*output_units  # enough to not force the network to generalize


#Initialize Weights
    
(w_IH, w_HO, w_TH, w_TO, bias_H, bias_O) = init_weights(input_units, hidden_units, output_units, task_units, bias)


#%% Generate patterns

input_patterns = get_inputPatterns(units_perdim, input_dim, sc = False)

tasks, task_map = get_singleTasks(input_dim, output_dim)


#%% Initialize Logs

E_log = []          # Error per pattern per epoch
input_log = []
hidden_log = []     # Hidden actvity per pattern per epoch
output_log = []     
target_log = []     # Expected output per pattern per epoch
io_map_log = []     # Task input-output map per pattern per epoch
tasks_log = []      # Task per pattern per epoch

# hidden and output activity per pattern after each epoch of training

test_hidden_log = []
test_output_log = []

# weight logs include initial values (bias weights not part of learning)

w_IH_log = [w_IH]   # per epoch
w_HO_log = [w_HO]
w_TH_log = [w_TH]
w_TO_log = [w_TO]

# Performance measures per epoch

mae_log = []                # Mean Absolute Error 
rmse_log = []               # Root Mean Square Error
pgc_log = []                # Percent Good Choices
class_log = []              # Correct/Incorrect Classifications (max output same as target pattern)
good_log = []               # Good/Bad Classifications (abs error < threshold across nodes)
expected_log = []           # Expected choices during test 
choice_log = []             # Actual choices during test

# Performance measures per epoch by task

mae_byTask_log = []
rmse_byTask_log = []
pgc_byTask_log = []
class_byTask_log = []
good_byTask_log = []
expected_byTask_log = []
choice_byTask_log = []


#%% Train Network

train = True
hard_stop = 600
iterations = 0
no_prog = 0
prog_thresh = 150


while train:
        
    in_patterns = input_patterns[np.random.permutation(len(input_patterns))]
    tasks_perm = tasks[np.random.permutation(len(tasks))].reshape((len(tasks), -1))
    
    target_patterns, task_patterns, io_map, patterns = get_trainPatterns(in_patterns, tasks_perm, units_perdim, output_dim, task_map, mult = False)
    
    target_log += [target_patterns]
    tasks_log += [task_patterns]
    io_map_log += [io_map]
    input_log += [patterns]
    
    trials = len(target_patterns)
    
    E = np.zeros((trials, output_units))
    output_patterns = np.zeros((trials, output_units))
    hidden_patterns = np.zeros((trials, hidden_units))
    
    for trial in range(trials):

        # Compute activity for hidden and output layers
    
        hidden_net = np.reshape(patterns[trial], (1, -1)) @ w_IH + np.reshape(task_patterns[trial], (1, -1)) @ w_TH + np.reshape(bias_H, (1, -1))
    
        hidden_act  = activity(hidden_net)
        
        hidden_patterns[trial] = hidden_act
        
        output_net = hidden_act @ w_HO + np.reshape(task_patterns[trial], (1, -1)) @ w_TO + np.reshape(bias_O, (1, -1))
    
        output_act = activity(output_net)
        
        output_patterns[trial] = output_act
    
        
        # Compute error and deltas
        
        E[trial] = output_act - target_patterns[trial]
                     
        delta_out = activity(output_net, derive = True)*E[trial]
        
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
    E_log += [E]
    w_IH_log += [w_IH]
    w_HO_log += [w_HO]
    w_TH_log += [w_TH]
    w_TO_log += [w_TO]
    
    
    # Test Performance
    
    weights = [w_IH, w_TH, w_HO, w_TO, bias_H, bias_O]    # could use any weights across epochs
    
    test_hidden, test_output = forward_pass(patterns, task_patterns, weights)
    
    test_hidden_log += [test_hidden]
    test_output_log += [test_output]
    
    (mae, rmse, pgc, classification, good, expected, choice, mae_byTask, rmse_byTask, 
     pgc_byTask, class_byTask, good_byTask, expected_byTask, choice_byTask) = performance(target_patterns, test_output, task_patterns, mult = False, simult = simult)
    
    mae_log += [mae]
    rmse_log += [rmse]
    pgc_log += [pgc]
    class_log += [classification]
    good_log += [good]
    expected_log += [expected]
    choice_log += [choice]
    
    mae_byTask_log += [mae_byTask]
    rmse_byTask_log += [rmse_byTask]
    pgc_byTask_log += [pgc_byTask]
    class_byTask_log += [class_byTask]
    good_byTask_log += [good_byTask]
    expected_byTask_log += [expected_byTask]
    choice_byTask_log += [choice_byTask]
    
    
    # Stop Training?
    
    iterations += 1
    
    if iterations == hard_stop:
        train = False
        
    elif pgc == 100:
        train = False
    
    if pgc == pgc_log[len(pgc_log)-2]:  # pgc_log has already been updated with new value
        no_prog += 1
        
    if no_prog == prog_thresh:
        train = False
   
     
#%% Visualize Training

MSE = np.mean(np.array(E_log)**2, axis = (1, 2))
RMSE = np.mean(np.array(rmse_log), axis = 1)
MAE = np.mean(np.array(mae_log), axis = 1)
PGC = np.array(pgc_log)
PGC_byTask = np.array(pgc_byTask_log)

plt.figure('Error Measures')
plt.plot(MSE, 'k')
plt.plot(MAE, 'r')
plt.plot(RMSE, 'b')
plt.legend(['MSE', 'MAE', 'RMSE'])
plt.xlabel('Epoch')

plt.figure('PGC')
plt.plot(PGC, 'g')
plt.xlabel('Epoch')

plt.figure('PGC by Task')
plt.plot(PGC_byTask)
plt.xlabel('Epoch')


#%% Average Hidden Activity by Task

task_node = np.where(task_patterns == np.max(task_patterns))[1]

hidden_byTask = np.zeros((task_units, hidden_units))

# could use any test_hidden that has been logged

for node in range(task_units):
    hidden_byTask[node, :] = np.mean(test_hidden[task_node == node, :], axis = 0)
    
task_corrMat = np.corrcoef(hidden_byTask)

plt.figure('Hidden Representation Similarity ST')
plt.imshow(task_corrMat, cmap='rainbow', interpolation='nearest')
plt.colorbar()
plt.clim(-1, 1)
plt.xticks(np.arange(task_units))
plt.yticks(np.arange(task_units))
plt.show()

#%% Export variables

import pickle

with open('singleTasks.pickle', 'wb') as handle:
    pickle.dump([units_perdim, input_dim, output_dim, w_IH_log, w_HO_log, w_TH_log,
                w_TO_log, bias_H, bias_O], handle, protocol = pickle.HIGHEST_PROTOCOL)