# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 06:37:46 2017

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import copy
from itertools import permutations as perm

np.random.seed(28)


### When building patterns try building all the possible patterns and then repeat that matrix in permuted order

#%% Building input patterns

def get_inputPatterns(n_patterns, n_units, base):
    
    permutations = np.array(list(perm(base)))
    
    n_copies = int(n_patterns/len(permutations))
    
    colors = np.tile(permutations, (n_copies, 1))
    
    words = copy.copy(colors)
    
    colors = colors[np.random.permutation(len(colors)),]

    words = words[np.random.permutation(len(words)),]

    patterns = np.hstack((colors, words))

    return patterns

num_patterns = 6000
input_units = 4      # 2 for colors, 2 for words

base_input = np.float64(np.array([1, 0]))

input_patterns = get_inputPatterns(num_patterns, input_units, base_input)

#%% Build control patterns

def get_controlPatterns(n_patterns, n_units, base):
    
    permutations = np.array(list(perm(base)))
    
    n_copies = int(n_patterns/len(permutations))
    
    patterns = np.tile(permutations, (n_copies, 1))
    
    patterns = patterns[np.random.permutation(len(patterns)),]
        
    return patterns

control_units = 2     # 1 for colors, 1 for words
    
control_patterns = get_controlPatterns(num_patterns, control_units, base_input)

#%%  Build train patterns

def get_trainPatterns(in_patterns, c_patterns, out_units):
    
    patterns = np.zeros(np.shape(c_patterns))
    cut = int(len(in_patterns[0, :])/out_units)
    
    control = np.zeros(len(in_patterns[:, 0]))
    
    for i in range(len(in_patterns[:, 0])):
        choose = np.where(c_patterns[i, :] == 1)[0]
        control[i] = choose
        patterns[i, :] = in_patterns[i, cut*choose:cut*(choose + 1)]

    return patterns, control

output_units = int(input_units/control_units)    
    
train_patterns, control = get_trainPatterns(input_patterns, control_patterns, output_units)


#%% Train network


hidden_units = input_units
bias_w = -1
alpha = 0.3

def init_weights(in_units, h_units, out_units, c_units):
    
    scale = 0.1
    
    weights_IH = scale*np.random.uniform(-1, 1, (in_units, h_units))
    weights_HO = scale*np.random.uniform(-1, 1, (h_units, out_units))
    weights_CH = scale*np.random.uniform(-1, 1, (c_units, h_units))
    weights_CO = scale*np.random.uniform(-1, 1, (c_units, out_units))
    bias_H = np.ones(h_units)
    bias_O = np.ones(out_units)
    
    return (weights_IH, weights_HO, weights_CH, weights_CO, bias_H, bias_O)
    
    
def activity(net_input, derive = False):
    
    f_act = 1/(1 + np.exp(-1*net_input))
    
    if derive:
        
        return np.reshape(f_act*(1 - f_act), (1, -1))
    else:
        
        return np.reshape(f_act, (1, -1))
  

# Initialize Weights

(w_IH, w_HO, w_CH, w_CO, bias_H, bias_O) = init_weights(input_units, hidden_units, output_units, control_units)
    
#Error = [[], []]

MSE = np.zeros(num_patterns)
output_patterns = np.zeros((num_patterns, output_units))

for i in range(num_patterns):
    
    # Compute activity for hidden and output layers
    
    hidden_net = np.reshape(input_patterns[i], (1, -1)) @ w_IH + np.reshape(control_patterns[i], (1, -1)) @ w_CH + bias_w*np.reshape(bias_H, (1, -1))

    hidden_act  = activity(hidden_net)
    
    output_net = hidden_act @ w_HO + np.reshape(control_patterns[i], (1, -1)) @ w_CO + bias_w*np.reshape(bias_O, (1, -1))

    output_act = activity(output_net)
    
    output_patterns[i] = output_act

    
    # Compute error and deltas
    
    MSE[i] = np.mean(0.5*(train_patterns[i] - output_act)**2)
                 
    delta_out = activity(output_net, derive = True)*(output_act - train_patterns[i])
    
    delta_hidden = activity(hidden_net, derive = True)*(delta_out @ np.transpose(w_HO))    
    
    
    # Compute gradient
    
    def gradient(activity, delta):
        return np.transpose(activity) @ delta
    
    dE_wHO = gradient(hidden_act, delta_out)

    dE_wIH = gradient(np.reshape(input_patterns[i], (1, -1)), delta_hidden)
    
    dE_wCO = gradient(np.reshape(control_patterns[i], (1, -1)), delta_out)
    
    dE_wCH = gradient(np.reshape(control_patterns[i], (1, -1)), delta_hidden)
    
    
    # Change Weights
    
    def delta_W(rate, weights, e_gradient):
        return weights - rate*e_gradient
    
    w_HO = delta_W(alpha, w_HO, dE_wHO)
    
    w_IH = delta_W(alpha, w_IH, dE_wIH)
    
    w_CO = delta_W(alpha, w_CO, dE_wCO)
    
    w_CH = delta_W(alpha, w_CH, dE_wCH)
    
    

#%% Plot Error

#for i in range(len(Error)):
#    if len(Error[i]) == 0:
#         continue
#    error = Error[i]
#    temp = copy.copy(error[0])
#    for j in range(1, len(error)):
#        temp = np.vstack((temp, error[j]))
#    Error[i] = temp

#plt.figure()
#plt.plot(np.sum(Error[0], axis = 1), '--.')
#plt.ylim([0, 0.5])
#
#plt.figure()
#plt.plot(np.sum(Error[1], axis = 1), '--.')
#plt.ylim([0, 0.5])    

plt.figure()
plt.plot(MSE)

plt.figure()
plt.plot(output_patterns[:, 0])
plt.plot(train_patterns[:, 0], alpha = 0.2)
plt.ylim([-1, 2])

