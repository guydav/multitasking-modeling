# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:28:22 2017

@author: user
"""

import numpy as np
from itertools import permutations as perm
from matplotlib import pyplot as plt

np.random.seed(28)

#%% Building input patterns

def get_inputPatterns(n_patterns, n_units, base):
    
    if n_units == len(base):
        
        permutations = np.array(list(perm(base)))
        
        n_copies = int(n_patterns/len(permutations))
        
        patterns = np.tile(permutations, (n_copies, 1))
        
        patterns = patterns[np.random.permutation(len(patterns)),]
 
    return patterns

num_patterns = 1000
input_units = 2     # 2 for colors, 2 for words

base_input = np.array([1, 0])

input_patterns = get_inputPatterns(num_patterns, input_units, base_input)

#%% #%% Build control patterns

#def get_controlPatterns(n_patterns, n_units, base):
#    
#    patterns = np.zeros((n_patterns, n_units))
#    
#    for i in range(n_patterns):
#        patterns[i, :] = np.random.permutation(base)
#    
#    return patterns
#
#control_units = 2     # 1 for colors, 1 for words
#    
#control_patterns = get_controlPatterns(num_patterns, control_units, base_input)

#%%  Build train patterns

#def get_trainPatterns(in_patterns, c_patterns):
#    
#    patterns = np.zeros(np.shape(c_patterns))
#    half = len(in_patterns[0, :])/2
#    
#    for i in range(len(in_patterns[:, 0])):
#        choose = np.where(c_patterns[i, :] == 1)[0]
#        patterns[i, :] = in_patterns[i, half*choose:half*(choose + 1)]
#
#    return patterns
#
#train_patterns = get_trainPatterns(input_patterns, control_patterns)

train_patterns = input_patterns

#%% Train network


hidden_units = 2*input_units
output_units = input_units
alpha = 0.8

def bias(h_units, out_units):
    
    bias_h = np.ones(h_units)
    bias_out = np.ones(out_units)
    
    return bias_h, bias_out

def init_weights(in_units, h_units, out_units, c_units):
    
    scale = 0.1
    
    weights_IH = scale*np.random.uniform(-1, 1, (in_units, h_units))
    weights_HO = scale*np.random.uniform(-1, 1, (h_units, out_units))
    weights_CH = scale*np.random.uniform(-1, 1, (c_units, h_units))
    weights_CO = scale*np.random.uniform(-1, 1, (c_units, out_units))
    
    return (weights_IH, weights_HO, weights_CH, weights_CO)
    
def init_weights(in_units, h_units, out_units):
    
    scale = 0.1
    
    weights_IH = scale*np.random.uniform(-1, 1, (in_units, h_units))
    weights_HO = scale*np.random.uniform(-1, 1, (h_units, out_units))
        
    return (weights_IH, weights_HO)
    
    
def activity(net_input, derive = False):
    
    f_act = 1/(1 + np.exp(-net_input))
    
    if derive:
        
        return f_act*(1-f_act)
    else:
        
        return f_act
  

# Initialize Weights

bias_H, bias_O = bias(hidden_units, output_units)

w_IH, w_HO = init_weights(input_units, hidden_units, output_units)

Error = np.zeros((num_patterns, output_units))
output_patterns = np.zeros((num_patterns, output_units))

for i in range(num_patterns):
    
    # Compute activity for hidden and output layers
    
    hidden_net = np.reshape(input_patterns[i], (1, -1)) @ w_IH - np.reshape(bias_H, (-1, hidden_units))

    hidden_act  = activity(hidden_net)
    
    output_net = hidden_act @ w_HO - np.reshape(bias_O, (-1, output_units))
    
    output_act = activity(output_net)
    
    output_patterns[i] = output_act
    
    # Compute error and deltas
    
    Error[i] = 0.5*(train_patterns[i] - output_act)**2
                 
    delta_out = activity(output_act, derive = True)*(output_act - train_patterns[i])
    
    delta_hidden = activity(hidden_act, derive = True)*(delta_out @ np.transpose(w_HO))    
    
    
    # Compute gradient
    
    def gradient(activity, delta):
        return np.transpose(activity) @ delta
    
    dE_wHO = gradient(hidden_act, delta_out)

    dE_wIH = gradient(np.reshape(input_patterns[i], (1, -1)), delta_hidden)
    
    # Change Weights
    
    def delta_W(rate, weights, e_gradient):
        return weights - rate*e_gradient
    
    w_HO = delta_W(alpha, w_HO, dE_wHO)
    
    w_IH = delta_W(alpha, w_IH, dE_wIH)
    
    
#%% Plot Error

plt.figure()
plt.plot(np.sum(Error, axis = 1))
plt.ylim([0, 0.5])


