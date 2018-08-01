# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:31:27 2017

@author: user
"""

import numpy as np
from itertools import combinations as comb

#%% Multi-tasking


in_dim = 3
out_dim = 3
simult = 3


#def get_multiTasks(in_dim, out_dim, simult):
    
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
        
    


