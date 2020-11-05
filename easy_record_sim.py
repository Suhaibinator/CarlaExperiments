#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:46:04 2020

@author: suhaib
"""

import pickle
import array
from deap import base, creator

creator.create("FitnessMaxMin", base.Fitness, weights=(-1.0, -1.0)) # Min: f1 & Min: f2
creator.create("Individual", array.array, typecode='f', fitness=creator.FitnessMaxMin, params=None)


rangefinder_ver = 34
gen = 507

def get_best_ind_num(rangefinder_version, gen):
    with open('./rangefinder_v' + str(rangefinder_version) + '/gen' + str(gen) + '_CS1_checkpoint.pkl', 'rb') as f:
        cp = pickle.load(f)
        
    pop_cs = cp['population']
    
    x_cs = [ind.fitness.values[0] for ind in pop_cs]
    y_cs = [ind.fitness.values[1] for ind in pop_cs]
    
    x_targ, y_targ = 00, 0
    
    jj = [(x_cs[i]-x_targ)**2+2*(y_cs[i]-y_targ)**2 for i in range(len(x_cs))]
    cs_ind = jj.index(min(jj))
    return cs_ind

ind = get_best_ind_num(rangefinder_ver, gen)

from record_sim import record_sim_to_video

record_sim_to_video(gen, True, 1, 1, ind, rangefinder_ver)