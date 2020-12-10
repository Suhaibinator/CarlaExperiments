#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:32:40 2020

@author: suhaib
"""


from eval_ind_3obj import Game

import time

from deap import base, creator
import pickle
import array
import copy
import torch
import numpy as np


n_obs = 5 # number of inputs
n_actions = 2 # number of outputs
track_num = 1
context_skill = True

def genotype_to_phenotype(vector, ns):
    # net_sample is needed to build the net from the vector
    # Individuals in the populations are represented as lists. 
    # In order to evaluate their performance, they need to be converted to 
    # neural network (phenotype)
    net_copy = copy.deepcopy(ns)
    vector_copy = copy.deepcopy(np.array(vector))
    
    for p in net_copy.parameters():
        len_slice = p.numel()
        replace_with = vector_copy[0:len_slice].reshape(p.data.size())
        p.data = torch.from_numpy( replace_with )
        vector_copy = np.delete(vector_copy, np.arange(len_slice))    
    return net_copy

f = open('ScalerData', 'rb')
scaler = pickle.load(f)
f.close()

creator.create("FitnessMaxMin", base.Fitness, weights=(-1.0, -1.0)) # Min: f1 & Min: f2
creator.create("Individual", array.array, typecode='f', fitness=creator.FitnessMaxMin, params=None)

gen = 400
rangefinder_ver = 113
if rangefinder_ver == 112:
    context_skill = True
    gen = 367
else:
    context_skill = False
    gen = 125


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

def eval_ind(gen, context, steer_mult, torque_mult, ind_num):
    if context:
        from neural_net_torch import Context_Skill_Net as S_o_net # Skill-only Model
        with open("rangefinder_v" + str(rangefinder_ver) + "/gen" + str(gen) +"_CS1_checkpoint.pkl", 'rb') as file:
            cp = pickle.load(file)
    else:
        from neural_net_torch import Skill_only_Net as S_o_net # Skill-only Model
        with open("rangefinder_v" + str(rangefinder_ver) + "/gen" + str(gen) +"_CS1_checkpoint.pkl", 'rb') as file:
            cp = pickle.load(file)
    net_sample = S_o_net(n_obs, n_actions)
    pop = cp['population']
    
    
    base_phys = {'flwf': 3.5, 'frwf': 3.5, 'rlwf': 3.5, 'rrwf': 3.5, 'mass': 2090,
                 'flwmsa': 70, 'frwmsa': 70, 'speed': 60, 'steer1': 0.9,
                 'steer2': 0.8, 'steer3': 0.7, 'torque1': 500.76}
    # Note, I believe speed will become less important now...
    
    fric_multiplier = 1
    mass_multiplier = 1
    msa_multiplier = 1
    speed_multiplier = 1
    steer_multiplier = steer_mult
    torque_multiplier = torque_mult
    
    base_phys['flwf'] = fric_multiplier*base_phys['flwf']
    base_phys['frwf'] = fric_multiplier*base_phys['frwf']
    base_phys['rlwf'] = fric_multiplier*base_phys['rlwf']
    base_phys['rrw'] = fric_multiplier*base_phys['rrwf']
    
    base_phys['mass'] = mass_multiplier*base_phys['mass']
    
    base_phys['flwmsa'] = msa_multiplier*base_phys['flwmsa']
    base_phys['frwmsa'] = msa_multiplier*base_phys['frwmsa']
    
    base_phys['speed'] = speed_multiplier*base_phys['speed']
    
    base_phys['steer1'] = steer_multiplier*base_phys['steer1']
    base_phys['steer2'] = steer_multiplier*base_phys['steer2']
    base_phys['steer3'] = steer_multiplier*base_phys['steer3']
    
    base_phys['torque1'] = torque_multiplier*base_phys['torque1']
    
    return Game(genotype_to_phenotype(pop[ind_num], net_sample), scaler, 2006, base_phys, track_num)

ind = get_best_ind_num(rangefinder_ver, gen)
results = {}
for i in np.linspace(0.75, 1.25, 25):
    for j in np.linspace(0.75, 1.25, 25):
        print('i' + str(i) + 'j' + str(j))
        results[(i, j)] = eval_ind(gen, context_skill, i, j, ind)
with open('contour_results_v' + str(rangefinder_ver) + '_gen' + str(gen) + '_Track' + str(track_num) + '_' + ("cs" if context_skill else "sonly"), 'wb') as f:
    pickle.dump(results, f)