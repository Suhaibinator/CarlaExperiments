#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 02:34:32 2020

@author: suhaib
"""


from record_rangefidner_sim import Game

import time

from deap import base, creator
import pickle
import array
import copy
import torch
import numpy as np


n_obs = 5 # number of inputs
n_actions = 2 # number of outputs


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

def record_sim_to_video(gen, context, steer_mult, torque_mult, ind_num, rangefinder_ver):
    import os
    if context:
        from neural_net_torch import Context_Skill_Net as S_o_net # Skill-only Model
        with open("rangefinder_v" + str(rangefinder_ver) + "/gen" + str(gen) +"_CS1_checkpoint.pkl", 'rb') as file:
            cp = pickle.load(file)
        #with open("context_skill_six/gen" + str(gen) +"_CS1_checkpoint.pkl", 'rb') as file:
        #    cp = pickle.load(file)
    else:
        from neural_net_torch import Skill_only_Net as S_o_net # Skill-only Model
        
        with open("skill_only_six/gen" + str(gen) +"_CS1_checkpoint.pkl", 'rb') as file:
            cp = pickle.load(file)
    net_sample = S_o_net(n_obs, n_actions)
    pop = cp['population']
    
    dir_name = "rangefinder_v" + str(rangefinder_ver) + ("context_" if context else "skill_") + "gen" + str(gen) + "_steer" + str(steer_mult) + "_torque" + str(torque_mult) + "_ind" + str(ind_num)
    os.mkdir('./_out/'+dir_name)
    
    
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
    
    print(Game(genotype_to_phenotype(pop[ind_num], net_sample), scaler, 2000, base_phys, dir_name))
    #print(Game(genotype_to_phenotype(pop[ind_num], net_sample), scaler, 2000, base_phys))
    
    import subprocess
    #import glob
    import os
    
    SCREENWIDTH  = 1280
    SCREENHEIGHT = 720
    FPS          = 10
    
    orig_dir = os.getcwd()
    os.chdir('./_out/' + dir_name)
    
    image_files = []
    for file in os.listdir(os.getcwd()):
        if file.endswith(".png"):
            image_files.append(file)
    
    for index, oldfile in enumerate(sorted(image_files), start=1):
        newfile = '%06d.png' % (index)
        os.rename (oldfile, newfile)
    
    command = "ffmpeg -framerate " + str(FPS) + " -r " + str(FPS) + " -s " + str(SCREENWIDTH) + "x" + str(SCREENHEIGHT) + \
    " -i %06d.png -c:v libx264 -crf 10 -pix_fmt yuv420p " + str(dir_name) + ".mp4"
    
    subprocess.call(command, shell=True)
    os.chdir(orig_dir)