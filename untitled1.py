#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:19:08 2020

@author: suhaib
"""


from matplotlib import pyplot as plt
import pickle

with open("Coords", 'rb') as f:
    coords = pickle.load(f)
    
x = [i[0] for i in coords]
y = [i[1] for i in coords]

with open("Coords_dodge", 'rb') as f:
    coords = pickle.load(f)
    
x2 = [i[0] for i in coords]
y2 = [i[1] for i in coords]

plt.plot(x, y, x2, y2)