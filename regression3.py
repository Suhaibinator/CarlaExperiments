#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 00:12:11 2020

@author: suhaib
"""


import pickle

from sklearn.neighbors import NearestNeighbors
import numpy as np

with open('nbrs_center', 'rb') as f:
    nbrs_center = pickle.load(f)
    
def get_distance(test_x, test_y):
    dist, _ = nbrs_center.kneighbors(np.array([[test_x, test_y]]))
    return dist[0][0]
