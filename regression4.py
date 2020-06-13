#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

from sklearn.neighbors import NearestNeighbors
import numpy as np


with open('nbrs_center_with_coords.pkl', 'rb') as f:
    nbrs_center, train_x, train_y = pickle.load(f)

def get_distance(test_x, test_y):
    dist, indices = nbrs_center.kneighbors(np.array([[test_x, test_y]]))
    index = indices[0][0]
    nearestX = train_x[index]
    nearestY = train_y[index]
    deltaX = 0
    deltaY = 0
    if index > 2:
        deltaX = nearestX-train_x[index-2]
        deltaY = nearestY-train_y[index-2]
    else:
        deltaX = train_x[1]-train_x[0]
        deltaY = train_y[1]-train_y[0]
    otherDeltaX = test_x-nearestX
    otherDeltaY = test_y-nearestY
    
    cross_result = np.cross([deltaX, deltaY], [otherDeltaX, otherDeltaY])
    return (cross_result/abs(cross_result))*dist[0][0]