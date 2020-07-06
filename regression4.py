#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import math
from sklearn.neighbors import NearestNeighbors
import numpy as np


with open('nbrs_center_with_coords.pkl', 'rb') as f:
    nbrs_center, train_x, train_y = pickle.load(f)
with open('sparse_points.pkl', 'rb') as f:
    points, sparse_nbrs_model = pickle.load(f)

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

def get_new_target(current_x, current_y, current_target_x, current_target_y, current_target_rank):
    #result, indices = nbrs.kneighbors(np.array([[current_x, current_y]]))
    if math.sqrt((current_x - current_target_x)**2 + (current_y - current_target_y)**2) < 2:
        new_point = points[min(len(points), current_target_rank+5)]
        return new_point[0], new_point[1], current_target_rank+5
    return current_target_x, current_target_y, current_target_rank