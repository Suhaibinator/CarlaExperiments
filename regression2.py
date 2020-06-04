# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:25:34 2020

@author: Suhaib
"""

import pickle

from sklearn.neighbors import NearestNeighbors
import numpy as np


with open('nbrs', 'rb') as f:
    nbrs, nbrs_right, lowerLaneX, lowerLaneY, upperLaneX, upperLaneY = pickle.load(f)

def classify(test_x, test_y, train_x, train_y):
    _, indices = nbrs.kneighbors(np.array([[test_x, test_y]]))
    index = indices[0][0]
    nearestX = train_x[index]
    nearestY = train_y[index]
    deltaX = 0
    deltaY = 0
    if index > 2:
        deltaX = nearestX-train_x[index-2]
        deltaX = nearestY-train_y[index-2]
    else:
        deltaX = train_x[1]-train_x[0]
        deltaX = lowerLaneY[1]-train_y[0]
    otherDeltaX = test_x-nearestX
    otherDeltaY = test_y-nearestY
    
    cross_result = np.cross([deltaX, deltaY], [otherDeltaX, otherDeltaY])
    return cross_result > 0
    
def classify_lower(test_x, test_y):
    return classify(test_x, test_y, lowerLaneX, lowerLaneY)
def classify_upper(test_x, test_y):
    return classify(test_x, test_y, upperLaneY, upperLaneY)
