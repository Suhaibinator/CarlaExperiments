#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 02:18:15 2020

@author: suhaib
"""

import math
import numpy as np
import pickle

line_dist = 10
num_points = 321 # This should be an odd number

mid_ind = int(num_points/2.0)

with open('reg8.3_data', 'rb') as f:
    #8.3 is track A new
    #8.2 is track B new
    #8.4 is fixed track B new
    nbrs_right, rightLane, nbrs_left, leftLane, midLane, center_nbrs = pickle.load(f)

def generate_rays(pos_x, pos_y, yaw):
    angle = yaw
    angle3 = angle
    angle2 = angle+35
    angle1 = angle+70
    angle4 = angle-35
    angle5 = angle-70
    
    ep1_ray1_x = pos_x + line_dist*math.cos(angle1*math.pi/180)
    ep1_ray1_y = pos_y + line_dist*math.sin(angle1*math.pi/180)
    ep2_ray1_x = pos_x# - line_dist*math.cos(angle1*math.pi/180)
    ep2_ray1_y = pos_y# - line_dist*math.sin(angle1*math.pi/180)
    x1 = np.linspace(ep1_ray1_x, ep2_ray1_x, num_points)
    y1 = np.linspace(ep1_ray1_y, ep2_ray1_y, num_points)
    
    ep1_ray2_x = pos_x + line_dist*math.cos(angle2*math.pi/180)
    ep1_ray2_y = pos_y + line_dist*math.sin(angle2*math.pi/180)
    ep2_ray2_x = pos_x# - line_dist*math.cos(angle2*math.pi/180)
    ep2_ray2_y = pos_y# - line_dist*math.sin(angle2*math.pi/180)
    x2 = np.linspace(ep1_ray2_x, ep2_ray2_x, num_points)
    y2 = np.linspace(ep1_ray2_y, ep2_ray2_y, num_points)
    
    ep1_ray3_x = pos_x + line_dist*math.cos(angle3*math.pi/180)
    ep1_ray3_y = pos_y + line_dist*math.sin(angle3*math.pi/180)
    ep2_ray3_x = pos_x# - line_dist*math.cos(angle3*math.pi/180)
    ep2_ray3_y = pos_y# - line_dist*math.sin(angle3*math.pi/180)
    x3 = np.linspace(ep1_ray3_x, ep2_ray3_x, num_points)
    y3 = np.linspace(ep1_ray3_y, ep2_ray3_y, num_points)
    
    ep1_ray4_x = pos_x + line_dist*math.cos(angle4*math.pi/180)
    ep1_ray4_y = pos_y + line_dist*math.sin(angle4*math.pi/180)
    ep2_ray4_x = pos_x# - line_dist*math.cos(angle4*math.pi/180)
    ep2_ray4_y = pos_y# - line_dist*math.sin(angle4*math.pi/180)
    x4 = np.linspace(ep1_ray4_x, ep2_ray4_x, num_points)
    y4 = np.linspace(ep1_ray4_y, ep2_ray4_y, num_points)
    
    ep1_ray5_x = pos_x + line_dist*math.cos(angle5*math.pi/180)
    ep1_ray5_y = pos_y + line_dist*math.sin(angle5*math.pi/180)
    ep2_ray5_x = pos_x# - line_dist*math.cos(angle5*math.pi/180)
    ep2_ray5_y = pos_y# - line_dist*math.sin(angle5*math.pi/180)
    x5 = np.linspace(ep1_ray5_x, ep2_ray5_x, num_points)
    y5 = np.linspace(ep1_ray5_y, ep2_ray5_y, num_points)
    
    return np.array([[x1[i], y1[i]] for i in range(num_points)]),\
        np.array([[x2[i], y2[i]] for i in range(num_points)]),\
        np.array([[x3[i], y3[i]] for i in range(num_points)]),\
        np.array([[x4[i], y4[i]] for i in range(num_points)]),\
        np.array([[x5[i], y5[i]] for i in range(num_points)])

def find_intersect_dist(pos_x, pos_y, ray):
    distances_right, _ = nbrs_right.kneighbors(ray)
    distances_left, _ = nbrs_left.kneighbors(ray)
    ind_r = np.argmin(distances_right)
    ind_l = np.argmin(distances_left)
    if min(distances_left[ind_l], distances_right[ind_r]) > 1.2 and classify_left(pos_x, pos_y) != classify_right(pos_x, pos_y):
        return math.sqrt((ray[0, 0] - pos_x)**2 + (ray[0, 1] - pos_y)**2)
    return max(math.sqrt((pos_x-ray[ind_r][0])**2+(pos_y-ray[ind_r][1])**2), math.sqrt((pos_x-ray[ind_l][0])**2+(pos_y-ray[ind_l][1])**2))

def find_intersect_point(pos_x, pos_y, ray):
    distances_right, _ = nbrs_right.kneighbors(ray)
    distances_left, _ = nbrs_left.kneighbors(ray)
    ind_r = np.argmin(distances_right)
    ind_l = np.argmin(distances_left)
    if min(distances_left[ind_l], distances_right[ind_r]) > 1.2 and classify_left(pos_x, pos_y) != classify_right(pos_x, pos_y):
        return ray[0, 0], ray[0, 1]
    if math.sqrt((pos_x-ray[ind_r][0])**2+(pos_y-ray[ind_r][1])**2) > math.sqrt((pos_x-ray[ind_l][0])**2+(pos_y-ray[ind_l][1])**2):
        return ray[ind_r][0], ray[ind_r][1]
    return ray[ind_l][0], ray[ind_l][1]

def get_distance(test_x, test_y):
    dist, indices = center_nbrs.kneighbors(np.array([[test_x, test_y]]))
    index = indices[0][0]
    nearestX = midLane[index][0]
    nearestY = midLane[index][1]
    deltaX = 0
    deltaY = 0
    if index > 2:
        deltaX = nearestX-midLane[index-2][0]
        deltaY = nearestY-midLane[index-2][1]
    else:
        deltaX = midLane[1][0]-midLane[0][0]
        deltaY = midLane[1][1]-midLane[0][1]
    otherDeltaX = test_x-nearestX
    otherDeltaY = test_y-nearestY
    
    cross_result = np.cross([deltaX, deltaY], [otherDeltaX, otherDeltaY])
    return (cross_result/abs(cross_result))*dist[0][0]

def classify(test_x, test_y, train_x, train_y):
    _, indices = nbrs_right.kneighbors(np.array([[test_x, test_y]]))
    index = indices[0][0]
    nearestX = train_x[index]
    nearestY = train_y[index]
    deltaX = 0
    deltaY = 0
    if index > 2:
        deltaX = nearestX-train_x[index-2]
        deltaY = nearestY-train_y[index-2]
    else:
        deltaX = train_x[2]-train_x[0]
        deltaY = train_y[2]-train_y[0]
    otherDeltaX = test_x-nearestX
    otherDeltaY = test_y-nearestY
    
    cross_result = np.cross([deltaX, deltaY], [otherDeltaX, otherDeltaY])
    return cross_result > 0
    
def classify_left(test_x, test_y):
    return classify(test_x, test_y, leftLane[:, 0], leftLane[:, 1])
def classify_right(test_x, test_y):
    return classify(test_x, test_y, rightLane[:, 0], rightLane[:, 1])