#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 01:53:47 2020

@author: suhaib
"""

import math
import numpy as np
import pickle

line_dist = 8
num_points = 161 # This should be an odd number

with open('reg5_data', 'rb') as f:
    nbrs_right, nbrs_left = pickle.load(f)

def generate_rays(pos_x, pos_y, yaw):
    angle = 90-yaw
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
    return distances_right[ind_r][0] if math.sqrt((pos_x-ray[ind_r][0])**2+(pos_y-ray[ind_r][1])**2) < math.sqrt((pos_x-ray[ind_l][0])**2+(pos_y-ray[ind_l][1])**2) else distances_left[ind_l][0]
