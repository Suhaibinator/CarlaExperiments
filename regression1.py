# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:25:34 2020

@author: Suhaib
"""

import carla
import math
import time
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle

def gen_bound(x1, y1, x2, y2):
    angle = math.atan((x1-x2)/(y2-y1))
    deltaX = (3.5/2)*math.cos(angle)
    deltaY = (3.5/2)*math.sin(angle)
    return (x2-1*deltaX, x2+1*deltaX, y2-1*deltaY, y2+1*deltaY)


world = carla.Client('127.0.0.1', 2000).get_world()


minX = 25.669288635253906
minY = 83.67339324951172

waypoints = world.get_map().generate_waypoints(0.5)

selected_waypoints = [x for x in waypoints if x.transform.location.x > minX and x.transform.location.y > minY and x.road_id == 76 and x.lane_id == -2]

selected_x = [i.transform.location.x for i in selected_waypoints]
selected_y = [i.transform.location.y for i in selected_waypoints]

upperLaneX = [0.0]*(len(selected_waypoints)-1)
upperLaneY = [0.0]*(len(selected_waypoints)-1)
lowerLaneX = [0.0]*(len(selected_waypoints)-1)
lowerLaneY = [0.0]*(len(selected_waypoints)-1)
midX = [0.0]*(len(selected_waypoints)-1)
midY = [0.0]*(len(selected_waypoints)-1)

for i in range(1, len(selected_x)):
    result = gen_bound(selected_x[i-1], selected_y[i-1], selected_x[i], selected_y[i])
    upperLaneX[i-1] = result[1]
    upperLaneY[i-1] = result[3]
    lowerLaneX[i-1] = result[0]
    lowerLaneY[i-1] = result[2]
    midX[i-1] = 0.5*(result[1]+result[0])
    midY[i-1] = 0.5*(result[3]+result[2])
for i in range(1, len(upperLaneX)):
    world.debug.draw_line(carla.Location(upperLaneX[i-1], upperLaneY[i-1], 5), carla.Location(upperLaneX[i], upperLaneY[i], 5), life_time=60, color=carla.Color(0,0,255,0))    
    world.debug.draw_line(carla.Location(lowerLaneX[i-1], lowerLaneY[i-1], 5), carla.Location(lowerLaneX[i], lowerLaneY[i], 5), life_time=60, color=carla.Color(0,255,255,0))    
    world.debug.draw_line(carla.Location(midX[i-1], midY[i-1], 5), carla.Location(midX[i], midY[i], 5), life_time=60, color=carla.Color(0,125,155,0))    

for i in range(len(midX)):
    time.sleep(0.05)
    new_transform = carla.Transform(carla.Location(x=midX[i], y=midY[i], z=15), carla.Rotation(pitch=-90, yaw=0, roll=0))
    world.get_spectator().set_transform(new_transform)


rightLane = np.array([[upperLaneX[i], upperLaneY[i]] for i in range(len(lowerLaneX))])
nbrs_right = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(rightLane)
leftLane = np.array([[lowerLaneX[i], lowerLaneY[i]] for i in range(len(lowerLaneX))])
nbrs_left = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(leftLane)
midLane = np.array([[midX[i], midY[i]] for i in range(len(midX))])
center_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(midLane)


with open('reg8.1_data', 'wb') as f:
    pickle.dump((nbrs_right, rightLane, nbrs_left, leftLane, midLane, center_nbrs), f)

