#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 03:16:23 2020

@author: suhaib
"""


import carla
from matplotlib import pyplot as plt

waypoints = carla.Client('127.0.0.1', 2000).get_world().get_map().generate_waypoints(0.5)


def count_it(way):
    x = way.transform.location.x
    y = way.transform.location.y
    if way.road_id == 31 and x > 7.32:
        return True
    if way.road_id == 16 and x > 7.01:
        return True
    if way.road_id == 532 and way.lane_id == -3 and y < -13:
        return True
    if way.road_id == 532 and way.lane_id == -4 and x > 16.6:
        return True
    if x > 16.7 and y > -15.2 and x < 18.17 and y < -13.5:
        return True
    if x > 7.4 and x < 7.8 and y > -44 and y <-42:
        return True
    if x > 25 and x < 79 and y < -7.2 and y > -50:
        return True
    return False

selected_waypoints = [x for x in waypoints if count_it(x)]

all_x = [i.transform.location.x for i in waypoints]
all_y = [i.transform.location.y for i in waypoints]

selected_x = [i.transform.location.x for i in selected_waypoints]
selected_y = [i.transform.location.y for i in selected_waypoints]

tgt_x = 229.689
tgt_y = 70.366

dists = [(all_x[i]-tgt_x)**2 + (all_y[i]-tgt_y)**2 for i in range(len(all_x))]
print(dists.index(min(dists)))

plt.plot(all_x, all_y, '*', selected_x, selected_y, '*')

