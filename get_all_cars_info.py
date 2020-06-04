#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:49:44 2020

@author: suhaib
"""


from par_gcp_phys_barebone import World
from matplotlib import pyplot as plt

cars = ["vehicle.dodge_charger.police", "vehicle.bmw.isetta", "vehicle.audi.tt", "vehicle.audi.a2", "vehicle.nissan.patrol", "vehicle.tesla.model3"]
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)
client.get_world().tick()

data = {}
for car in cars:
    
    world = World(client.get_world(), car, None, "Hero")
    
    [client.get_world().tick() for i in range(4)]
    
    data[car] = world.player.get_physics_control()
    world.destroy()
    print(car)
    print(data[car].torque_curve[2].y)
    [client.get_world().tick() for i in range(4)]
    plt.plot([i.x for i in data[car].steering_curve], [i.y for i in data[car].steering_curve])

plt.legend(cars)

for i, j in data.items():
    print(i)
    print([(f.x, f.y) for f in j.torque_curve])
    print()
