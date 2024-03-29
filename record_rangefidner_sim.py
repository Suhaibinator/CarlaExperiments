#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import torch
import sys
try:
    import carla
except:
    sys.path.append('/home/suhaib/carla_files/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg')
    import carla


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

early_termination = False

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import numpy as np
import pickle

import regression9 as main_reg

track = 3

target_x = 25	
target_y = 193.7	
starting_x = 229.8	
starting_y = 81.1	
starting_yaw = 92.0042	
if track == 2:	
    t = target_x	
    target_x = starting_x	
    starting_x = t	
    t = target_y	
    target_y = starting_y	
    starting_y = t	
    starting_yaw = 0	
elif track == 3:	
    starting_x = 7.55	
    starting_y = -66	
    yaw = 90	
    target_x = 78.7	
    target_y = -49.6

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, actor_filter, phys_settings, cam_path, actor_role_name='hero'):
        self.world = carla_world
        self.actor_role_name = actor_role_name
        self.map = self.world.get_map()
        self.player = None
        self.speed = 60
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.cam_path = cam_path
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        if phys_settings is not None:
            self.apply_physics(phys_settings)
        self.recording_enabled = False
        self.recording_start = 0
        self.f0 = 0
        self.f1 = 0
        self.recording_start = 0

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        while self.player is None:
            new_transform = carla.Transform(carla.Location(x=starting_x, y=starting_y, z=1), carla.Rotation(pitch=0, yaw=starting_yaw, roll=0))
            self.world.get_spectator().set_transform(new_transform)
            self.player = self.world.try_spawn_actor(blueprint, new_transform)        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.cam_path)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        #self.player.set_autopilot(True)
    
    def apply_physics(self, phys_settings):
        self.world.tick()
        """
        Default wheel settings:
        front_left_wheel  = carla.WheelPhysicsControl(tire_friction=3.5, damping_rate=0.25, max_steer_angle=70, radius=36.7)
        front_right_wheel = carla.WheelPhysicsControl(tire_friction=3.5, damping_rate=0.25, max_steer_angle=70, radius=36.7)
        rear_left_wheel   = carla.WheelPhysicsControl(tire_friction=3.5, damping_rate=0.25, max_steer_angle=0.0,  radius=36.0)
        rear_right_wheel  = carla.WheelPhysicsControl(tire_friction=3.5, damping_rate=0.25, max_steer_angle=0.0,  radius=36.0)
        """


        front_left_wheel  = carla.WheelPhysicsControl(tire_friction=phys_settings['flwf'], damping_rate=0.25, max_steer_angle=phys_settings['flwmsa'], radius=36.7)
        front_right_wheel = carla.WheelPhysicsControl(tire_friction=phys_settings['frwf'], damping_rate=0.25, max_steer_angle=phys_settings['frwmsa'], radius=36.7)
        rear_left_wheel   = carla.WheelPhysicsControl(tire_friction=phys_settings['rlwf'], damping_rate=0.25, max_steer_angle=0.0,  radius=36.0)
        rear_right_wheel  = carla.WheelPhysicsControl(tire_friction=phys_settings['rrwf'], damping_rate=0.25, max_steer_angle=0.0,  radius=36.0)
    
        wheels = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]
    
        # Change Vehicle Physics Control parameters of the vehicle
        physics_control = self.player.get_physics_control()
    
        #physics_control.max_rpm = phys_settings['max_rpm']
        physics_control.mass = phys_settings['mass']
        #physics_control.drag_coefficient = phys_settings['drag_coefficient']
        physics_control.wheels = wheels
        
        physics_control.steering_curve = [carla.Vector2D(0, 1), carla.Vector2D(20, phys_settings['steer1']), carla.Vector2D(60, phys_settings['steer2']), carla.Vector2D(120, phys_settings['steer3'])]
        physics_control.torque_curve = [carla.Vector2D(0, 400), carla.Vector2D(890, phys_settings['torque1']), carla.Vector2D(5729.577, 400)]
        
        self.speed = phys_settings['speed']
        
        # Apply Vehicle Physics Control for the vehicle
        self.player.apply_physics_control(physics_control)
        #rot = self.player.get_transform().rotation.yaw
        #self.player.set_velocity(carla.Vector3D(self.speed*math.cos(rot*math.pi/180)/3.6, self.speed*math.sin(rot*math.pi/180)/3.6, 0))

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.player.get_world().set_weather(preset[0])


    def render(self, display):
        self.camera_manager.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()
    def eval_reward(self):
        current_loc = self.player.get_transform().location
        v = self.player.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        to_add = abs(main_reg.get_distance(current_loc.x, current_loc.y))
        self.f0 += 10 if current_loc.y < 81 or speed < 5.0 else 0
        self.f0 += to_add
        return True
        #print("F0 val: " + str(self.f0))
        
    def eval_target_distance_reward(self):
        pos = self.player.get_location()
        return 10*math.sqrt((pos.x-target_x)**2+(pos.y-target_y)**2)
        
# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot, net, scaler):
        self._net = net
        self._scaler = scaler
        self.world = world
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            #world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0


    def parse_events(self, client, world):
        current_transform = world.player.get_transform()
        pos = current_transform.location
        if (pos.x-target_x)**2+((pos.y-target_y))**2 < 10:
            return 5
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                current_transform = world.player.get_transform()
                pos = current_transform.location
                self.calculate_steering_throttle(pos.x, pos.y, current_transform.rotation.yaw, world)
            world.player.apply_control(self._control)

    def calculate_steering_throttle(self, pos_x, pos_y, yaw, world):
        ray1, ray2, ray3, ray4, ray5 = main_reg.generate_rays(pos_x, pos_y, yaw)
        dist1 = main_reg.find_intersect_dist(pos_x, pos_y, ray1)
        dist2 = main_reg.find_intersect_dist(pos_x, pos_y, ray2)
        dist3 = main_reg.find_intersect_dist(pos_x, pos_y, ray3)
        dist4 = main_reg.find_intersect_dist(pos_x, pos_y, ray4)
        dist5 = main_reg.find_intersect_dist(pos_x, pos_y, ray5)
        
        
        x1r, y1r = main_reg.find_intersect_point(pos_x, pos_y, ray1)
        x2r, y2r = main_reg.find_intersect_point(pos_x, pos_y, ray2)
        x3r, y3r = main_reg.find_intersect_point(pos_x, pos_y, ray3)
        x4r, y4r = main_reg.find_intersect_point(pos_x, pos_y, ray4)
        x5r, y5r = main_reg.find_intersect_point(pos_x, pos_y, ray5)
        
        world.world.debug.draw_line(carla.Location(x1r, y1r, world.player.get_location().z+1), carla.Location(ray1[0][0], ray1[0][1], world.player.get_location().z+1), life_time=world.timestep, color=carla.Color(0,255,0,0))
        world.world.debug.draw_line(carla.Location(x2r, y2r, world.player.get_location().z+1), carla.Location(ray2[0][0], ray2[0][1], world.player.get_location().z+1), life_time=world.timestep, color=carla.Color(0,255,0,0))
        world.world.debug.draw_line(carla.Location(x3r, y3r, world.player.get_location().z+1), carla.Location(ray3[0][0], ray3[0][1], world.player.get_location().z+1), life_time=world.timestep, color=carla.Color(0,255,0,0))
        world.world.debug.draw_line(carla.Location(x4r, y4r, world.player.get_location().z+1), carla.Location(ray4[0][0], ray4[0][1], world.player.get_location().z+1), life_time=world.timestep, color=carla.Color(0,255,0,0))
        world.world.debug.draw_line(carla.Location(x5r, y5r, world.player.get_location().z+1), carla.Location(ray5[0][0], ray5[0][1], world.player.get_location().z+1), life_time=world.timestep, color=carla.Color(0,255,0,0))
        
        world.world.debug.draw_line(carla.Location(ray1[-1][0], ray1[-1][1], world.player.get_location().z+1), carla.Location(x1r, y1r, world.player.get_location().z+1), life_time=world.timestep)
        world.world.debug.draw_line(carla.Location(ray2[-1][0], ray2[-1][1], world.player.get_location().z+1), carla.Location(x2r, y2r, world.player.get_location().z+1), life_time=world.timestep)
        world.world.debug.draw_line(carla.Location(ray3[-1][0], ray3[-1][1], world.player.get_location().z+1), carla.Location(x3r, y3r, world.player.get_location().z+1), life_time=world.timestep)
        world.world.debug.draw_line(carla.Location(ray4[-1][0], ray4[-1][1], world.player.get_location().z+1), carla.Location(x4r, y4r, world.player.get_location().z+1), life_time=world.timestep)
        world.world.debug.draw_line(carla.Location(ray5[-1][0], ray5[-1][1], world.player.get_location().z+1), carla.Location(x5r, y5r, world.player.get_location().z+1), life_time=world.timestep)
        
        world.world.debug.draw_string(carla.Location(ray5[-1][0], ray5[-1][1], world.player.get_location().z+6), str(math.sqrt(world.player.get_velocity().x**2 + world.player.get_velocity().y**2)), draw_shadow=False, life_time=world.timestep*2)
        
        with torch.no_grad():
            chosen_action = self._net(torch.FloatTensor([dist1, dist2, dist3, dist4, dist5]))
            #print("Action: " + str(chosen_action))
            self._steer_cache = chosen_action[0].item()
            self._control.steer = round(self._steer_cache, 1)
            self._control.throttle = chosen_action[1].item()

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame_number, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
        self.broken = []
        self.solid = []
        self.waypoints = [[], [], [], [], []]

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, cam_path):
        self.sensor = None
        self.cam_path = cam_path
        self.surface = None
        self._parent = parent_actor
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-8, z=12), carla.Rotation(pitch=-45)),
            carla.Transform(carla.Location(x=1.0, z=8.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', '1280')
                bp.set_attribute('image_size_y', '720')
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
            
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(1280, 720) / 100.0
            lidar_data += (0.5 * 1280, 0.5 * 720)
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (1280, 720, 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
        if self.recording:
            image.save_to_disk('_out/' + self.cam_path +'/%08d' % image.frame_number)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args, net, scaler, port, phys_settings, cam_path):
    world = None
    
    timestep = 1/30

    """
    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)
    args = Bunch({'autopilot':False, 'debug':False, 'filter':'vehicle.tesla.model3', 'height':720, 'host':'127.0.0.1', 'port':2000, 'res':'1280x720', 'rolename':'hero', 'width':1280})
    """
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        
        print(args)

        if (client.get_world().get_map().name != "Town03"):
            client.load_world('Town03')
        
        world = World(client.get_world(), args.filter, phys_settings, cam_path, args.rolename)
        world.timestep = timestep
        world.camera_manager.toggle_recording()
        controller = KeyboardControl(world, args.autopilot, net, scaler)
        if False:
            
            if track != 3:
                with open('reg8.3_data', 'rb') as f:
                    nbrs_right, rightLane, nbrs_left, leftLane, midLane, center_nbrs = pickle.load(f)
                for i in range(1, len(midLane)):
                    world.world.debug.draw_line(carla.Location(midLane[i-1][0], midLane[i-1][1], 2), carla.Location(midLane[i][0], midLane[i][1], 2), life_time=20, color=carla.Color(0,125,155,0))    
            else:
                with open('reg8.4_data', 'rb') as f:
                    nbrs_right, rightLane, nbrs_left, leftLane, midLane, center_nbrs = pickle.load(f)
                for i in range(1, len(midLane)):
                    world.world.debug.draw_line(carla.Location(midLane[i-1][0], midLane[i-1][1], 2), carla.Location(midLane[i][0], midLane[i][1], 2), life_time=25, color=carla.Color(0,125,155,0))
                    world.world.debug.draw_line(carla.Location(leftLane[i-1][0], leftLane[i-1][1], 2), carla.Location(leftLane[i][0], leftLane[i][1], 2), life_time=25, color=carla.Color(0,250,155,0))
                    world.world.debug.draw_line(carla.Location(rightLane[i-1][0], rightLane[i-1][1], 2), carla.Location(rightLane[i][0], rightLane[i][1], 2), life_time=25, color=carla.Color(205,125,155,0))


        data = []
        vel_data = []
        settings = world.world.get_settings()
        if not settings.synchronous_mode or settings.fixed_delta_seconds != timestep:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = timestep
            world.world.apply_settings(settings)


        for i in range(math.ceil(25/timestep)):
            world.world.tick()
            result = controller.parse_events(client, world)
            data.append((world.player.get_location().x, world.player.get_location().y, world.player.get_transform().rotation.yaw))
            data.append((world.player.get_velocity().x, world.player.get_velocity().y))
            if result == 5:
                with open('loc_data', 'wb') as f:
                    pickle.dump(data, f)
                with open('vel_data', 'wb') as f:
                    pickle.dump(vel_data, f)
                return world.f0, world.eval_target_distance_reward()
            elif result:
                return
            world.eval_reward()

        current_transform = world.player.get_transform()
        pos = current_transform.location
        f0 = world.f0
        f1 = world.eval_target_distance_reward()
    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()
    import os
    print(os.getcwd())
    with open('loc_data', 'wb') as f:
        pickle.dump(data, f)
    with open('vel_data', 'wb') as f:
        pickle.dump(vel_data, f)
    return f0, f1

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def Game(neural_net, scaler, port, phys_settings, cam_path):
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=port,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='actor filter (default: "vehicle.tesla.model3")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        f0, f1 = game_loop(args, neural_net, scaler, port, phys_settings, cam_path)
        print

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    return f0, f1
