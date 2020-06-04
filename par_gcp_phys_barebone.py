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
    sys.path.append('/home/suhaib_abdulquddos/carla_files/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg')
    sys.path.append('/home/suhaib/Desktop/GRA/carla2/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg')
    import carla

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref




from regression3 import get_distance

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
    def __init__(self, carla_world, actor_filter, phys_settings, actor_role_name='hero'):
        self.world = carla_world
        self.actor_role_name = actor_role_name
        self.map = self.world.get_map()
        self.player = None
        self.speed = 60
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
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

    def restart(self):
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        while self.player is None:
            new_transform = carla.Transform(carla.Location(x=229.8, y=81.1, z=1), carla.Rotation(pitch=0, yaw=92.0042, roll=0))
            self.world.get_spectator().set_transform(new_transform)
            self.player = self.world.try_spawn_actor(blueprint, new_transform)        # Set up the sensors.

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
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()
    def eval_reward(self):
        current_loc = self.player.get_transform().location
        v = self.player.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        self.f0 += 10 if current_loc.y < 81 or speed < 5.0 else 0
        self.f0 += get_distance(current_loc.x, current_loc.y)
        #print("F0 val: " + str(self.f0))

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
        #print("Distance: " + str((pos.x-25)**2+(pos.y-193.7)**2))
        if (pos.x-25)**2+((pos.y-193.7)/6)**2 < 20:
            return 5
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                v = world.player.get_velocity()
                #speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                
                current_transform = world.player.get_transform()
                pos = current_transform.location
                
                self.calculate_steering_throttle(v.x, v.y, pos.x, pos.y, current_transform.rotation.yaw)
                
                #self._control.throttle = 1.0 if speed < self.world.speed else 0.0
                #self._control.reverse = self._control.gear < 0
                

            world.player.apply_control(self._control)

    def calculate_steering_throttle(self, vel_x, vel_y, pos_x, pos_y, yaw):
        with torch.no_grad():
            chosen_action = self._net(torch.FloatTensor([vel_x, vel_y, pos_x, pos_y, yaw]))
            #print("Action: " + str(chosen_action))
            self._steer_cache = chosen_action[0].item()
            #self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
            self._control.steer = round(self._steer_cache, 1)
            self._control.throttle = chosen_action[1].item()
            #self._control.steer = 0



# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args, net, scaler, port, phys_settings):
    world = None
    
    timestep = 0.1

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

        #if (client.get_world().get_map().name != "Town03"):
        #    client.load_world('Town03')
        world = World(client.get_world(), args.filter, phys_settings, args.rolename)
        
        controller = KeyboardControl(world, args.autopilot, net, scaler)

        settings = world.world.get_settings()
        if not settings.synchronous_mode or settings.fixed_delta_seconds != timestep:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = timestep
            world.world.apply_settings(settings)


        for i in range(math.ceil(20/timestep)):
            world.world.tick()
            result = controller.parse_events(client, world)
            if result == 5:
                current_transform = world.player.get_transform()
                pos = current_transform.location
                return world.f0, math.sqrt((pos.x-25)**2+(pos.y-193.7)**2)
            elif result:
                return
            
            world.eval_reward()

        current_transform = world.player.get_transform()
        pos = current_transform.location
        f0 = world.f0
        f1 = math.sqrt((pos.x-25)**2+(pos.y-193.7)**2)
    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

    return f0, f1

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def Game(neural_net, scaler, port, phys_settings):
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

        f0, f1 = game_loop(args, neural_net, scaler, port, phys_settings)
        print

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    return f0, f1
