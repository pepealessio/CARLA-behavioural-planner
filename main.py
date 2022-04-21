#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division
from datetime import datetime

# System level imports
import sys
import os
import argparse
import logging
import time
import math
from matplotlib import markers
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.core.defchararray import index
import controller2d
import configparser 
import local_planner
import behavioural_planner
import cv2
import json 
from math import sin, cos, pi, tan, sqrt
from munch import DefaultMunch
import tkinter as tk
import threading
from tqdm import tqdm
from shapely.geometry import Point, LineString, Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import live_plotter as lv   # Custom live plotting library
from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils
from carla.sensor import Camera
from carla.image_converter import labels_to_array, depth_to_array, to_bgra_array
from carla.planner.city_track import CityTrack

import warnings
warnings.filterwarnings("ignore")

# Do not print in the std output. All print will be redirected in the log file
_print = print
print = logging.info


###############################################################################
# CONFIGURABLE PARAMENTERS DURING EXAM
###############################################################################
PLAYER_START_INDEX     = 57     # Spawn index for player
DESTINATION_INDEX      = 55     # Setting a Destination HERE
NUM_PEDESTRIANS        = 1      # total number of pedestrians to spawn
NUM_VEHICLES           = 1      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer
###############################################################################

###############################################################################
# ---------------------- PERSONAL ---------------------------------------------
###############################################################################
USE_CAMERA = True      # See the car camera (useful in non-local server)
VISUALIZE_STATE_INFO = True  # See a window with the state info
TRAFFIC_LIGHT_STOP_LINE_LEN = 4     # The lenght of the stop line for traffic lights
###############################################################################

ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 5000.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode
                                # used to make sure the server loads
                                # consistently

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1    
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
                                       # simulation ends

# Planning Constants
NUM_PATHS = 7
BP_LOOKAHEAD_BASE      = 16.0              # m
BP_LOOKAHEAD_TIME      = 1.0              # s
PATH_OFFSET            = 1.5              # m
CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0] # m
CIRCLE_RADII           = [1.5, 1.5, 1.5]  # m
TIME_GAP               = 1.0              # s
PATH_SELECT_WEIGHT     = 10
A_MAX                  = 2.5              # m/s^2
SLOW_SPEED             = 2.0              # m/s
STOP_LINE_BUFFER       = 3.5              # m
LEAD_VEHICLE_LOOKAHEAD = 20.0             # m
LP_FREQUENCY_DIVISOR   = 2                # Frequency divisor to make the 
                                          # local planner operate at a lower
                                          # frequency than the controller
                                          # (which operates at the simulation
                                          # frequency). Must be a natural
                                          # number.

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # selected path
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'

# Camera parameters
camera_parameters = {}
camera_parameters['x'] = 1.8
camera_parameters['y'] = 0
camera_parameters['z'] = 1.3
camera_parameters['width'] = 640
camera_parameters['height'] = 480
camera_parameters['fov'] = 120

    
class StateInfo(threading.Thread):
    """Thread class used to print information about the state of the vehicle and the FSM in a window.
    
    Arguments:
        _show_on(bool, default=True): if true the window will be constructed and showed. Otherwise nothing
        happen with every method."""

    def __init__(self, show_on=True):
        """Create the info window.
        
        Args:
            show_on (bool, default=True): if true the window will be constructed and showed. Otherwise nothing
            happen with every method."""
        threading.Thread.__init__(self)

        self.daemon = True  # Set thread ad daemon thread

        self._show_on = show_on

        if self._show_on:
            # Build the window
            self._root = tk.Tk()
            self._root.protocol("WM_DELETE_WINDOW", self._callback)
            self._root.geometry("1300x400")
            self._root.title("FSM info")
            self._root.configure(background='white')

            self._root.grid_columnconfigure(0, weight=1)
            self._root.grid_columnconfigure(1, weight=1)
            self._root.grid_rowconfigure(1, weight=1)
            self._root.grid_rowconfigure(3, weight=1)

            tk.Label(self._root, text='Simulation params:', font=("Microsoft YaHei UI", 16), bg='white', anchor="w").grid(row=0, column=0, sticky='nswe')

            self._param_text = tk.Text(self._root, width=15, height=0, font=("Microsoft YaHei UI Light", 13))
            self._param_text.bind("<Key>", lambda e: "break")
            self.set_param_info('loading ...')
            self._param_text['bg'] = 'white'
            self._param_text.grid(row=1, column=0, sticky='nswe')

            tk.Label(self._root, text='Behavioural Planner info:', font=("Microsoft YaHei UI", 16), bg='white', anchor="w").grid(row=2, column=0, sticky='nswe')

            self._text = tk.Text(self._root, width=15, height=8, font=("Microsoft YaHei UI Light", 13))
            self._text.bind("<Key>", lambda e: "break")
            self.set_state_info('loading ...')
            self._text['bg'] = 'white'
            self._text.grid(row=3, column=0, sticky='nswe')
            
            self._draw = tk.PanedWindow(self._root)
            
            self._fig = Figure(figsize=(5, 5), dpi=100)
            self._ax = self._fig.add_subplot(111)

            self._canvas = FigureCanvasTkAgg(self._fig, master=self._draw)  # A tk.DrawingArea.
            self._canvas.draw()
            self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self._canvas.show()

            self._draw.grid(row=0, column=1, rowspan=4, sticky='nswe')

            # Start the window in the thread to not stop the main
            self.start()

    def _callback(self):
        self._root.quit()

    def run(self):
        # The try structure is needed to avoid the RuntimeError who inform that the root window will be executed in a non-main thread.
        try:
            self._root.mainloop()
        except KeyboardInterrupt: # RuntimeError:
            pass

    def set_param_info(self, info):
        """Update the param info box."""
        if self._show_on:
            self._param_text.delete('1.0', tk.END)
            self._param_text.insert(tk.END, info)

    def set_state_info(self, info):
        """Update the state info box."""
        if self._show_on:
            self._text.delete('1.0', tk.END)
            self._text.insert(tk.END, info)

    def draw_things(self, *args):
        """Draw in the live plt image."""
        for arg in args:
            self._ax.plot(*arg[0], color=arg[1], marker='.', markersize=20)
        self._canvas.draw()
        self._ax.clear()

    def quit(self):
        """Destroy the window"""
        if self._show_on:
            self._root.destroy()

def rotate_x(angle):
    R = np.mat([[ 1,         0,           0],
                 [ 0, cos(angle), -sin(angle) ],
                 [ 0, sin(angle),  cos(angle) ]])
    return R

def rotate_y(angle):
    R = np.mat([[ cos(angle), 0,  sin(angle) ],
                 [ 0,         1,          0 ],
                 [-sin(angle), 0,  cos(angle) ]])
    return R

def rotate_z(angle):
    R = np.mat([[ cos(angle), -sin(angle), 0 ],
                 [ sin(angle),  cos(angle), 0 ],
                 [         0,          0, 1 ]])
    return R

# Transform the obstacle with its boundary point in the global frame
def obstacle_to_world(location, dimensions, orientation):
    box_pts = []

    x = location.x
    y = location.y
    z = location.z

    yaw = orientation.yaw * pi / 180

    xrad = dimensions.x
    yrad = dimensions.y
    zrad = dimensions.z

    # Border points in the obstacle frame
    cpos = np.array([
            [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0    ],
            [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
    
    # Rotation of the obstacle
    rotyaw = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]])
    
    # Location of the obstacle in the world frame
    cpos_shift = np.array([
            [x, x, x, x, x, x, x, x],
            [y, y, y, y, y, y, y, y]])
    
    cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)

    for j in range(cpos.shape[1]):
        box_pts.append([cpos[0,j], cpos[1,j]])
    
    return box_pts

def make_carla_settings(args, num_pedestrians, num_vehicles, seed_pedestrians, seed_vehicles):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()
    
    # There is no need for non-agent info requests if there are no pedestrians
    # or vehicles.
    get_non_player_agents_info = False
    if (num_pedestrians > 0 or num_vehicles > 0):
        get_non_player_agents_info = True

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info, 
        NumberOfVehicles=num_vehicles,
        NumberOfPedestrians=num_pedestrians,
        SeedVehicles=seed_vehicles,
        SeedPedestrians=seed_pedestrians,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)

    # Common cameras settings
    cam_height = camera_parameters['z'] 
    cam_x_pos = camera_parameters['x']
    cam_y_pos = camera_parameters['y']
    camera_width = camera_parameters['width']
    camera_height = camera_parameters['height']
    camera_fov = camera_parameters['fov']

    # Declare here your sensors

    #  RGB camera
    if USE_CAMERA:
        camera0 = Camera("CameraRGB")
        camera0.set_image_size(camera_width, camera_height)
        camera0.set(FOV=camera_fov)
        camera0.set_position(cam_x_pos, cam_y_pos, cam_height)
        settings.add_sensor(camera0)

    return settings

class Timer(object):
    """ Timer Class
    
    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements
    
    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    z   =  measurement.player_measurements.transform.location.z

    pitch = math.radians(measurement.player_measurements.transform.rotation.pitch)
    roll = math.radians(measurement.player_measurements.transform.rotation.roll)
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, z, pitch, roll, yaw)

def get_start_pos(scene):
    """Obtains player start x,y, yaw pose from the scene
    
    Obtains the player x,y, and yaw pose from the scene.

    Args:
        scene: The CARLA scene object

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

    return (x, y, yaw)

def get_player_collided_flag(measurement, 
                             prev_collision_vehicles, 
                             prev_collision_pedestrians,
                             prev_collision_other):
    """Obtains collision flag from player. Check if any of the three collision
    metrics (vehicles, pedestrians, others) from the player are true, if so the
    player has collided to something.

    Note: From the CARLA documentation:

    "Collisions are not annotated if the vehicle is not moving (<1km/h) to avoid
    annotating undesired collision due to mistakes in the AI of non-player
    agents."
    """
    player_meas = measurement.player_measurements
    current_collision_vehicles = player_meas.collision_vehicles
    current_collision_pedestrians = player_meas.collision_pedestrians
    current_collision_other = player_meas.collision_other

    collided_vehicles = current_collision_vehicles > prev_collision_vehicles
    collided_pedestrians = current_collision_pedestrians > \
                           prev_collision_pedestrians
    collided_other = current_collision_other > prev_collision_other

    return (collided_vehicles or collided_pedestrians or collided_other,
            current_collision_vehicles,
            current_collision_pedestrians,
            current_collision_other)

def send_control_command(client, throttle, steer, brake, 
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.
    
    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list, collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file: 
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i],
                                   collided_list[i]))

def write_collisioncount_file(collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_count.txt')

    with open(file_name, 'w') as collision_file: 
        collision_file.write(str(sum(collided_list)))

def make_correction(waypoint,previuos_waypoint,desired_speed):
    dx = waypoint[0] - previuos_waypoint[0]
    dy = waypoint[1] - previuos_waypoint[1]

    if dx < 0:
        moveY = -1.5
    elif dx > 0:
        moveY = 1.5
    else:
        moveY = 0

    if dy < 0:
        moveX = 1.5
    elif dy > 0:
        moveX = -1.5
    else:
        moveX = 0
    
    waypoint_on_lane = waypoint
    waypoint_on_lane[0] += moveX
    waypoint_on_lane[1] += moveY
    waypoint_on_lane[2] = desired_speed

    return waypoint_on_lane

def exec_waypoint_nav_demo(args, state_info, start_wp, stop_wp, num_pedestrians, num_vehicles, seed_pedestrians, seed_vehicles):
    """ Executes waypoint navigation demo.
    """

    try:

        with make_carla_client(args.host, args.port) as client:
            print('Carla client connected.')

            settings = make_carla_settings(args, num_pedestrians, num_vehicles, seed_pedestrians, seed_vehicles)

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Refer to the player start folder in the WorldOutliner to see the 
            # player start information
            player_start = start_wp

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode at %r...' % scene.map_name)
            client.start_episode(player_start)

            #############################################
            # Load Configurations
            #############################################

            # Load configuration file (options.cfg) and then parses for the various
            # options. Here we have two main options:
            # live_plotting and live_plotting_period, which controls whether
            # live plotting is enabled or how often the live plotter updates
            # during the simulation run.
            config = configparser.ConfigParser()
            config.read(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))         
            demo_opt = config['Demo Parameters']

            # Get options
            enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
            enable_live_plot = enable_live_plot == 'True'
            live_plot_period = float(demo_opt.get('live_plotting_period', 0))

            # Set options
            live_plot_timer = Timer(live_plot_period)
            
            # Settings Mission Planner
            mission_planner = CityTrack("Town01")

            #############################################
            # Determine simulation average timestep (and total frames)
            #############################################
            # Ensure at least one frame is used to compute average timestep
            num_iterations = ITER_FOR_SIM_TIMESTEP
            if (ITER_FOR_SIM_TIMESTEP < 1):
                num_iterations = 1

            # Gather current data from the CARLA server. This is used to get the
            # simulator starting game time. Note that we also need to
            # send a command back to the CARLA server because synchronous mode
            # is enabled.
            measurement_data, sensor_data = client.read_data()
            sim_start_stamp = measurement_data.game_timestamp / 1000.0
            # Send a control command to proceed to next iteration.
            # This mainly applies for simulations that are in synchronous mode.
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            # Computes the average timestep based on several initial iterations
            sim_duration = 0
            for i in range(num_iterations):
                # Gather current data
                measurement_data, sensor_data = client.read_data()
                # Send a control command to proceed to next iteration
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                # Last stamp
                if i == num_iterations - 1:
                    sim_duration = measurement_data.game_timestamp / 1000.0 -\
                                sim_start_stamp

            # Outputs average simulation timestep and computes how many frames
            # will elapse before the simulation should end based on various
            # parameters that we set in the beginning.
            SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
            print("SERVER SIMULATION STEP APPROXIMATION: " + \
                str(SIMULATION_TIME_STEP))
            TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
                                SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

            #############################################
            # Frame-by-Frame Iteration and Initialization
            #############################################
            # Store pose history starting from the start position
            measurement_data, sensor_data = client.read_data()
            start_timestamp = measurement_data.game_timestamp / 1000.0
            start_x, start_y, start_z, start_pitch, start_roll, start_yaw = get_current_pose(measurement_data)
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            x_history     = [start_x]
            y_history     = [start_y]
            yaw_history   = [start_yaw]
            time_history  = [0]
            speed_history = [0]
            collided_flag_history = [False]  # assume player starts off non-collided

            #############################################
            # Settings Waypoints
            #############################################
            starting    = scene.player_start_spots[start_wp]
            destination = scene.player_start_spots[stop_wp]

            # Starting position is the current position
            # (x, y, z, pitch, roll, yaw)
            source_pos = [starting.location.x, starting.location.y, starting.location.z]
            source_ori = [starting.orientation.x, starting.orientation.y]
            source = mission_planner.project_node(source_pos)

            # Destination position
            destination_pos = [destination.location.x, destination.location.y, destination.location.z]
            destination_ori = [destination.orientation.x, destination.orientation.y]
            destination = mission_planner.project_node(destination_pos)

            waypoints = []
            waypoints_route = mission_planner.compute_route(source, source_ori, destination, destination_ori)
            desired_speed = 5.0
            turn_speed    = 2.5

            intersection_nodes = mission_planner.get_intersection_nodes()
            intersection_pair = []
            turn_cooldown = 0
            prev_x = False
            prev_y = False
            # Put waypoints in the lane
            previuos_waypoint = mission_planner._map.convert_to_world(waypoints_route[0])
            for i in range(1,len(waypoints_route)):
                point = waypoints_route[i]

                waypoint = mission_planner._map.convert_to_world(point)

                current_waypoint = make_correction(waypoint,previuos_waypoint,desired_speed)
                
                dx = current_waypoint[0] - previuos_waypoint[0]
                dy = current_waypoint[1] - previuos_waypoint[1]

                is_turn = ((prev_x and abs(dy) > 0.1) or (prev_y and abs(dx) > 0.1)) and not(abs(dx) > 0.1 and abs(dy) > 0.1)

                prev_x = abs(dx) > 0.1
                prev_y = abs(dy) > 0.1

                if point in intersection_nodes:                
                    prev_start_intersection = mission_planner._map.convert_to_world(waypoints_route[i-2])
                    center_intersection = mission_planner._map.convert_to_world(waypoints_route[i])

                    start_intersection = mission_planner._map.convert_to_world(waypoints_route[i-1])
                    end_intersection = mission_planner._map.convert_to_world(waypoints_route[i+1])

                    start_intersection = make_correction(start_intersection,prev_start_intersection,turn_speed)
                    end_intersection = make_correction(end_intersection,center_intersection,turn_speed)
                    
                    dx = start_intersection[0] - end_intersection[0]
                    dy = start_intersection[1] - end_intersection[1]

                    if abs(dx) > 0 and abs(dy) > 0:
                        intersection_pair.append((center_intersection,len(waypoints)))
                        waypoints[-1][2] = turn_speed
                        
                        middle_point = [(start_intersection[0] + end_intersection[0]) /2,  (start_intersection[1] + end_intersection[1]) /2]

                        centering = 0.75

                        middle_intersection = [(centering*middle_point[0] + (1-centering)*center_intersection[0]),  (centering*middle_point[1] + (1-centering)*center_intersection[1])]

                        # Point at intersection:
                        A = [[start_intersection[0], start_intersection[1], 1],
                            [end_intersection[0], end_intersection[1], 1],
                            [middle_intersection[0], middle_intersection[1], 1]]
                            
                        b = [-start_intersection[0]**2 - start_intersection[1]**2, 
                            -end_intersection[0]**2 - end_intersection[1]**2,
                            -middle_intersection[0]**2 - middle_intersection[1]**2]

                        coeffs = np.matmul(np.linalg.inv(A), b)

                        x = start_intersection[0]
                        
                        center_x = -coeffs[0]/2
                        center_y = -coeffs[1]/2

                        r = sqrt(center_x**2 + center_y**2 - coeffs[2])

                        theta_start = math.atan2((start_intersection[1] - center_y),(start_intersection[0] - center_x))
                        theta_end = math.atan2((end_intersection[1] - center_y),(end_intersection[0] - center_x))

                        theta = theta_start

                        start_to_end = 1 if theta_start < theta_end else -1

                        while (start_to_end==1 and theta < theta_end) or (start_to_end==-1 and theta > theta_end):
                            waypoint_on_lane = [0,0,0]

                            waypoint_on_lane[0] = center_x + r * cos(theta)
                            waypoint_on_lane[1] = center_y + r * sin(theta)
                            waypoint_on_lane[2] = turn_speed

                            waypoints.append(waypoint_on_lane)
                            theta += (abs(theta_end - theta_start) * start_to_end) / 10
                        
                        turn_cooldown = 4
                else:
                    waypoint = mission_planner._map.convert_to_world(point)

                    if turn_cooldown > 0:
                        target_speed = turn_speed
                        turn_cooldown -= 1
                    else:
                        target_speed = desired_speed
                    
                    waypoint_on_lane = make_correction(waypoint,previuos_waypoint,target_speed)

                    waypoints.append(waypoint_on_lane)

                    previuos_waypoint = waypoint

            waypoints = np.array(waypoints)

            #######################################################################
            # -------------- Get initial traffic lights state ---------------------
            #######################################################################

            # Read traffic lights info
            traffic_light_info=dict(position=[], fences=[], states=[])
            for agent in measurement_data.non_player_agents:
                if agent.HasField('traffic_light'):
                    # Get position
                    traffic_light_info['position'].append((agent.traffic_light.transform.location.x, 
                                                        agent.traffic_light.transform.location.y))
                    # Compute stop line
                    bb = obstacle_to_world(agent.traffic_light.transform.location, 
                                        DefaultMunch.fromDict(dict(x=TRAFFIC_LIGHT_STOP_LINE_LEN,y=0,z=0)), 
                                        agent.traffic_light.transform.rotation)
                    traffic_light_info['fences'].append([*bb[0], *bb[4]])  # 0,1,2 is the right point, 
                                                                        # 4,5,6 is the left point (are
                                                                        # the same because the bb is 
                                                                        # compressed on the y)
                    # Get states
                    traffic_light_info['states'].append(agent.traffic_light.state)
            # traffic_light_info['visited'] = [False] * len(traffic_light_info['fences'])
                    
                
            ###########################################################
            # --------------- Read vehicle info ----------------
            ###########################################################
            vehicle_info=dict(position=[], fences=[], speeds=[])
            for agent in measurement_data.non_player_agents:
                if agent.HasField('vehicle'):
                    # Get position
                    vehicle_info['position'].append((agent.vehicle.transform.location.x, 
                                                        agent.vehicle.transform.location.y, 
                                                        agent.vehicle.transform.location.z))
                    # Compute stop line
                    bb = obstacle_to_world(agent.vehicle.transform.location, 
                                                agent.vehicle.bounding_box.extent, 
                                                agent.vehicle.transform.rotation)
                    vehicle_info['fences'].append(bb)
                    
                    # Get forward speed
                    vehicle_info['speeds'].append(agent.vehicle.forward_speed) 
                
            #############################################
            # Controller 2D Class Declaration
            #############################################
            # This is where we take the controller2d.py class
            # and apply it to the simulator
            controller = controller2d.Controller2D(waypoints)

            #############################################
            # Vehicle Trajectory Live Plotting Setup
            #############################################
            # Uses the live plotter to generate live feedback during the simulation
            # The two feedback includes the trajectory feedback and
            # the controller feedback (which includes the speed tracking).
            lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
            lp_1d = lv.LivePlotter(tk_title="Controls Feedback")

            ###
            # Add 2D position / trajectory plot
            ###
            trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                    title='Vehicle Trajectory',
                    figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                    edgecolor="black",
                    rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

            trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed 
                                            # coordinate system the X
                                            # axis in the graph is flipped
            trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

            # Add waypoint markers
            trajectory_fig.add_graph("waypoints", window_size=len(waypoints),
                                    x0=waypoints[:,0], y0=waypoints[:,1],
                                    linestyle="-", marker="", color='g')
            # Add trajectory markers
            trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                    x0=[start_x]*TOTAL_EPISODE_FRAMES, 
                                    y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                    color=[1, 0.5, 0])
            # Add starting position marker
            trajectory_fig.add_graph("start_pos", window_size=1, 
                                    x0=[start_x], y0=[start_y],
                                    marker=11, color=[1, 0.5, 0], 
                                    markertext="Start", marker_text_offset=1)

            trajectory_fig.add_graph("obstacles_points",
                                    window_size=8 * (num_pedestrians + num_vehicles) ,
                                    x0=[0]* (8 * (num_pedestrians + num_vehicles)), 
                                    y0=[0]* (8 * (num_pedestrians + num_vehicles)),
                                        linestyle="", marker="+", color='b')

            # Add end position marker
            trajectory_fig.add_graph("end_pos", window_size=1, 
                                    x0=[waypoints[-1, 0]], 
                                    y0=[waypoints[-1, 1]],
                                    marker="D", color='r', 
                                    markertext="End", marker_text_offset=1)
            # Add car marker
            trajectory_fig.add_graph("car", window_size=1, 
                                    marker="s", color='b', markertext="Car",
                                    marker_text_offset=1)
            # Add lead car information
            trajectory_fig.add_graph("leadcar", window_size=1, 
                                    marker="s", color='g', markertext="Lead Car",
                                    marker_text_offset=1)

            # Add lookahead path
            trajectory_fig.add_graph("selected_path", 
                                    window_size=INTERP_MAX_POINTS_PLOT,
                                    x0=[start_x]*INTERP_MAX_POINTS_PLOT, 
                                    y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                    color=[1, 0.5, 0.0],
                                    linewidth=3)

            # Add local path proposals
            for i in range(NUM_PATHS):
                trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                        x0=None, y0=None, color=[0.0, 0.0, 1.0])

            ###
            # Add 1D speed profile updater
            ###
            forward_speed_fig =\
                    lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
            forward_speed_fig.add_graph("forward_speed", 
                                        label="forward_speed", 
                                        window_size=TOTAL_EPISODE_FRAMES)
            forward_speed_fig.add_graph("reference_signal", 
                                        label="reference_Signal", 
                                        window_size=TOTAL_EPISODE_FRAMES)

            # Add throttle signals graph
            throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
            throttle_fig.add_graph("throttle", 
                                label="throttle", 
                                window_size=TOTAL_EPISODE_FRAMES)
            # Add brake signals graph
            brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
            brake_fig.add_graph("brake", 
                                label="brake", 
                                window_size=TOTAL_EPISODE_FRAMES)
            # Add steering signals graph
            steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
            steer_fig.add_graph("steer", 
                                label="steer", 
                                window_size=TOTAL_EPISODE_FRAMES)

            # live plotter is disabled, hide windows
            if not enable_live_plot:
                lp_traj._root.withdraw()
                lp_1d._root.withdraw()      

            #############################################
            # Local Planner Variables
            #############################################
            wp_goal_index   = 0
            local_waypoints = None
            path_validity   = np.zeros((NUM_PATHS, 1), dtype=bool)
            lp = local_planner.LocalPlanner(NUM_PATHS,
                                            PATH_OFFSET,
                                            CIRCLE_OFFSETS,
                                            CIRCLE_RADII,
                                            PATH_SELECT_WEIGHT,
                                            TIME_GAP,
                                            A_MAX,
                                            SLOW_SPEED,
                                            STOP_LINE_BUFFER)
            bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE,
                                                        LEAD_VEHICLE_LOOKAHEAD, 
                                                        traffic_light_info,
                                                        vehicle_info)

            #############################################
            # Scenario Execution Loop
            #############################################

            # Iterate the frames until the end of the waypoints is reached or
            # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
            # ouptuts the results to the controller output directory.
            reached_the_end = False
            skip_first_frame = True

            # Initialize the current timestamp.
            current_timestamp = start_timestamp

            # Initialize collision history
            prev_collision_vehicles    = 0
            prev_collision_pedestrians = 0
            prev_collision_other       = 0

            for frame in range(TOTAL_EPISODE_FRAMES):
                # Gather current data from the CARLA server
                measurement_data, sensor_data = client.read_data()

                ###########################################################
                # ------------------- Visualize Camera --------------------
                ###########################################################
                if USE_CAMERA:
                    camera_data = sensor_data.get('CameraRGB', None)
                    if camera_data is not None:
                        camera_data = to_bgra_array(camera_data)
                        cv2.imshow("CameraRGB", camera_data)

                ###########################################################
                # ------------------- State info --------------------------
                ###########################################################
                if VISUALIZE_STATE_INFO:
                    state_info.set_state_info(bp.get_state_info())
                    state_info.draw_things(*bp.get_to_draw())

                ###########################################################
                # --------------- Read traffic lights info ----------------
                ###########################################################
                traffic_light_info=dict(position=[], fences=[], states=[])
                for agent in measurement_data.non_player_agents:
                    if agent.HasField('traffic_light'):
                        # if not(agent.traffic_light.transform.location.x > 70 and agent.traffic_light.transform.location.x < 80) or \
                        #     not(agent.traffic_light.transform.location.y > -10 and agent.traffic_light.transform.location.y < 10):
                        #     continue
                        # Get position
                        traffic_light_info['position'].append((agent.traffic_light.transform.location.x, 
                                                            agent.traffic_light.transform.location.y))
                        # Get stop line
                        bb = obstacle_to_world(agent.traffic_light.transform.location,
                                            DefaultMunch.fromDict(dict(x=TRAFFIC_LIGHT_STOP_LINE_LEN,y=0,z=0)), 
                                            agent.traffic_light.transform.rotation)
                        traffic_light_info['fences'].append([*bb[0], *bb[4]])  #### UPDATE THIS
                        # Get states
                        traffic_light_info['states'].append(agent.traffic_light.state)
                bp.set_traffic_light(traffic_light_info)

                ###########################################################
                # --------------- Read vehicle info ----------------
                ###########################################################
                vehicle_info=dict(position=[], fences=[], speeds=[])
                for agent in measurement_data.non_player_agents:
                    if agent.HasField('vehicle'):
                        # Get position
                        vehicle_info['position'].append((agent.vehicle.transform.location.x, 
                                                            agent.vehicle.transform.location.y, 
                                                            agent.vehicle.transform.location.z))
                        # Compute stop line
                        bb = obstacle_to_world(agent.vehicle.transform.location, 
                                                    agent.vehicle.bounding_box.extent, 
                                                    agent.vehicle.transform.rotation)
                        vehicle_info['fences'].append(bb)
                        
                        # Get forward speed
                        vehicle_info['speeds'].append(agent.vehicle.forward_speed)
                bp.set_vehicle(vehicle_info) 

                # UPDATE HERE the obstacles list
                obstacles = []

                # Update pose and timestamp
                prev_timestamp = current_timestamp
                current_x, current_y, current_z, current_pitch, current_roll, current_yaw = \
                    get_current_pose(measurement_data)
                current_speed = measurement_data.player_measurements.forward_speed
                current_timestamp = float(measurement_data.game_timestamp) / 1000.0

                # Wait for some initial time before starting the demo
                if current_timestamp <= WAIT_TIME_BEFORE_START:
                    send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                    continue
                else:
                    current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START
                
                # Store history
                x_history.append(current_x)
                y_history.append(current_y)
                yaw_history.append(current_yaw)
                speed_history.append(current_speed)
                time_history.append(current_timestamp) 

                # Store collision history
                collided_flag,\
                prev_collision_vehicles,\
                prev_collision_pedestrians,\
                prev_collision_other = get_player_collided_flag(measurement_data,
                                                    prev_collision_vehicles,
                                                    prev_collision_pedestrians,
                                                    prev_collision_other)
                collided_flag_history.append(collided_flag)

                # Execute the behaviour and local planning in the current instance
                # Note that updating the local path during every controller update
                # produces issues with the tracking performance (imagine everytime
                # the controller tried to follow the path, a new path appears). For
                # this reason, the local planner (LP) will update every X frame,
                # stored in the variable LP_FREQUENCY_DIVISOR, as it is analogous
                # to be operating at a frequency that is a division to the 
                # simulation frequency.
                if frame % LP_FREQUENCY_DIVISOR == 0:
                    # Compute open loop speed estimate.
                    open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

                    # Calculate the goal state set in the local frame for the local planner.
                    # Current speed should be open loop for the velocity profile generation.
                    ego_state = [current_x, current_y, current_yaw, open_loop_speed]

                    # Set lookahead based on current speed.
                    bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

                    # Perform a state transition in the behavioural planner.
                    bp.transition_state(waypoints, ego_state, current_speed)

                    # Compute the goal state set from the behavioural planner's computed goal state.
                    goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)

                    # Calculate planned paths in the local frame.
                    paths, path_validity = lp.plan_paths(goal_state_set)

                    # Transform those paths back to the global frame.
                    paths = local_planner.transform_paths(paths, ego_state)

                    # Perform collision checking.
                    collision_check_array = lp._collision_checker.collision_check(paths, [])

                    # Compute the best local path.
                    best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
                    # If no path was feasible, continue to follow the previous best path.
                    if best_index == None:
                        best_path = lp._prev_best_path
                    else:
                        best_path = paths[best_index]
                        lp._prev_best_path = best_path

                    if best_path is not None:
                        # Compute the velocity profile for the path, and compute the waypoints.
                        desired_speed = bp._goal_state[2]
                        decelerate_to_stop = bp._state == behavioural_planner.DECELERATE_TO_STOP
                        local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, bp._lead_car_state, bp._follow_lead_vehicle)

                        if local_waypoints != None:
                            # Update the controller waypoint path with the best local path.
                            # This controller is similar to that developed in Course 1 of this
                            # specialization.  Linear interpolation computation on the waypoints
                            # is also used to ensure a fine resolution between points.
                            wp_distance = []   # distance array
                            local_waypoints_np = np.array(local_waypoints)
                            for i in range(1, local_waypoints_np.shape[0]):
                                wp_distance.append(
                                        np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
                                                (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
                            wp_distance.append(0)  # last distance is 0 because it is the distance
                                                # from the last waypoint to the last waypoint

                            # Linearly interpolate between waypoints and store in a list
                            wp_interp      = []    # interpolated values 
                                                # (rows = waypoints, columns = [x, y, v])
                            for i in range(local_waypoints_np.shape[0] - 1):
                                # Add original waypoint to interpolated waypoints list (and append
                                # it to the hash table)
                                wp_interp.append(list(local_waypoints_np[i]))
                        
                                # Interpolate to the next waypoint. First compute the number of
                                # points to interpolate based on the desired resolution and
                                # incrementally add interpolated points until the next waypoint
                                # is about to be reached.
                                num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                                            float(INTERP_DISTANCE_RES)) - 1)
                                wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
                                wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                                for j in range(num_pts_to_interp):
                                    next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                                    wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                            # add last waypoint at the end
                            wp_interp.append(list(local_waypoints_np[-1]))
                            
                            # Update the other controller values and controls
                            controller.update_waypoints(wp_interp)

                ###
                # Controller Update
                ###
                if local_waypoints != None and local_waypoints != []:
                    controller.update_values(current_x, current_y, current_yaw, 
                                            current_speed,
                                            current_timestamp, frame)
                    controller.update_controls()
                    cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
                else:
                    cmd_throttle = 0.0
                    cmd_steer = 0.0
                    cmd_brake = 0.0

                # Skip the first frame or if there exists no local paths
                if skip_first_frame and frame == 0:
                    pass
                elif local_waypoints == None:
                    pass
                else:
                    # Update live plotter with new feedback
                    trajectory_fig.roll("trajectory", current_x, current_y) 
                    trajectory_fig.roll("car", current_x, current_y) 
                    
                    # Load parked car points
                    if len(obstacles) > 0:
                        x = obstacles[:,:,0]
                        y = obstacles[:,:,1]
                        x = np.reshape(x, x.shape[0] * x.shape[1])
                        y = np.reshape(y, y.shape[0] * y.shape[1])

                        trajectory_fig.roll("obstacles_points", x, y)

                    
                    forward_speed_fig.roll("forward_speed", 
                                        current_timestamp, 
                                        current_speed)
                    forward_speed_fig.roll("reference_signal", 
                                        current_timestamp, 
                                        controller._desired_speed)
                    throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
                    brake_fig.roll("brake", current_timestamp, cmd_brake)
                    steer_fig.roll("steer", current_timestamp, cmd_steer)

                    # Local path plotter update
                    if frame % LP_FREQUENCY_DIVISOR == 0:
                        path_counter = 0
                        for i in range(NUM_PATHS):
                            # If a path was invalid in the set, there is no path to plot.
                            if path_validity[i]:
                                # Colour paths according to collision checking.
                                if not collision_check_array[path_counter]:
                                    colour = 'r'
                                elif i == best_index:
                                    colour = 'k'
                                else:
                                    colour = 'b'
                                trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1], colour)
                                path_counter += 1
                            else:
                                trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')
                    # When plotting lookahead path, only plot a number of points
                    # (INTERP_MAX_POINTS_PLOT amount of points). This is meant
                    # to decrease load when live plotting
                    wp_interp_np = np.array(wp_interp)
                    path_indices = np.floor(np.linspace(0, 
                                                        wp_interp_np.shape[0]-1,
                                                        INTERP_MAX_POINTS_PLOT))
                    trajectory_fig.update("selected_path", 
                            wp_interp_np[path_indices.astype(int), 0],
                            wp_interp_np[path_indices.astype(int), 1],
                            new_colour=[1, 0.5, 0.0])


                    # Refresh the live plot based on the refresh rate 
                    # set by the options
                    if enable_live_plot and \
                    live_plot_timer.has_exceeded_lap_period():
                        lp_traj.refresh()
                        lp_1d.refresh()
                        live_plot_timer.lap()

                # Output controller command to CARLA server
                send_control_command(client,
                                    throttle=cmd_throttle,
                                    steer=cmd_steer,
                                    brake=cmd_brake)

                # Find if reached the end of waypoint. If the car is within
                # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
                # the simulation will end.
                dist_to_last_waypoint = np.linalg.norm(np.array([
                    waypoints[-1][0] - current_x,
                    waypoints[-1][1] - current_y]))
                if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                    reached_the_end = True
                if reached_the_end:
                    break

            # End of demo - Stop vehicle and Store outputs to the controller output
            # directory.
            if reached_the_end:
                print("Reached the end of path. Writing to controller_output...")
            else:
                print("Exceeded assessment time. Writing to controller_output...")
            # Stop the car
            send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
            # Store the various outputs
            store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
            store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
            store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
            store_trajectory_plot(brake_fig.fig, 'brake_output.png')
            store_trajectory_plot(steer_fig.fig, 'steer_output.png')
            write_trajectory_file(x_history, y_history, speed_history, time_history,
                                collided_flag_history)
            write_collisioncount_file(collided_flag_history)

            lp_traj._root.destroy()
            lp_1d._root.destroy()

    except KeyboardInterrupt as e:
        lp_traj._root.destroy()
        lp_1d._root.destroy()
        raise e

    
def main():
    """Main function.

    Args:
        -v, --verbose: print debug information
        --host: IP of the host server (default: localhost)
        -p, --port: TCP port to listen to (default: 2000)
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
    """
    argparser = argparse.ArgumentParser(description='Midterm project main.')  #description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '-ss', '--start-and-stop',
        default=f'{PLAYER_START_INDEX},{DESTINATION_INDEX}',
        help='Start and stop WP couple. Start and stop must be separated by a comma and multiple couples must be ' + \
             f'separated by a dot like: "57,55.88,90". (default: {PLAYER_START_INDEX},{DESTINATION_INDEX})')
    argparser.add_argument(
        '-np', '--num-pedestrians',
        default=f'{NUM_PEDESTRIANS}',
        help='Number of pedestrians to generate. Multiple chooses must be separated by a comma like "1,3".' + \
             f'(default: {NUM_PEDESTRIANS})')
    argparser.add_argument(
        '-nv', '--num-vehicles',
        default=f'{NUM_VEHICLES}',
        help='Number of vehicles to generate. Multiple chooses must be separated by a comma like "1,3".' + \
             f'(default: {NUM_VEHICLES})')
    argparser.add_argument(
        '-sp', '--seed-pedestrians',
        default=f'{SEED_PEDESTRIANS}',
        help='Pedestrians seed. Multiple chooses must be separated by a comma like "1,3".' + \
             f'(default: {SEED_PEDESTRIANS})')
    argparser.add_argument(
        '-sv', '--seed-vehicles',
        default=f'{SEED_VEHICLES}',
        help='Vehicles seed. Multiple chooses must be separated by a comma like "1,3".' + \
             f'(default: {SEED_VEHICLES})')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_dir = f'./log/main_{datetime.now():%Y-%m-%d-%H-%M-%S}.log'
    # Create log file if not exists
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    with open(log_dir, "w") as f:
        f.write("")
    # set log config
    logging.basicConfig(format='%(asctime)s, %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', filename=log_dir, level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    endpoints_wp = [[int(cp[0]), int(cp[1])] for cp in [couple.split(',') for couple in args.start_and_stop.split('.')]]
    nums_pedestrians = [int(x) for x in args.num_pedestrians.split(',')]
    nums_vehicles = [int(x) for x in args.num_vehicles.split(',')]
    seeds_pedestrians = [int(x) for x in args.seed_pedestrians.split(',')]
    seeds_vehicles = [int(x) for x in args.seed_vehicles.split(',')]

    total_simulations = len(endpoints_wp) * len(nums_pedestrians) * len(nums_vehicles) * len(seeds_pedestrians) * len(seeds_vehicles)

    print(f'Running {total_simulations} simulations. Parameters:' + \
          f'\nendpoints_wp: {endpoints_wp}' + \
          f'\nnums_pedestrians: {nums_pedestrians}' + \
          f'\nnums_vehicles: {nums_vehicles}' + \
          f'\nseeds_pedestrians: {seeds_pedestrians}' + \
          f'\nseeds_vehicles: {seeds_vehicles}')

    pbar = tqdm(desc='Executing simulation', total=total_simulations)  # Init progress bar

    for start_wp, stop_wp in endpoints_wp:
        for num_pedestrians in nums_pedestrians:
            for num_vehicles in nums_vehicles:
                for seed_pedestrians in seeds_pedestrians:
                    for seed_vehicles in seeds_vehicles:

                        logging.info(f"Executing simulation for {list(zip(('start_wp', 'stop_wp', 'num_pedestrians', 'num_vehicles', 'seed_pedestrians', 'seed_vehicles'),(start_wp, stop_wp, num_pedestrians, num_vehicles, seed_pedestrians, seed_vehicles)))}")

                        state_info = StateInfo(VISUALIZE_STATE_INFO)
                        state_info.set_param_info(f'path: ({start_wp}, {stop_wp}), {num_pedestrians} pedestrians (seed: {seed_pedestrians}), ' + \
                                                  f'{num_vehicles} vehicles (seed: {seed_vehicles})')

                        try:
                            exec_waypoint_nav_demo(args, state_info, start_wp, stop_wp, num_pedestrians, num_vehicles, seed_pedestrians, seed_vehicles)
                            pbar.update()  # Update progress bar
                        except TCPConnectionError as error:
                            logging.error(error)
                            logging.error(f"Simulation with param {list(zip(('start_wp', 'stop_wp', 'num_pedestrians', 'num_vehicles', 'seed_pedestrians', 'seed_vehicles'),(start_wp, stop_wp, num_pedestrians, num_vehicles, seed_pedestrians, seed_vehicles)))} not executed")
                            pbar.update()  # Update progress bar
                        except KeyboardInterrupt:
                            _print('\nSkipping this scene. To exit, click CTRL+C another time.')
                            time.sleep(2)
                            pbar.update()  # Update progress bar
                        finally:
                            state_info.quit()
                            
    
    pbar.close()   # Finalize Progress bar


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('Cancelled by user. Bye!')
