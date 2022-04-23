#!/usr/bin/env python3
import numpy as np
from shapely.geometry import Point, LineString, Polygon, CAP_STYLE
from shapely.affinity import rotate
from shapely.ops import unary_union
import matplotlib.pyplot as plt


# State machine states
FOLLOW_LANE = 'Follow Lane'
DECELERATE_TO_STOP = 'Decelerate to Stop'
STAY_STOPPED = 'Stay Stopped'

# Stop speed threshold
STOP_THRESHOLD = 0.1

# Enumerate Trafficlight State
TRAFFICLIGHT_GREEN = 0
TRAFFICLIGHT_YELLOW = 1
TRAFFICLIGHT_RED = 2
TRAFFICLIGHT_YELLOW_MIN_TIME = 3.5  # sec 

# Define x dimension of the bounding box to check for obstacles.
BB_PATH = 1.5  # m
BB_PEDESTRIAN_LEFT = 4 # m
BB_PEDESTRIAN_RIGHT = 1.5 # m


class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead, traffic_lights, vehicle, pedestrians):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._lead_car_state                = None
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count                    = 0
        self._lookahead_collision_index     = 0
        self._traffic_lights                = traffic_lights
        self._vehicle                       = vehicle
        self._pedestrians                   = pedestrians
        self._state_info                    = ''
        self._current_input                 = ''
        self._init_plot()

    def _init_plot(self):
        """Initialize the environment to plot information in a live figure.

        Note:
            This method must be colled just once, before any call at _draw method. Otherwise Exception
            will be rised in ythe runtime.
        """
        self._fig = plt.figure('Geometry Help')
        self._ax = self._fig.add_subplot(111) 
        self._fig.canvas.draw()
        self._renderer = self._fig.canvas.renderer

    def _draw(self, geometry, angle=0, short='-', settings={}):
        """Draw a geometry object in the figure spawned by the behavioural planner.

        Note:
            The geometry will be added in the plot after calling the method '_finalize_draw'. Also,
            wher that was called, all the other geometry need to be re-added whit this function for the
            next plot, if needed.
        
        Args:
            geometry (Point or LineString or Polygon): The geometry to print in the figure.
            angle (float, default=0): the angle, in radians, which represents the direction of 
            the vehicle, to obtain a representation of the geometries always in the viewing direction.
            short (str, default='-'): the first matplotlib argument to define a style.
            settings (dict, default={}): The matplotlib settings to associate at that geometry. 
        """
        geometry = rotate(geometry, (-angle + np.pi/2), (0,0), use_radians=True)
        if type(geometry) == Point:
            self._ax.plot(*geometry.coords.xy, short, **settings)
        elif type(geometry) == LineString:
            self._ax.plot(*geometry.coords.xy, short, **settings)
        elif type(geometry) == Polygon:
            self._ax.plot(*geometry.exterior.xy, short, **settings)

    def _finalize_draw(self):
        """This method shows everything passed to the _draw method in the figure.
        
        Note:
            When this method is called, the previous geometries in the image will continue to be displayed 
            but, if not re-inserted with the _draw method, they will no longer be displayed the next time 
            this method is used.
        """
        self._fig.draw(self._renderer)
        #self._fig.legend()
        plt.pause(.001)
        self._ax.cla()
        self._ax.set_aspect('equal', 'datalim')
        self._ax.invert_xaxis()
    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def set_traffic_light(self, traffic_lights):
        """Update the traffic light information. These are coming from Carla."""
        for k in traffic_lights:
            self._traffic_lights[k] = traffic_lights[k]
    
    def set_vehicle(self, vehicle):
        """Update the other vehicle information. These are coming from Carla."""
        for k in vehicle:
            self._vehicle[k] = vehicle[k]

    def set_pedestrians(self, pedestrians):
        """Update the pedestrian information. These are coming from Carla."""
        for k in pedestrians:
            self._pedestrians[k] = pedestrians[k]

    def get_bp_info(self):
        """Return some string with information on the state of the FSM and useful
        current input."""
        return self._state_info, self._current_input

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations) 
                              before moving from stop sign.
        """
        # Update state info
        self._state_info = f'Current State: {self._state}'
        # Update input info
        self._current_input = 'Current inputs:'

        # ---------------- GOAL INDEX ---------------------------
        # Transform ego info in shapely geometry
        ego_point = Point(ego_state[0], ego_state[1])
        ego_direction = ego_state[2]
        # Draw the ego state point.
        self._draw(ego_point, angle=ego_direction, short='g.', settings=dict(markersize=20, label='EgoPoint'))
        
        # Get closest waypoint
        closest_len, closest_index = get_closest_index(waypoints, ego_state)

        # Get goal based on the current lookahead
        goal_index, is_close_itc, goal_path = self.get_goal_index(waypoints, ego_point, ego_direction, closest_len, closest_index)
        self._draw(goal_path, angle=ego_direction, short='y^-', settings=dict(markersize=10, label='Goal Path'))

        # Skip no moving goal to not remain stuck
        while waypoints[goal_index][2] <= 0.1: goal_index += 1  

        # -------------- TRAFFIC LIGHTS --------------------------
        # Check for traffic lights presence
        traffic_lights_index, traffic_light_present, traffic_light_state, distance_from_traffic_lights, tl_line = \
            self.check_for_traffic_lights(waypoints, is_close_itc, closest_index, goal_index, ego_state)
        
        # Draw trafficlight stop line if present
        if traffic_light_present: self._draw(tl_line, angle=ego_direction, short='-', settings=dict(markersize=10, label='Trafficlight'))

        # Update input info about trafficlights
        self._current_input += f'\n - Traffic lights: ' + \
            f'{"no" if not traffic_light_present else ""}' + \
            (f'{"GREEN" if traffic_light_state == 0 else "YELLOW" if traffic_light_state == 1 else "RED"}, distance={round(distance_from_traffic_lights, 2)} m' if traffic_light_present else '') 

        # --------------- VEHICLES --------------------------------
        # Check for vehicle presence
        vehicle_presence, vehicles, veh_chech_area = self.check_for_vehicle(ego_point, goal_path)

        # Draw all the found vehicle
        for i, v in enumerate([v[3] for v in vehicles]):
            self._draw(v, angle=ego_direction, short='r--', settings=dict(label=f'Vehicle {i}'))
        # Draw the vehicle check area
        self._draw(veh_chech_area, angle=ego_direction, short='b--', settings=dict(label='Check vehicle area'))

        # Update input info about vehicles
        for i, v in enumerate([v for v in vehicles]):
            self._current_input += f'\n - Vehicle {i}: ' + \
                f'Position={tuple((round(x, 1) for x in v[0][:2]))}, Speed={round(v[1], 2)} m/s, Distance={round(v[2], 2)} m'

        # --------------- PEDESTRIANS ------------------------------
        # Check for pedestrian presence
        pedestrian_presence, pedestrians, ped_chech_area = self.check_for_pedestrians(ego_point, goal_path)
        # Draw all the found pedestrian
        for i, p in enumerate(pedestrians):
            self._draw(p, angle=ego_direction, short='r:', settings=dict(label=f'Pedestrian {i}'))
        # Draw the pedestrian check area
        self._draw(ped_chech_area, angle=ego_direction, short='--', settings=dict(label='Check vehicle area'))

        # Update input info about vehicles
        for i, p in enumerate([p for p in pedestrians]):
            self._current_input += f'\n - Pedestrian {i}: ' + \
                f'Position={tuple((round(x, 1) for x in p[0][:2]))}, Speed={round(p[1], 2)} m/s, Distance={round(p[2], 2)} m'

        # --------------- Update current input draw ----------------
        self._finalize_draw()

        # ------------- FSM EVOLUTION -------------------------------
        self._goal_index = goal_index
        self._goal_state = waypoints[goal_index]

        # if vehicle_presence:
        #     self._follow_lead_vehicle = True
        #     self._lead_car_state = [*vehicle_position[0:2], vehicle_speed]
        # else:
        #     self._follow_lead_vehicle = False
        #     self._lead_car_state = None

        # # FOLLOW_LANE: In this state the vehicle move to reach the goal.
        # if self._state == FOLLOW_LANE:

        #     # 0,x,x,x; 1,G,x,x
        #     if not traffic_light_present or (traffic_light_present and traffic_light_state == TRAFFICLIGHT_GREEN):
        #         # Set the next state
        #         self._state = FOLLOW_LANE
        #         # Set goal
        #         self._goal_index = goal_index
        #         self._goal_state = waypoints[goal_index]

        #     # 1,Y,1,x
        #     elif traffic_light_present and \
        #             (traffic_light_state == TRAFFICLIGHT_YELLOW) and \
        #             not (distance_from_traffic_lights / ego_state[3] < TRAFFICLIGHT_YELLOW_MIN_TIME):
        #         # Set the next state
        #         self._state = FOLLOW_LANE         
        #         # Set goal
        #         self._goal_index = goal_index
        #         self._goal_state = waypoints[goal_index]
            
        #     # 1,Y,0,x
        #     elif traffic_light_present and \
        #             (traffic_light_state == TRAFFICLIGHT_GREEN) and \
        #             (distance_from_traffic_lights / ego_state[3] < TRAFFICLIGHT_YELLOW_MIN_TIME):
        #         # Set the next state
        #         self._state = DECELERATE_TO_STOP         
        #         # Set goal
        #         self._goal_index = traffic_lights_index
        #         self._goal_state = waypoints[traffic_lights_index]

        #     # 1,R,x,x
        #     elif traffic_light_present and \
        #             (traffic_light_state == TRAFFICLIGHT_RED):
        #         # Set the next state
        #         self._state = DECELERATE_TO_STOP         
        #         # Set goal
        #         self._goal_index = traffic_lights_index
        #         self._goal_state = waypoints[traffic_lights_index]
        
        # # DECELERATE_TO_STOP: In this state we suppose to have enough space to slow down until the 
        # # stop line. 
        # elif self._state == DECELERATE_TO_STOP:
                        
        #     # 0,x,x,x, 1,G,x,x
        #     if not traffic_light_present or \
        #             (traffic_light_present and (traffic_light_state == TRAFFICLIGHT_GREEN)):
        #         # Set the next state
        #         self._state = FOLLOW_LANE
        #         # Set goal
        #         self._goal_index = goal_index
        #         self._goal_state = waypoints[goal_index]

        #     # 1,R,x,0; 1,Y,x,0
        #     elif traffic_light_present and \
        #             (traffic_light_state == TRAFFICLIGHT_YELLOW or traffic_light_state == TRAFFICLIGHT_RED) and \
        #             not(abs(closed_loop_speed) <= STOP_THRESHOLD):
        #         # Set the next state
        #         self._state = DECELERATE_TO_STOP
        #         # Set goal
        #         self._goal_index = traffic_lights_index
        #         self._goal_state = waypoints[traffic_lights_index]
        #         self._goal_state[2] = 0

        #     # 1,R,x,1; 1,Y,x,1
        #     elif traffic_light_present and \
        #             (traffic_light_state == TRAFFICLIGHT_YELLOW or traffic_light_state == TRAFFICLIGHT_RED) and \
        #             (abs(closed_loop_speed) <= STOP_THRESHOLD):
        #         # Set the next state
        #         self._state = STAY_STOPPED
        #         # Set goal
        #         self._goal_index = traffic_lights_index
        #         self._goal_state = waypoints[traffic_lights_index]
        #         self._goal_state[2] = 0

        # # STAY_STOPPED: In this state the vehicle is stopped, waiting for the green light.
        # elif self._state == STAY_STOPPED:

        #     # 0,0,0,0; 1,G,x,x                
        #     if not traffic_light_present or \
        #             (traffic_light_present and (traffic_light_state == TRAFFICLIGHT_GREEN)):
        #         # Set the next state
        #         self._state = FOLLOW_LANE
        #         # Set goal
        #         self._goal_index = goal_index
        #         self._goal_state = waypoints[goal_index]

        #     # 1,Y,x,x; 1,R,x,x  
        #     elif traffic_light_present and \
        #             (traffic_light_state == TRAFFICLIGHT_YELLOW or traffic_light_state == TRAFFICLIGHT_RED):
        #         # Set the next state
        #         self._state = STAY_STOPPED
        #         # Set goal
        #         self._goal_index = traffic_lights_index
        #         self._goal_state = waypoints[traffic_lights_index]
        #         self._goal_state[2] = 0
                
        # else:
        #     raise ValueError('Invalid state value.')

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    # TODO: UPDATE DOCSTRING
    def get_goal_index(self, waypoints, ego_point, ego_direction, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_point (Point): MISSING
            ego_direction (float): MISSING
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
            is_ego_in_the_middle (bool): MISSING
            goal_line (LineString): MISSING
        """
        # check if ego state is in the middle of wp_i and wp_i+1 with 10^-2 meters precision
        closest_point = Point(waypoints[closest_index][0], waypoints[closest_index][1])
        dist_ego2close = ego_point.distance(closest_point)

        ego_plus_cdist = Point(ego_point.x + closest_len * np.cos(ego_direction), ego_point.y + closest_len * np.sin(ego_direction))
        dist_close2egop = closest_point.distance(ego_plus_cdist)

        is_ego_itm = dist_close2egop > dist_ego2close

        # compute a list with all the points into the path
        arc_points = [ego_point]

        if is_ego_itm and not (closest_index == len(waypoints) - 1):  # if closest point is before the ego skip that point, it's not useful
            wp_index = closest_index + 1
        else:
            wp_index = closest_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            wp_point = Point(waypoints[wp_index][0], waypoints[wp_index][1])
            arc_points.append(wp_point)
            goal_index = wp_index

        # Otherwise, find our next waypoint.
        else:
            arc_length = 0
            wp_i1 = ego_point
            while wp_index < len(waypoints) - 1:
                wp_i2 = Point(waypoints[wp_index][0], waypoints[wp_index][1])
                arc_points.append(wp_i2)
                arc_length += wp_i1.distance(wp_i2)
                
                if arc_length > self._lookahead: break
                wp_index += 1

            goal_index = wp_index

        # draw the arc_line
        arc = LineString(arc_points)

        return goal_index % len(waypoints), is_ego_itm, arc

    # TODO: UPDATE DOCSTRING
    def check_for_traffic_lights(self, waypoints, is_ego_itm, closest_index, goal_index, ego_state):
        """Checks for a traffic light that is intervening the goal path.

        Checks for a traffic light that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a stop line), a
        boolean flag indicating if a stop sign obstruction was found, the state of
        the traffic light, and the distance from the closest point.
        
        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            is_ego_itm (bool): MISSING
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
            goal_index (current): Current goal index for the vehicle to reach
                i.e. waypoints[goal_index] gives the goal waypoint
            ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
        """
        # Default return parameter
        intersection_flag = False
        traffic_light_state = None
        dist_from_tl = float('inf')
        tl_line = None

        # If ego is after the closest index, consider that
        if is_ego_itm:
            waypoints = np.insert(waypoints, closest_index*waypoints.shape[1], np.array([ego_state[0], ego_state[1], 0])).reshape(waypoints.shape[0]+1, waypoints.shape[1])
            goal_index += 1

        for i in range(closest_index, goal_index):
            # Check to see if path segment crosses any of the stop lines.

            for key, traffic_light_fence in enumerate(self._traffic_lights['fences']):
                # Ideal path segment
                path_wp1_wp2 = LineString([waypoints[i][0:2], waypoints[i+1][0:2]])
                # Traffic Light segment
                tl_line = LineString([traffic_light_fence[0:2], traffic_light_fence[2:4]])
                # intersection between the Ideal path and the Traffic Light line.
                intersection_flag = path_wp1_wp2.intersects(tl_line)

                # If there is an intersection with a stop line, update the goal state to stop before the goal line.
                if intersection_flag:
                    intersection_point = Point(path_wp1_wp2.intersection(tl_line).coords)

                    goal_index = i
                    traffic_light_state = self._traffic_lights['states'][key]

                    ego_point = Point(ego_state[0], ego_state[1])
                    dist_from_tl = ego_point.distance(intersection_point)
                    break
            
            if intersection_flag:
                break

        # Fix the added waypoint for the calculation
        if is_ego_itm:
            goal_index -= 1

        return goal_index, intersection_flag, traffic_light_state, dist_from_tl, tl_line

    # TODO: Update docstring
    def check_for_vehicle(self, ego_point, goal_path):
        """UPDATE
        """
        # Default return parameter
        intersection_flag = False
        vehicle_position = None
        vehicle_speed = 0

        # Starting from the goal line, create an area to check for vehicle
        path_bb = goal_path.buffer(BB_PATH, cap_style=CAP_STYLE.flat)

        # Check all vehicles whose bounding box intersects the control area
        intersection = []
        for key, vehicle_bb in enumerate(self._vehicle['fences']):
            vehicle = Polygon(vehicle_bb)

            if vehicle.intersects(path_bb):
                other_vehicle_point = Point(self._vehicle['position'][key][0], self._vehicle['position'][key][1])
                dist_from_vehicle = ego_point.distance(other_vehicle_point)

                vehicle_position = self._vehicle['position'][key]
                vehicle_speed = self._vehicle['speeds'][key]

                intersection.append([vehicle_position, vehicle_speed, dist_from_vehicle, vehicle])

        # The lead vehicle can be said to be present if there is at least one vehicle in the area.
        intersection_flag = len(intersection) > 0

        # Sort the vehicle by their distance from ego
        intersection = sorted(intersection, key=lambda x: x[2])

        return intersection_flag, intersection, path_bb

    # TODO: Update Docstring
    def check_for_pedestrians(self, ego_point, goal_path):
        """UPDATE
        """
        # Default return parameter
        intersection_flag = False
        pedestrian_position = None
        pedestrian_speed = 0

        # Starting from the goal line, create an area to check for vehicle. The area, in this case,
        # was created as teh union of a little area on the right and a laregr area (to check pedestrian crossing).
        # NOTE: The sign are inverted respect to the shapely documentation because shapely use a reverse x choords.
        path_bb = unary_union([goal_path.buffer(BB_PEDESTRIAN_RIGHT, single_sided=True), goal_path.buffer(-BB_PEDESTRIAN_LEFT, single_sided=True)])

        # Check all pedestrians whose bounding box intersects the control area
        intersection = []
        for key, pedestrian_bb in enumerate(self._pedestrians['fences']):
            pedestrian = Polygon(pedestrian_bb)

            if pedestrian.intersects(path_bb):
                other_pedestrian_point = Point(self._pedestrians['position'][key][0], self._pedestrians['position'][key][1])
                dist_from_pedestrian = ego_point.distance(other_pedestrian_point)

                pedestrian_position = self._pedestrians['position'][key]
                pedestrian_speed = self._pedestrians['speeds'][key]

                intersection.append([pedestrian_position, pedestrian_speed, dist_from_pedestrian, pedestrian])

        # A pedestrian can be said to be present if there is at least one vehicle in the area.
        intersection_flag = len(intersection) > 0

        # Sort the vehicle by their distance from ego
        intersection = sorted(intersection, key=lambda x: x[2])

        return intersection_flag, intersection, path_bb

# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.        
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False
