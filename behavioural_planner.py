#!/usr/bin/env python3
from turtle import distance, position, shape
from unicodedata import decimal
from matplotlib import markers
from matplotlib.pyplot import legend
import numpy as np
import math
from shapely.geometry import Point, LineString, Polygon


# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
# Stop speed threshold
STOP_THRESHOLD = 0.1

TRAFFICLIGHT_GREEN = 0
TRAFFICLIGHT_YELLOW = 1
TRAFFICLIGHT_RED = 2
TRAFFICLIGHT_YELLOW_MIN_TIME = 3.5  # sec 

BB_PATH = 1.5  # m


class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead, traffic_lights, vehicle):
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

        self._to_draw = []
        self._state_info = ''
    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def set_traffic_light(self, traffic_lights):
        for k in traffic_lights:
            self._traffic_lights[k] = traffic_lights[k]
    
    def set_vehicle(self,vehicle):
        for k in vehicle:
            self._vehicle[k]=vehicle[k]

    def get_state_info(self):
        return self._state_info

    def get_to_draw(self):
        to_ret = self._to_draw
        self._to_draw = []
        return to_ret

    def draw(self, obj, setting_dict):
        for e in self._to_draw:
            if e == obj:
                return
        self._to_draw.append((obj, setting_dict))

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
        # Draw the ego-state
        self.draw(Point(ego_state[0], ego_state[1]).coords.xy, dict(color='#8dc576', marker='.', markersize=20, label='ego state'))

        # Get closest waypoint
        closest_len, closest_index = get_closest_index(waypoints, ego_state)
        # Get goal based on the current lookahead
        goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
        while waypoints[goal_index][2] <= 0.1: goal_index += 1  # Skip no moving goal to not remain stuck

        # Check for traffic lights presence
        traffic_lights_index, traffic_light_present, traffic_light_state, distance_from_traffic_lights = \
            self.check_for_traffic_lights(waypoints, closest_index, goal_index, ego_state)

        vehicle_index, vehicle_presence, vehicle_position, vehicle_speed, dist_from_vehicle = \
            self.check_lead_vehicle(waypoints, closest_index, goal_index, ego_state)

        # Set debug state info
        self._state_info = f'Current State: {"FOLLOW_LANE" if self._state == 0 else "DECELERATE" if self._state == 1 else "STOPPED"}' + \
            f'\n\nCurrent Input:' + \
            f'\n  - ego-state: {[round(x, 2) for x in ego_state]}' + \
            f'\n  - Traffic lights: {"no" if not traffic_light_present else ""}' + \
            (f'{"green" if traffic_light_state == 0 else "yellow" if traffic_light_state == 1 else "red"}' + \
             f', distance: {round(distance_from_traffic_lights, 2)} m' if traffic_light_present else '') + \
            (f'\n  - Lead vehicle: {"no" if not vehicle_presence else ""}') + \
            (f'{[round(x, 2) for x in vehicle_position]}, speed: {round(vehicle_speed, 2)} m/s, distance: {round(dist_from_vehicle, 2)} m' if vehicle_presence else '')

        if vehicle_presence:
            self._follow_lead_vehicle = True
            self._lead_car_state = [*vehicle_position[0:2], vehicle_speed]
        else:
            self._follow_lead_vehicle = False
            self._lead_car_state = None

        # FOLLOW_LANE: In this state the vehicle move to reach the goal.
        if self._state == FOLLOW_LANE:

            # 0,x,x,x; 1,G,x,x
            if not traffic_light_present or (traffic_light_present and traffic_light_state == TRAFFICLIGHT_GREEN):
                # Set the next state
                self._state = FOLLOW_LANE
                # Set goal
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]

            # 1,Y,1,x
            elif traffic_light_present and \
                    (traffic_light_state == TRAFFICLIGHT_YELLOW) and \
                    not (distance_from_traffic_lights / ego_state[3] < TRAFFICLIGHT_YELLOW_MIN_TIME):
                # Set the next state
                self._state = FOLLOW_LANE         
                # Set goal
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
            
            # 1,Y,0,x
            elif traffic_light_present and \
                    (traffic_light_state == TRAFFICLIGHT_GREEN) and \
                    (distance_from_traffic_lights / ego_state[3] < TRAFFICLIGHT_YELLOW_MIN_TIME):
                # Set the next state
                self._state = DECELERATE_TO_STOP         
                # Set goal
                self._goal_index = traffic_lights_index
                self._goal_state = waypoints[traffic_lights_index]

            # 1,R,x,x
            elif traffic_light_present and \
                    (traffic_light_state == TRAFFICLIGHT_RED):
                # Set the next state
                self._state = DECELERATE_TO_STOP         
                # Set goal
                self._goal_index = traffic_lights_index
                self._goal_state = waypoints[traffic_lights_index]
        
        # DECELERATE_TO_STOP: In this state we suppose to have enough space to slow down until the 
        # stop line. 
        elif self._state == DECELERATE_TO_STOP:
                        
            # 0,x,x,x, 1,G,x,x
            if not traffic_light_present or \
                    (traffic_light_present and (traffic_light_state == TRAFFICLIGHT_GREEN)):
                # Set the next state
                self._state = FOLLOW_LANE
                # Set goal
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]

            # 1,R,x,0; 1,Y,x,0
            elif traffic_light_present and \
                    (traffic_light_state == TRAFFICLIGHT_YELLOW or traffic_light_state == TRAFFICLIGHT_RED) and \
                    not(abs(closed_loop_speed) <= STOP_THRESHOLD):
                # Set the next state
                self._state = DECELERATE_TO_STOP
                # Set goal
                self._goal_index = traffic_lights_index
                self._goal_state = waypoints[traffic_lights_index]
                self._goal_state[2] = 0

            # 1,R,x,1; 1,Y,x,1
            elif traffic_light_present and \
                    (traffic_light_state == TRAFFICLIGHT_YELLOW or traffic_light_state == TRAFFICLIGHT_RED) and \
                    (abs(closed_loop_speed) <= STOP_THRESHOLD):
                # Set the next state
                self._state = STAY_STOPPED
                # Set goal
                self._goal_index = traffic_lights_index
                self._goal_state = waypoints[traffic_lights_index]
                self._goal_state[2] = 0

        # STAY_STOPPED: In this state the vehicle is stopped, waiting for the green light.
        elif self._state == STAY_STOPPED:

            # 0,0,0,0; 1,G,x,x                
            if not traffic_light_present or \
                    (traffic_light_present and (traffic_light_state == TRAFFICLIGHT_GREEN)):
                # Set the next state
                self._state = FOLLOW_LANE
                # Set goal
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]

            # 1,Y,x,x; 1,R,x,x  
            elif traffic_light_present and \
                    (traffic_light_state == TRAFFICLIGHT_YELLOW or traffic_light_state == TRAFFICLIGHT_RED):
                # Set the next state
                self._state = STAY_STOPPED
                # Set goal
                self._goal_index = traffic_lights_index
                self._goal_state = waypoints[traffic_lights_index]
                self._goal_state[2] = 0
                
        else:
            raise ValueError('Invalid state value.')

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
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
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_line = LineString([(waypoints[wp_index][0], waypoints[wp_index][1]), (waypoints[wp_index+1][0], waypoints[wp_index+1][1])])
            self.draw(arc_line.coords.xy, dict(color='#ad76c5', marker='.', markersize=7))
            arc_length += arc_line.length
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)

    def check_for_traffic_lights(self, waypoints, closest_index, goal_index, ego_state):
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
        intersection_flag = False
        traffic_light_state = None
        dist_from_tl = float('inf')

        # Add ego_state as waypoint if is before the closest point
        ego_point = Point(ego_state[0], ego_state[1])
        closest_point = Point(waypoints[closest_index][0], waypoints[closest_index][1])
        distance_ego2closest = ego_point.distance(closest_point)
        # Project the ego point for distance_ego2closest meters in the ego direction and check if that is the same of the closest
        to_check_point = Point(ego_point.x + distance_ego2closest * np.cos(ego_state[2]),
                               ego_point.y + distance_ego2closest * np.sin(ego_state[2]))
        distance_closest2check = closest_point.distance(to_check_point)
        added_waypoint =  distance_closest2check < distance_ego2closest
        if added_waypoint:
            waypoints = np.insert(waypoints, closest_index*waypoints.shape[1], np.array([ego_state[0], ego_state[1], 0])).reshape(waypoints.shape[0]+1, waypoints.shape[1])
            goal_index += 1

        for i in range(closest_index, goal_index):
            # Check to see if path segment crosses any of the stop lines.

            for key, traffic_light_fence in enumerate(self._traffic_lights['fences']):
                # Ideal path segment
                path_wp1_wp2 = LineString([waypoints[i][0:2], waypoints[i+1][0:2]])
                self.draw(path_wp1_wp2.coords.xy, dict(color='#c576b5', marker='.', linestyle='--', markersize=10))
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

                    to_check_point = Point(ego_point.x + dist_from_tl * np.cos(ego_state[2]),
                                           ego_point.y + dist_from_tl * np.sin(ego_state[2]))

                    # Check if passed the line. In that case, invert the sign.
                    if to_check_point.distance(intersection_point) > dist_from_tl:
                        dist_from_tl *= -1

                    self.draw(tl_line.coords.xy, dict(color='#c58676', linewidth=2.5, label='traffic light'))
                    break
            
            if intersection_flag:
                break

        if added_waypoint:
            goal_index -= 1
        return goal_index, intersection_flag, traffic_light_state, dist_from_tl

    def check_lead_vehicle(self, waypoints, closest_index, goal_index, ego_state):
        """Checks for a lead vehicle that is intervening the goal path.

        Checks for a lead vehicle that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a stop line), a
        boolean flag indicating if a stop sign obstruction was found, the state of
        the lead vehicle, and the distance from the closest point.
        
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
        distance_seen = 0
        intersection_flag = False
        vehicle_position = None
        vehicle_speed = 0
        dist_from_intersection = float('inf')

        # Check if ego is the first waypoint because close vehicle can be not checked.
        ego_point = Point(ego_state[0], ego_state[1])
        closest_point = Point(waypoints[closest_index][0], waypoints[closest_index][1])
        dist_from_closest = ego_point.distance(closest_point)
        to_check_point = Point(ego_point.x + dist_from_closest * np.cos(ego_state[2]),
                               ego_point.y + dist_from_closest * np.sin(ego_state[2]))
        dist_from_to_check_point = to_check_point.distance(closest_point)
        added_waypoint = dist_from_to_check_point < dist_from_closest
        if added_waypoint:
            waypoints = np.insert(waypoints, closest_index*waypoints.shape[1], np.array([ego_state[0], ego_state[1], 0])).reshape(waypoints.shape[0]+1, waypoints.shape[1])
            goal_index += 1

        # Check until the Lookahead 
        if not intersection_flag:
            for i in range(closest_index, len(waypoints)):
                # Check to see if path segment crosses any of the stop lines.
                path_wp1_wp2 = LineString([waypoints[i][0:2], waypoints[i+1][0:2]])
                distance_seen += path_wp1_wp2.length

                _dy = waypoints[i+1][1] - waypoints[i][1]
                _dx = waypoints[i+1][0] - waypoints[i][0]
                path_angle = np.arctan(_dy/_dx)

                # path_wp1_wp2 but wider
                path_bb = Polygon([(waypoints[i][0] + BB_PATH * np.cos(path_angle - np.pi / 2),
                                    waypoints[i][1] + BB_PATH * np.sin(path_angle - np.pi / 2)),
                                   (waypoints[i][0] + BB_PATH * np.cos(path_angle + np.pi / 2),
                                    waypoints[i][1] + BB_PATH * np.sin(path_angle + np.pi / 2)),
                                   (waypoints[i+1][0] + BB_PATH * np.cos(path_angle + np.pi / 2),
                                    waypoints[i+1][1] + BB_PATH * np.sin(path_angle + np.pi / 2)),
                                   (waypoints[i+1][0] + BB_PATH * np.cos(path_angle - np.pi / 2),
                                    waypoints[i+1][1] + BB_PATH * np.sin(path_angle - np.pi / 2))])
                
                self.draw(path_wp1_wp2.coords.xy, dict(color='#76b5c5', linestyle=':'))
                self.draw(path_bb.exterior.xy, dict(color='#76b5c5', linestyle=':'))

                for key, vehicle_bb in enumerate(self._vehicle['fences']):
                    vehicle = Polygon(vehicle_bb)

                    intersection_flag = vehicle.intersects(path_bb)

                    if intersection_flag:
                        goal_index = i 

                        other_vehicle_point = Point(self._vehicle['position'][key][0], self._vehicle['position'][key][1])
                        dist_from_intersection = ego_point.distance(other_vehicle_point)

                        if dist_from_intersection > self._follow_lead_vehicle_lookahead:
                            intersection_flag = False
                        else:
                            vehicle_position = self._vehicle['position'][key]
                            vehicle_speed = self._vehicle['speeds'][key]
                            self.draw(vehicle.exterior.xy, dict(color='red', linestyle='-', label='vehicle intersection'))
                        break
                
                if intersection_flag or (distance_seen >= self._follow_lead_vehicle_lookahead):
                    break

        if added_waypoint:
            goal_index -= 1

        return goal_index, intersection_flag, vehicle_position, vehicle_speed, dist_from_intersection
                
    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        if not self._follow_lead_vehicle:
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.   
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector, 
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), 
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector, 
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
                return
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False

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
