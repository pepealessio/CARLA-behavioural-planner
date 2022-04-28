#!/usr/bin/env python3
import numpy as np
from shapely.geometry import Point, LineString, Polygon, CAP_STYLE
from shapely.affinity import rotate
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import behaviourial_fsm


# Define x dimension of the bounding box to check for obstacles.
BB_PATH = 1.5  # m
BB_PEDESTRIAN_LEFT = 4 # m
BB_PEDESTRIAN_RIGHT = 1.5 # m


class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead, traffic_lights, vehicle, pedestrians):
        self._fsm                           = behaviourial_fsm.get_fsm()
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = self._fsm.get_current_state()
        self._lead_car_state                = None
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count                    = 0
        self._lookahead_collision_index     = 0
        self._waypoints                     = None
        self._traffic_lights                = traffic_lights
        self._vehicle                       = vehicle
        self._pedestrians                   = pedestrians
        self._state_info                    = ''
        self._current_input                 = ''
        self._before_tl_present             = False
        self._before_vehicle_present        = False
        self._before_pedestrian_present     = False
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
        self._legend = None

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
        self._legend = self._fig.legend()
        plt.pause(.001)
        self._ax.cla()
        if self._legend is not None: 
            self._legend.remove()
            self._legend = None
        self._ax.set_aspect('equal', 'datalim')
        self._ax.invert_xaxis()
    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def get_waypoints(self):
        return self._waypoints

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
        self._waypoints = waypoints

        # Update state info
        self._state_info = f'Current State: {self._state}'
        # Update input info
        self._current_input = 'Current inputs:'

        # ---------------- EGO STATE ---------------------------
        # Transform ego info in shapely geometry
        ego_point = Point(ego_state[0], ego_state[1])
        ego_direction = ego_state[2]
        # Draw the ego state point.
        self._draw(ego_point, angle=ego_direction, short='g.', settings=dict(markersize=35, label='Ego Point'))
        # Update input info about trafficlights
        self._current_input += f'\n - Ego state: Position={tuple((round(x, 1) for x in ego_point.coords[0]))}, Orientation={round(np.degrees(ego_direction))}°, Velocity={round(closed_loop_speed, 2)} m/s'

        # ---------------- GOAL INDEX ---------------------------
        # Get closest waypoint
        closest_len, closest_index = get_closest_index(waypoints, ego_point)

        # Get goal based on the current lookahead
        goal_index, goal_path = self.get_goal_index(waypoints, ego_point, ego_direction, closest_len, closest_index)
        self._draw(goal_path, angle=ego_direction, short='y^-', settings=dict(markersize=10, label='Goal Path'))

        # Skip no moving goal to not remain stuck
        while waypoints[goal_index][2] <= 0.1: goal_index += 1  

        # -------------- TRAFFIC LIGHTS --------------------------
        # Check for traffic lights presence
        traffic_light_presence, traffic_lights = self.check_for_traffic_lights(ego_point, goal_path)
        
        # Draw all the traffic lights
        for i, tl in enumerate([tl[3] for tl in traffic_lights]):
            self._draw(tl, angle=ego_direction, short='-', settings=dict(markersize=10, color='#ff7878', label=f'Trafficlight {i}'))

        # Update input info about trafficlights
        for i, tl in enumerate([tl for tl in traffic_lights]):
            self._current_input += f'\n - Traffic lights {i}: ' + \
            f'{"GREEN" if tl[1] == 0 else "YELLOW" if tl[1] == 1 else "RED"}, distance={round(tl[2], 2)} m'
        
        # --------------- VEHICLES --------------------------------
        # Check for vehicle presence
        vehicle_presence, vehicles, veh_chech_area = self.check_for_vehicle(ego_point, goal_path)

        # Draw all the found vehicle
        for i, v in enumerate([v[4] for v in vehicles]):
            self._draw(v, angle=ego_direction, short='--', settings=dict(color='#ff4d4d', label=f'Vehicle {i}'))
        # Draw the vehicle check area
        self._draw(veh_chech_area, angle=ego_direction, short='b--', settings=dict(linewidth=2, label='Check vehicle area'))

        # Update input info about vehicles
        for i, v in enumerate([v for v in vehicles]):
            self._current_input += f'\n - Vehicle {i}: ' + \
                f'Position={tuple((round(x, 1) for x in v[1][:2]))}, Speed={round(v[2], 2)} m/s, Distance={round(v[3], 2)} m'

        # --------------- PEDESTRIANS ------------------------------
        # Check for pedestrian presence
        pedestrian_presence, pedestrians, ped_chech_area = self.check_for_pedestrians(ego_point, goal_path)
        # Draw all the found pedestrian
        for i, p in enumerate(pedestrians):
            self._draw(p[4], angle=ego_direction, short='--', settings=dict(color='#fc2626', label=f'Pedestrian {i}'))
        # Draw the pedestrian check area
        self._draw(ped_chech_area, angle=ego_direction, short='c:', settings=dict(label='Check pedestrian area'))
        if pedestrian_presence:
            self._draw(Point(self._waypoints[pedestrians[0][0]][0:2]), ego_direction, "r.", dict(markersize=20))

        # Update input info about vehicles
        for i, p in enumerate([p for p in pedestrians]):
            self._current_input += f'\n - Pedestrian {i}: ' + \
                f'Position={tuple((round(x, 1) for x in p[1][:2]))}, Speed={round(p[2], 2)} m/s, Distance={round(p[3], 2)} m'

        # --------------- Update presence of obstacles -------------
        # self._before_pedestrian_present = pedestrian_presence
        # self._before_vehicle_present = vehicle_presence
        # self._before_tl_present = traffic_light_presence

        # --------------- Update current input draw ----------------
        self._finalize_draw()

        # ------------- FSM EVOLUTION -------------------------------
        # Set the input
        self._fsm.set_readings(waypoints=waypoints,
                               ego_state=ego_state,
                               closed_loop_speed=closed_loop_speed,
                               goal_index=goal_index,
                               traffic_light_presence=traffic_light_presence, 
                               traffic_lights=traffic_lights, 
                               vehicle_presence=vehicle_presence,
                               vehicles=vehicles,
                               pedestrian_presence=pedestrian_presence, 
                               pedestrians=pedestrians)
        # Evolve the FSM
        self._fsm.process()

        # Get the current state
        self._state = self._fsm.get_current_state()

        # Get the output
        self._goal_index = self._fsm.get_from_memory('goal_index')
        self._goal_state = self._fsm.get_from_memory('goal_state')
        self._follow_lead_vehicle = self._fsm.get_from_memory('follow_lead_vehicle')
        self._lead_car_state = self._fsm.get_from_memory('lead_car_state')

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
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
            ego_point (Point): the point represent the position of the vehicle.
            ego_direction (float): the angle who represent the direction of teh vehicle
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
            is_ego_in_the_middle (bool): If the vehicle is after the closest index
            goal_line (LineString): a linestring representing the ideal path
        """
        # list with all the points into the path
        arc_points = [ego_point]

        # Update lookahead if before something was present
        lookahead = self._lookahead
        if self._before_tl_present or self._before_vehicle_present or self._before_pedestrian_present:
            lookahead += 15  # m of margin
        
        if check_is_after(self._waypoints, ego_point, closest_index, margin=1):
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
                
                if arc_length > lookahead: break
                wp_index += 1

            goal_index = wp_index

        # draw the arc_line
        arc = LineString(arc_points)

        return goal_index % len(waypoints), arc

    def check_for_traffic_lights(self, ego_point, goal_path):
        """Check in the path for presence of vehicle.

        Args:
            ego_point (Point): The point represent the position of the vehicle.
            goal_path (LineString): The linestring represent the path until the goal.

        Returns:
            [intersection_flag, intersection]:
                intersection_flag (bool): If true, at least one vehicle is present.
                intersection (List[Tuple[int, int, float, LineString]]): a list containing, for each traffic light, the closest index,
                    the traffic light state, the distance from the traffic light, and the stop line of the traffic line."""
        # Check for all traffic lights 
        intersection = []
        for key, traffic_light_fence in enumerate(self._traffic_lights['fences']):

            # Traffic Light segment
            tl_line = LineString([traffic_light_fence[0:2], traffic_light_fence[2:4]])
            # intersection between the Ideal path and the Traffic Light line.
            intersection_flag = goal_path.intersects(tl_line)

            # If there is an intersection with a stop line, update the goal state to stop before the goal line.
            if intersection_flag:
                intersection_point = Point(goal_path.intersection(tl_line).coords)

                closest_index = self.get_stop_index(ego_point, intersection_point)
                dist_from_tl = ego_point.distance(intersection_point)
                traffic_light_state = self._traffic_lights['states'][key]

                intersection.append([closest_index, traffic_light_state, dist_from_tl, tl_line])

        # The trafficlight can be said to be present if there is at least one trafficlight in the area.
        intersection_flag = len(intersection) > 0

        return intersection_flag, intersection

    def check_for_vehicle(self,ego_point, goal_path):
        """Check in the path for presence of vehicle.

        Args:
            ego_point (Point): The point represent the position of the vehicle.
            goal_path (LineString): The linestring represent the path until the goal.

        Returns:
            [intersection_flag, intersection, path_bb]:
                intersection_flag (bool): If true, at least one vehicle is present.
                intersection (List[Tuple[int, Point, float, float, Polygon]]): a list containing, for each vehicle, the closest index,
                    the position point, the speed, the distance from the vehicle, and the bounding box of the vehicle.
                path_bb (Polygon): the check area.
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
                closest_index = self.get_stop_index(ego_point, other_vehicle_point)
                dist_from_vehicle = ego_point.distance(other_vehicle_point)

                vehicle_position = self._vehicle['position'][key]
                vehicle_speed = self._vehicle['speeds'][key]

                intersection.append([closest_index, vehicle_position, vehicle_speed, dist_from_vehicle, vehicle])

        # The lead vehicle can be said to be present if there is at least one vehicle in the area.
        intersection_flag = len(intersection) > 0

        # Sort the vehicle by their distance from ego
        intersection = sorted(intersection, key=lambda x: x[3])

        return intersection_flag, intersection, path_bb

    def check_for_pedestrians(self,ego_point, goal_path):
        """Check in the path for presence of pedestrians.

        Args:
            ego_point (Point): The point represent the position of the vehicle.
            goal_path (LineString): The linestring represent the path until the goal.

        Returns:
            [intersection_flag, intersection, path_bb]:
                intersection_flag (bool): If true, at least one pedestrian is present.
                intersection (List[Tuple[int, Point, float, float, Polygon]]): a list containing, for each pedestrian, the closest index,
                    the position point, the speed, the distance from the pedestrian, and the bounding box of the pedestrian.
                path_bb (Polygon): the check area.
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
                closest_index = self.get_stop_index(ego_point, other_pedestrian_point)
                dist_from_pedestrian = ego_point.distance(other_pedestrian_point)

                pedestrian_position = self._pedestrians['position'][key]
                pedestrian_speed = self._pedestrians['speeds'][key]

                intersection.append([closest_index, pedestrian_position, pedestrian_speed, dist_from_pedestrian, pedestrian])

        # A pedestrian can be said to be present if there is at least one vehicle in the area.
        intersection_flag = len(intersection) > 0

        # Sort the vehicle by their distance from ego
        intersection = sorted(intersection, key=lambda x: x[3])

        return intersection_flag, intersection, path_bb
    
    # NOTE: case obstacle_index = 0 is not defined 
    def get_stop_index(self, ego_point, obstacle_point, margin=1):
        _, obstacle_index = get_closest_index(self._waypoints,obstacle_point)
        is_obstacle_before, obst_proj_point = check_is_before(self._waypoints, obstacle_point, obstacle_index)

        is_ego_before_prev, ego_proj_point = check_is_before(self._waypoints, ego_point, obstacle_index-1)
        
        if not is_obstacle_before:
            stop_index = obstacle_index
        else:
            if is_ego_before_prev:
                stop_index = obstacle_index-1
            else:
                # If there's not enough space between the waypoints, don't add the waypoint
                middle_point = LineString([ego_proj_point, obst_proj_point]).interpolate(0.4, normalized=True)
                distance_from_closest, closest_index = get_closest_index(self._waypoints, middle_point)
                if distance_from_closest > margin:
                    # Add new waypoint
                    wps = self._waypoints.tolist()
                    wps.insert(obstacle_index, [middle_point.x, middle_point.y, self._waypoints[obstacle_index][2]])
                    self._waypoints = np.array(wps)

                    stop_index = obstacle_index
                else:
                    stop_index = closest_index
        return stop_index 

def get_closest_index(waypoints, point):
    """Gets closest index a given list of waypoints to the point.

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
        point (Point): position to see. (global frame)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        wp_i = Point(waypoints[i][0], waypoints[i][1])
        temp = point.distance(wp_i)
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    
    return closest_len, closest_index


def check_is_after(waypoint,point, wp_index, margin=0):
    wp = Point(waypoint[wp_index][0:2])
   
    if wp_index==0:
        return True, wp
    
    prev_wp = Point(waypoint[wp_index-1][0:2])
    
    segment = LineString([prev_wp, wp])
    proj_point = project_on_linestring(point, segment)
    dist_wp_prev = wp.distance(prev_wp) - margin
    dist_prev_proj = prev_wp.distance(proj_point)

    return (dist_wp_prev <= dist_prev_proj), proj_point


def check_is_before(waypoint, point, wp_index, margin=0):
    is_before, proj_point = check_is_after(waypoint, point, wp_index, margin)
    return not is_before, proj_point


def project_on_linestring(point, linestring):
    """Project point on a linestring.
    
    .. math::
        cos(alpha) = (v - u).(x - u) / (|x - u|*|v - u|)
        d = cos(alpha)*|x - u| = (v - u).(x - u) / |v - u|
        P(x) = u + d*(v - u)/|v - u|
    
    """
    x = np.array(point.coords[0])

    u = np.array(linestring.coords[0])
    v = np.array(linestring.coords[len(linestring.coords)-1])

    n = v - u
    n /= np.linalg.norm(n, 2)

    p = u + n*np.dot(x - u, n)

    return Point(p)


