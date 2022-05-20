#!/usr/bin/env python3

import numpy as np
from fsm.mealy import FSM


# State machine states
FOLLOW_LANE = 'Follow Lane'
DECELERATE_TO_STOP = 'Decelerate to Stop'

# Stop speed threshold
STOP_THRESHOLD = 1.5  # m/s

# Enumerate Trafficlight State
TRAFFICLIGHT_GREEN = 0
TRAFFICLIGHT_YELLOW = 1
TRAFFICLIGHT_RED = 2
TRAFFICLIGHT_YELLOW_MIN_TIME = 1  # sec


"""In this file we implemented an FSM for the behavioural planner."""


def get_fsm():
    """Return an istance of the FSM as described in the Paragraph 2.2 of the report.
    """
    # --------- Defines the action functions for the fsm -------------
    def t_L00(d):
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_L01(d):
        lead_vehicle = d['vehicles'][0]
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_V00(d):
        lead_vehicle = d['vehicles'][0]
        d['goal_state'] = d['waypoints'][lead_vehicle[0]]
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_T00(d):
        traffic_light = d['traffic_lights'][0]
        d['goal_index'] = traffic_light[0]
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_T01(d):
        traffic_light = d['traffic_lights'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = traffic_light[0]
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_T10(d):
        traffic_light = d['traffic_lights'][0]
        d['goal_index'] = traffic_light[0]
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_T11(d):
        traffic_light = d['traffic_lights'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = traffic_light[0]
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]
    
    def t_P00(d):
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = pedestrian[0]
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_P01(d):
        pedestrian = d['pedestrians'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = pedestrian[0]
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_P10(d):
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = pedestrian[0]
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_P11(d):
        pedestrian = d['pedestrians'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = pedestrian[0]
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_Mtp00(d):
        pedestrian = d['pedestrians'][0]
        traffic_light = d['traffic_lights'][0]
        d['goal_index'] = np.min((traffic_light[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_Mtp01(d):
        pedestrian = d['pedestrians'][0]
        traffic_light = d['traffic_lights'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = np.min((traffic_light[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_Mtp10(d):
        pedestrian = d['pedestrians'][0]
        traffic_light = d['traffic_lights'][0]
        d['goal_index'] = np.min((traffic_light[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_Mtp11(d):
        pedestrian = d['pedestrians'][0]
        traffic_light = d['traffic_lights'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = np.min((traffic_light[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_Mtv00(d):
        lead_vehicle = d['vehicles'][0]
        traffic_light = d['traffic_lights'][0]
        d['goal_index'] = np.min((traffic_light[0], lead_vehicle[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_Mtv01(d):
        traffic_light = d['traffic_lights'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = np.min((traffic_light[0], lead_vehicle[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_Mtv10(d):
        traffic_light = d['traffic_lights'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = np.min((traffic_light[0], lead_vehicle[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_Mtv11(d):
        traffic_light = d['traffic_lights'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = np.min((traffic_light[0], lead_vehicle[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_Mvp00(d):
        lead_vehicle = d['vehicles'][0]
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = np.min((pedestrian[0], lead_vehicle[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_Mvp01(d):
        lead_vehicle = d['vehicles'][0]
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = np.min((pedestrian[0], lead_vehicle[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_Mvp10(d):
        lead_vehicle = d['vehicles'][0]
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = np.min((pedestrian[0], lead_vehicle[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_Mvp11(d):
        lead_vehicle = d['vehicles'][0]
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = np.min((pedestrian[0], lead_vehicle[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_M00(d):
        lead_vehicle = d['vehicles'][0]
        traffic_light = d['traffic_lights'][0]
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = np.min((traffic_light[0], lead_vehicle[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_M01(d):
        lead_vehicle = d['vehicles'][0]
        traffic_light = d['traffic_lights'][0]
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = np.min((traffic_light[0], lead_vehicle[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_M10(d):
        lead_vehicle = d['vehicles'][0]
        traffic_light = d['traffic_lights'][0]
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = np.min((traffic_light[0], lead_vehicle[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_M11(d):
        lead_vehicle = d['vehicles'][0]
        traffic_light = d['traffic_lights'][0]
        pedestrian = d['pedestrians'][0]
        d['goal_index'] = np.min((traffic_light[0], lead_vehicle[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]
    
    # ---------- Define conditions for the transition --------------
    def fl_fl_L00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def fl_fl_L01(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not lead_stopped) and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def fl_dts_V00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and lead_stopped and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = (not d['pedestrian_presence']) and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def fl_dts_T00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass)
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2
    
    def fl_dts_T01(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass)
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def fl_dts_P00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def fl_dts_P01(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and (not lead_stopped) and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = d['pedestrian_presence'] and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def fl_dts_Mtp00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass)
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def fl_dts_Mtp01(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass)
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def fl_dts_Mtv00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass)
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def fl_dts_Mvp00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and lead_stopped and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = d['pedestrian_presence'] and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def fl_dts_M00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = (d['closed_loop_speed'] != 0) and (traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME)
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass)
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def dts_fl_L00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def dts_fl_L01(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not lead_stopped) and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def dts_dts_V00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and lead_stopped and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def dts_dts_T00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW)
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def dts_dts_T01(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW)
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def dts_dts_P00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def dts_dts_P01(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and (not lead_stopped) and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def dts_dts_Mtp00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW)
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def dts_dts_Mtp01(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW)
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and (not lead_stopped) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def dts_dts_Mtv00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW)
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def dts_dts_Mvp00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and lead_stopped and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def dts_dts_M00(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        if d['vehicle_presence']:
            lead_stopped = d['vehicles'][0][2] <= STOP_THRESHOLD
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW)
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and lead_stopped and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    # --------- Define the FSM -----------------
    fsm = FSM(True)

    # Add states
    fsm.add_state(FOLLOW_LANE)
    fsm.add_state(DECELERATE_TO_STOP)
    fsm.set_initial_state(FOLLOW_LANE)

    # Add transition
    fsm.add_transition(FOLLOW_LANE, fl_fl_L00, FOLLOW_LANE, action=t_L00)
    fsm.add_transition(FOLLOW_LANE, fl_fl_L01, FOLLOW_LANE, action=t_L01)

    fsm.add_transition(FOLLOW_LANE, fl_dts_V00, DECELERATE_TO_STOP, action=t_V00)
    fsm.add_transition(FOLLOW_LANE, fl_dts_T00, DECELERATE_TO_STOP, action=t_T00)
    fsm.add_transition(FOLLOW_LANE, fl_dts_T01, DECELERATE_TO_STOP, action=t_T01)
    fsm.add_transition(FOLLOW_LANE, fl_dts_P00, DECELERATE_TO_STOP, action=t_P00)
    fsm.add_transition(FOLLOW_LANE, fl_dts_P01, DECELERATE_TO_STOP, action=t_P01)
    fsm.add_transition(FOLLOW_LANE, fl_dts_Mtp00, DECELERATE_TO_STOP, action=t_Mtp00)
    fsm.add_transition(FOLLOW_LANE, fl_dts_Mtp01, DECELERATE_TO_STOP, action=t_Mtp01)
    fsm.add_transition(FOLLOW_LANE, fl_dts_Mtv00, DECELERATE_TO_STOP, action=t_Mtv00)
    fsm.add_transition(FOLLOW_LANE, fl_dts_Mvp00, DECELERATE_TO_STOP, action=t_Mvp00)
    fsm.add_transition(FOLLOW_LANE, fl_dts_M00, DECELERATE_TO_STOP, action=t_M00)

    fsm.add_transition(DECELERATE_TO_STOP, dts_fl_L00, FOLLOW_LANE, action=t_L00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_fl_L01, FOLLOW_LANE, action=t_L01)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_V00, DECELERATE_TO_STOP, action=t_V00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_T00, DECELERATE_TO_STOP, action=t_T00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_T01, DECELERATE_TO_STOP, action=t_T01)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_P00, DECELERATE_TO_STOP, action=t_P00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_P01, DECELERATE_TO_STOP, action=t_P01)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_Mtp00, DECELERATE_TO_STOP, action=t_Mtp00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_Mtp01, DECELERATE_TO_STOP, action=t_Mtp01)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_Mtv00, DECELERATE_TO_STOP, action=t_Mtv00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_Mvp00, DECELERATE_TO_STOP, action=t_Mvp00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_M00, DECELERATE_TO_STOP, action=t_M00)

    return fsm
