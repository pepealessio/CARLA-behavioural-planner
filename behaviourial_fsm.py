#!/usr/bin/env python3

import numpy as np
from fsm.mealy import FSM


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


"""In this file we implemented an FSM for the behavioural planner."""

def get_fsm():
    # ---------- Define conditions for the transition --------------
    def fl_fl_1(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def fl_fl_2(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def fl_dts_1(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and not can_pass
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def fl_dts_2(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and not can_pass
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def fl_dts_3(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3 

    def fl_dts_4(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and not can_pass
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def fl_dts_5(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def fl_dts_6(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and not can_pass
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def dts_fl_1(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def dts_fl_2(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def dts_dts_1(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not stopped)
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) and (not stopped)
        return t1 or t2

    def dts_dts_2(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not stopped)
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) and (not stopped)
        return t1 or t2

    def dts_dts_3(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and (not d['traffic_light_presence']) and (not stopped)
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN) and (not stopped)
        t3 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass and (not stopped)
        return t1 or t2 or t3

    def dts_dts_4(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass) and (not stopped)
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) and (not stopped)
        return t1 or t2

    def dts_dts_5(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and (not d['traffic_light_presence']) and (not stopped)
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN) and (not stopped)
        t3 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass and (not stopped)
        return t1 or t2 or t3

    def dts_dts_6(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass) and (not stopped)
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) and (not stopped)
        return t1 or t2

    def dts_ss_1(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and stopped
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) and stopped
        return t1 or t2

    def dts_ss_2(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and stopped
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) and stopped
        return t1 or t2

    def dts_ss_3(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and (not d['traffic_light_presence']) and stopped
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN) and stopped
        t3 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass and stopped
        return t1 or t2 or t3

    def dts_ss_4(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass) and stopped
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) and stopped
        return t1 or t2

    def dts_ss_5(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and (not d['traffic_light_presence']) and stopped
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN) and stopped
        t3 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass and stopped
        return t1 or t2 or t3

    def dts_ss_6(d):
        stopped = d['closed_loop_speed'] < STOP_THRESHOLD
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass) and stopped
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) and stopped
        return t1 or t2

    def ss_fl_1(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def ss_fl_2(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and (not d['traffic_light_presence'])
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        return t1 or t2

    def ss_ss_1(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW)
        t2 = (not d['pedestrian_presence']) and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) 
        return t1 or t2

    def ss_ss_2(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
        t1 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW)
        t2 = (not d['pedestrian_presence']) and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED) 
        return t1 or t2

    def ss_ss_3(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def ss_ss_4(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass)
        t2 = d['pedestrian_presence'] and (not d['vehicle_presence']) and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2

    def ss_ss_5(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and (not d['traffic_light_presence'])
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_GREEN)
        t3 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and can_pass
        return t1 or t2 or t3

    def ss_ss_6(d):
        if d['traffic_light_presence']:
            traffic_light = d['traffic_lights'][0]
            can_pass = traffic_light[2] / d['closed_loop_speed'] < TRAFFICLIGHT_YELLOW_MIN_TIME
        t1 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_YELLOW) and (not can_pass)
        t2 = d['pedestrian_presence'] and d['vehicle_presence'] and d['traffic_light_presence'] and (traffic_light[1] == TRAFFICLIGHT_RED)
        return t1 or t2
    
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

    def t_M00(d):
        pedestrian = d['pedestrians'][0]
        traffic_light = d['traffic_lights'][0]
        d['goal_index'] = np.min((traffic_light[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_M01(d):
        pedestrian = d['pedestrians'][0]
        traffic_light = d['traffic_lights'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = np.min((traffic_light[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    def t_M10(d):
        pedestrian = d['pedestrians'][0]
        traffic_light = d['traffic_lights'][0]
        d['goal_index'] = np.min((traffic_light[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = False
        d['lead_car_state'] = None

    def t_M11(d):
        pedestrian = d['pedestrians'][0]
        traffic_light = d['traffic_lights'][0]
        lead_vehicle = d['vehicles'][0]
        d['goal_index'] = np.min((traffic_light[0], pedestrian[0]))
        d['goal_state'] = d['waypoints'][d['goal_index']]
        d['goal_state'][2] = 0
        d['follow_lead_vehicle'] = True
        d['lead_car_state'] = [*lead_vehicle[1][0:2], lead_vehicle[2]]

    # --------- Define the FSM -----------------
    fsm = FSM(True)

    # Add states
    fsm.add_state(FOLLOW_LANE)
    fsm.add_state(DECELERATE_TO_STOP)
    fsm.add_state(STAY_STOPPED)
    fsm.set_initial_state(FOLLOW_LANE)

    # Add transition
    fsm.add_transition(FOLLOW_LANE, fl_fl_1, FOLLOW_LANE, action=t_L00)
    fsm.add_transition(FOLLOW_LANE, fl_fl_2, FOLLOW_LANE, action=t_L01)
    fsm.add_transition(FOLLOW_LANE, fl_dts_1, DECELERATE_TO_STOP, action=t_T00)
    fsm.add_transition(FOLLOW_LANE, fl_dts_2, DECELERATE_TO_STOP, action=t_T01)
    fsm.add_transition(FOLLOW_LANE, fl_dts_3, DECELERATE_TO_STOP, action=t_P00)
    fsm.add_transition(FOLLOW_LANE, fl_dts_4, DECELERATE_TO_STOP, action=t_M00)
    fsm.add_transition(FOLLOW_LANE, fl_dts_5, DECELERATE_TO_STOP, action=t_P01)
    fsm.add_transition(FOLLOW_LANE, fl_dts_6, DECELERATE_TO_STOP, action=t_M01)
    fsm.add_transition(DECELERATE_TO_STOP, dts_fl_1, FOLLOW_LANE, action=t_L00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_fl_2, FOLLOW_LANE, action=t_L01)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_1, DECELERATE_TO_STOP, action=t_T00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_2, DECELERATE_TO_STOP, action=t_T01)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_3, DECELERATE_TO_STOP, action=t_P00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_4, DECELERATE_TO_STOP, action=t_M00)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_5, DECELERATE_TO_STOP, action=t_P01)
    fsm.add_transition(DECELERATE_TO_STOP, dts_dts_6, DECELERATE_TO_STOP, action=t_M01)
    fsm.add_transition(DECELERATE_TO_STOP, dts_ss_1, STAY_STOPPED, action=t_T10)
    fsm.add_transition(DECELERATE_TO_STOP, dts_ss_2, STAY_STOPPED, action=t_T11)
    fsm.add_transition(DECELERATE_TO_STOP, dts_ss_3, STAY_STOPPED, action=t_P10)
    fsm.add_transition(DECELERATE_TO_STOP, dts_ss_4, STAY_STOPPED, action=t_M10)
    fsm.add_transition(DECELERATE_TO_STOP, dts_ss_5, STAY_STOPPED, action=t_P11)
    fsm.add_transition(DECELERATE_TO_STOP, dts_ss_6, STAY_STOPPED, action=t_M11)
    fsm.add_transition(STAY_STOPPED, ss_fl_1, FOLLOW_LANE, action=t_L00)
    fsm.add_transition(STAY_STOPPED, ss_fl_2, FOLLOW_LANE, action=t_L01)
    fsm.add_transition(STAY_STOPPED, ss_ss_1, STAY_STOPPED, action=t_T10)
    fsm.add_transition(STAY_STOPPED, ss_ss_2, STAY_STOPPED, action=t_T11)
    fsm.add_transition(STAY_STOPPED, ss_ss_3, STAY_STOPPED, action=t_P10)
    fsm.add_transition(STAY_STOPPED, ss_ss_4, STAY_STOPPED, action=t_M10)
    fsm.add_transition(STAY_STOPPED, ss_ss_5, STAY_STOPPED, action=t_P11)
    fsm.add_transition(STAY_STOPPED, ss_ss_6, STAY_STOPPED, action=t_M11)

    return fsm