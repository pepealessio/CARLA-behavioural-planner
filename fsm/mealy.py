#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This file contain a Mealy FSM implementation. 

Note:
	The FSM state can be represented with any hashable object.
	The transition are represented with functions defined by the user who check if the requirements are achieved.
	The input will be passed in input as a dictionary to the FSM.
	The actions for the various states can be executeb by the FSM if a function will be passed in the transition definition.
	The process method evaluate the next state and do the action for the current transition.
"""

from ._fsm import _AbstractFSM
from .exception import FSMException


__author__ 		= "Alessio Pepe"
__copyright__ 	= "Copyright 2021"
__credits__ 	= [] 
__license__ 	= "GPL"
__version__ 	= "0.1.2"
__maintainer__ 	= "Alessio Pepe"
__email__ 		= "pepealessio.ap@gmail.com"
__status__ 		= "Prototype"  					#  "Prototype", "Development", or "Production"


class FSM(_AbstractFSM):
    """Implementation of a Mealy FSM.

    Attributes:
        _complete (bool): specify if the FSM is complete or if some auto-transition can be inferred.
        _current_state (any): The actual state in the FSM.
        _memory (dict): A dict used to store the input value to the FSM.
        _states (dict): A dict who associate to the FSM states the action to do and a dict, who associate
                        the input to the next state (if the transition is not defined we suppose that input 
                        have as transition itself).
    
    Examples:
        >>> from fsm.mealy import FSM
        >>> fsm = FSM()
        >>> fsm.add_state('STATE_0')
        >>> fsm.add_state('STATE_1')
        >>> fsm.add_transition('STATE_0', lambda d: d['v'] >= 0.5, 'STATE_1', action=lambda d: print('Transition from 0 to 1'))
        >>> fsm.add_transition('STATE_1', lambda d: d['v'] >= 0.5, 'STATE_0', action=lambda d: print('Transition from 1 to 0')) 
        >>> fsm.set_initial_state('STATE_0')
        >>> 
        >>> fsm.set_readings(v=0.11)
        >>> fsm.process()
        >>> fsm.get_current_state()
        'STATE_0'
        >>> fsm.set_readings(v=0.89)      
        >>> fsm.process()                 
        Transition from 0 to 1
        >>> fsm.get_current_state()
        'STATE_1'
    """

    def add_state(self, state) -> None:
        """Insert a new state in the FSM.

        Args:
            state:	the state to add. If is already present nothing happens. A state can be represented by any
                    hashable object
        """
        self._add_state(state)

    def add_transition(self, state, check_input_func, next_state, action=None) -> None:
        """Associate at wthe couple (state, input) the next state.

        Args:
            state: 	The transition starting state.
            check_input_func: 	A function "func(dict) -> bool" who check the input for the current state. This
                                fuction must return True if the condition is keep, otherwise False.
            next_state:	The next state associated to the transition.
            action:	a function "func(dict) -> None" to execute in this state. Do not set this action if you do 
                    not want to execute actions.

        Raise:
            FSMException: if the state does not exists; if the next state was not added in the FSM.
        """
        self._add_transition(state, check_input_func, next_state, action)	
        
    def process(self) -> None:
        """Received the current input, execute the action if defined and compute the new state.

        Raise:
            RuntimeError:	if the initial state was not setted.
            FSMException:	if the transition does not exist and the FSM was setted in complete mode.
        """
        self._process(mealy=True)
