#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This file contain the abstract class of the FSM."""

from .exception import FSMException


__author__ 		= "Alessio Pepe"
__copyright__ 	= "Copyright 2021"
__credits__ 	= [] 
__license__ 	= "GPL"
__version__ 	= "0.1.1"
__maintainer__ 	= "Alessio Pepe"
__email__ 		= "pepealessio.ap@gmail.com"
__status__ 		= "Prototype"  					#  "Prototype", "Development", or "Production"


class _AbstractFSM(object):
    """Abstract class who represent a basic FSM.
    
    Attributes:
        _complete (bool): specify if the FSM is complete or if some auto-transition can be inferred.
        _current_state (any): The actual state in the FSM.
        _memory (dict): A dict used to store the input value to the FSM.
        _states (dict): A dict who associate to the FSM states the action to do and a dict, who associate
                        the input to the next state (if the transition is not defined we suppose that input 
                        have as transition itself).
        """

    def __init__(self, complete=False) -> None:
        """This method create the FSM.

        Args:
            complete (bool, default=False): if True, if a transition is not defined, an FSMException will be
                                            rised. If False, the FSM suppose that transition to remain in the 
                                            current state.
        """ 
        self._complete = complete
        self._current_state = None
        self._memory = {}
        self._states = {}  # (state -> (action, (check_input_func -> (next_state, action) ) ) )

    def __len__(self):
        """Return:
            The number of the states of the FSM.
        """
        return len(self._states)

    def __str__(self) -> str:
        return 'FSM:\n' + \
                'Number of states: {}\n'.format(len(self)) + \
                'Current State: {}\n'.format(self._current_state)

    def add_state(self):
        """ABSTRACT METHOD: Add the state in the FSM. The method must have a parameter for the action if is Moore FSM."""
        raise NotImplementedError('This class do not implement this method. Please use one of the class from "mealy.FSM" or "moore.FSM".')

    def _add_state(self, state, action=None) -> None:
        """Insert a new state in the FSM.

        Args:
            state:	the state to add. If is already present nothing happens. A state can be represented by any
                    hashable object
            action:	a function "func(dict) -> None" to execute in this state. Do not set this action if you do 
                    not want to execute actions.
        """
        self._states[state] = (action, {})

    def add_transition(self):
        """ABSTRACT METHOD: Add the transition in the FSM. The method must have a parameter for the action if is Mealy FSM."""
        raise NotImplementedError('This class do not implement this method. Please use one of the class from "mealy.FSM" or "moore.FSM".')

    def _add_transition(self, state, check_input_func, next_state, action=None) -> None:
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
        if state not in self._states:
            raise FSMException('The state must be inserted in the FSM. "{}" is not in the FSM'.format(state)) 
        elif next_state not in self._states:
            raise FSMException('The next state must be inserted in the FSM. "{}" is not in the FSM'.format(next_state)) 

        self._states[state][1][check_input_func] = (next_state, action)

    def remove_state(self, state) -> None:
        """Remove the state and all the transition associated at that state.

        Args:
            state: the state to remove.

        Raise:
            FSMException: if the state does not exists.
        """
        try:
            del self._states[state]
        except KeyError:
            raise FSMException('The state "{}" is not in the FSM.'.format(state))

    def remove_transition(self, state, check_input_func) -> None:
        """Remove the transition for a couple (state, input).

        Args:
            state:	The state for which you want to remove the transition.
            check_input_func:	The input_function for which you want to remove the transition.

        Raise:
            FSMException: if the state or the couple (state, input) does not exists.
        """
        try:
            del self._states[state][1][check_input_func]
        except KeyError as e:
            raise FSMException('The state or the transition are not in the FSM.') 

    def set_initial_state(self, initial_state) -> None:
        """Set the initial state of teh FSM

        Args:
            initial_state:	The inital state of the FSM.

        Raise:
            RuntimeError:	if the initial state was already setted.
            FSMException:	if the initial_state is not in the FSM.
        """
        if self._current_state is not None:
            raise RuntimeError('The initial_state is already setted.')
        if initial_state not in self._states:
            raise FSMException('The state must be inserted in the FSM. "{}" is not in the FSM'.format(initial_state))
        self._current_state = initial_state		

    def get_current_state(self):
        """Getter method for the current state.

        Return:
            current_state:	the current state of the FSM. None if the FSM is not started anymore.
        """
        return self._current_state

    def set_readings(self, **kwargs) -> None:
        """Set the input that the FSM must use to evaluate transition. The previous value was not subscribed if not passed.

        Args:
            **kwargs:	All the variables the fsm need to access. 	
        """
        for k in kwargs:
            self._memory[k] = kwargs[k]

    def get_from_memory(self, arg_name):
        """Getter method for elements in the FSM memory.
        
        Args:
            arg_name (str): The name of the arg to retrive.
        """
        try:
            return self._memory[arg_name]
        except KeyError:
            raise FSMException(f'The arg {arg_name} is not in the FSM memory.')

    def process(self) -> None:
        """ABSTRACT METHOD: Execute action and compute the next state of the FSM."""
        raise NotImplementedError('This class do not implement this method. Please use one of the class from "mealy.FSM" or "moore.FSM".')

    def _process(self, mealy=False, moore=False):
        """Evolve the state of the FSM, using different action in different time if Mealy or Moore.
        
        Args:
            mealy (bool, default=False): if true, the transition action will be executed before the state exchange.
            moore (bool, defalut=False): if true, the state action will be executed before the transition check.
        """
        if self._current_state is None:
            raise FSMException('"initial_state" need to be defined.')

        # Get the action to do and the transition rules of the current state
        action, transition = self._states[self._current_state]

        # Execute the state action if defined, just in the moore fsm.
        if moore and action is not None:
            try:
                action(self._memory)
            except KeyError as e:
                FSMException('Your defined action are accessing a non present element in the readings. This is the exception: {}'.format(e))

        # Compute the next state.
        action = None
        next_state = None

        for check_function in transition:
            if check_function(self._memory):
                next_state, action = transition[check_function]
                break

        # Execute the transition action if defined, just in mealy fsm.
        if mealy and action is not None:
            try:
                action(self._memory)
            except KeyError as e:
                FSMException('Your defined action are accessing a non present element in the readings. This is the exception: {}'.format(e))
        
        if next_state is None:
            if self._complete:
                raise FSMException('The transition for state "{}" and the current input is not defined.'.format(self._current_state))
            else:
                next_state = self._current_state

        self._current_state = next_state
