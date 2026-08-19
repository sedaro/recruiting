# MODELING & SIMULATION

from random import random

import numpy as np

def propagate_velocity(time_step, position, velocity, other_position, m_other):
    """Propagate the velocity of the agent from `time` to `time + timeStep`."""
    # Use law of gravitation to update velocity
    r_self = np.array([position['x'], position['y'], position['z']])
    v_self = np.array([velocity['x'], velocity['y'], velocity['z']])
    r_other = np.array([other_position['x'], other_position['y'], other_position['z']])

    r = r_self - r_other
    dvdt = -m_other * r / np.linalg.norm(r)**3
    v_self = v_self + dvdt * time_step

    return {'x': v_self[0], 'y': v_self[1], 'z': v_self[2]}

def propagate_position(time_step, position, velocity):
    """Propagate the position of the agent from `time` to `time + timeStep`."""
    # Apply velocity to position
    r_self = np.array([position['x'], position['y'], position['z']])
    v_self = np.array([velocity['x'], velocity['y'], velocity['z']])

    r_self = r_self + v_self * time_step

    return {'x': r_self[0], 'y': r_self[1], 'z': r_self[2]}

def propagate_mass(mass):
    return mass

def identity(arg):
    return arg

def timestep_manager(velocity):
    """Compute the length of the next simulation timeStep for the agent"""
    return 100

def time_manager(time, timeStep):
    """Compute the time for the next simulation step for the agent"""
    return time + timeStep

'''
NOTE: Declare what agents should exist, what functions should be run to update their state, 
    and bind the consumed arguments and produced results to each other.

Query syntax:
- `<variableName>` will do a dictionary lookup of `variableName` in the current state of the agent
   the query is running for.
- prev!(<query>)` will get the value of `query` from the previous step of simulation.
- `agent!(<agentId>)` will get the most recent state produced by `agentId`. Agents run in
   parallel, and a step waits until none of them is behind the time it is stepping from,
   so this is `agentId`'s state at that time or the one step after it.
- `<query>.<name>` will evaluate `query` and then look up `name` in the resulting dictionary.
- `consumed` is always a tuple of the function's arguments, in order. A tuple of one needs its
   trailing comma: `(velocity,)`.
- `produced` is usually one query, and the function returns the value for it. It may also be a
   tuple, and then the function returns one value per element: `(velocity, mass,)` for a
   function whose `return` is `velocity, mass`.
'''

AGENTS = {
    'Body1': [
        {
            'consumed': '''(
                prev!(velocity),
            )''',
            'produced': '''velocity''',
            'function': identity,
        },
        {
            'consumed': '''(
                prev!(timeStep),
                prev!(position),
                velocity,
            )''',
            'produced': '''position''',
            'function': propagate_position,
        },
        {
            'consumed': '''(
                prev!(mass),
            )''',
            'produced': '''mass''',
            'function': propagate_mass,
        },
        {
            'consumed': '''(
                prev!(time),
                timeStep
            )''',
            'produced': '''time''',
            'function': time_manager,
        },
        {
            'consumed': '''(
                velocity,
            )''',
            'produced': '''timeStep''',
            'function': timestep_manager,
        }
    ],
    'Body2': [
        {
            'consumed': '''(
                prev!(timeStep),
                prev!(position),
                prev!(velocity),
                agent!(Body1).position,
                agent!(Body1).mass,
            )''',
            'produced': '''velocity''',
            'function': propagate_velocity,
        },
        {
            'consumed': '''(
                prev!(timeStep),
                prev!(position),
                velocity,
            )''',
            'produced': '''position''',
            'function': propagate_position,
        },
        {
            'consumed': '''(
                prev!(mass),
            )''',
            'produced': '''mass''',
            'function': propagate_mass,
        },
        {
            'consumed': '''(
                prev!(time),
                timeStep
            )''',
            'produced': '''time''',
            'function': time_manager,
        },
        {
            'consumed': '''(
                velocity,
            )''',
            'produced': '''timeStep''',
            'function': timestep_manager,
        }
    ]
}

# NOTE: initial values are set here. we intentionally separate the data from the functions operating on it.
DATA = {
    'Body1': {
        'timeStep': 0.01,
        'time': 0.0,
        'position': {'x': -0.73, 'y': 0, 'z': 0},
        'velocity': {'x': 0, 'y': -0.0015, 'z': 0},
        'mass': 1
    },
    'Body2': {
        'timeStep': 0.01,
        'time': 0.0,
        'position': {'x': 60.34, 'y': 0, 'z': 0},
        'velocity': {'x': 0, 'y': 0.13 , 'z': 0},
        'mass': 0.123
    }
}