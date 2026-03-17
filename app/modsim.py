import numpy as np

agents = {}


def statemanager(agent, consumed, produced):
    """A decorator to register a function as a state manager for an agent.

    Use this to declare what agents should exist, what functions should be run to update their state,
    and to bind the consumed arguments and produced results to each other.

    Query syntax:
    - `<variableName>` will do a dictionary lookup of `variableName` in the current state of the agent
    the query is running for.
    - prev!(<query>)` will get the value of `query` from the previous step of simulation.
    - `agent!(<agentId>)` will get the most recent state produced by `agentId`.
    - `<query>.<name>` will evaluate `query` and then look up `name` in the resulting dictionary.
    """

    def decorator(func):
        if agent not in agents:
            agents[agent] = []
        agents[agent].append({"function": func, "consumed": consumed, "produced": produced})
        return func

    return decorator


def runge_kutta4(fcn, t0, x0, dt):
    """Runge Kutta 4-th order fix-step integrator."""
    k1 = fcn(t0, x0)
    k2 = fcn(t0 + dt / 2, x0 + dt * k1 / 2)
    k3 = fcn(t0 + dt / 2, x0 + dt * k2 / 2)
    k4 = fcn(t0 + dt, x0 + dt * k3)
    return x0 + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


#
# Body 2
#
@statemanager(
    agent="Body2",
    consumed="""(
        prev!(timeStep),
        prev!(position),
        prev!(velocity),
        agent!(Body1).position,
        agent!(Body1).mass,
    )""",
    produced="state",
)
def prop_body2_orbit(time_step, position, velocity, other_position, m_other):
    """Propagate the orbit of the agent from `time` to `time + timeStep`."""
    # Use law of gravitation to update velocity and position
    r_self = np.array([position["x"], position["y"], position["z"]])
    v_self = np.array([velocity["x"], velocity["y"], velocity["z"]])
    r_other = np.array([other_position["x"], other_position["y"], other_position["z"]])

    def fcn(_, y):
        r = y[0:3] - r_other
        dv_dt = -m_other * r / np.linalg.norm(r) ** 3
        dr_dt = y[3:6]
        return np.hstack((dr_dt, dv_dt))

    y0 = np.hstack((r_self, v_self))
    y1 = runge_kutta4(fcn, 0, y0, time_step)

    return y1.tolist()


@statemanager(agent="Body2", consumed="(state,)", produced="position")
def set_body2_position(state):
    return {"x": state[0], "y": state[1], "z": state[2]}


@statemanager(agent="Body2", consumed="(state,)", produced="velocity")
def set_body2_velocity(state):
    return {"x": state[3], "y": state[4], "z": state[5]}


@statemanager(agent="Body2", consumed="(prev!(mass), )", produced="mass")
def prop_body2_mass(mass):
    return mass


@statemanager(agent="Body2", consumed="(agent!(Body1).position, position)", produced="timeStep")
def sensor(target_position, self_position):
    r_self = np.array([self_position["x"], self_position["y"], self_position["z"]])
    r_target = np.array([target_position["x"], target_position["y"], target_position["z"]])
    distance = np.linalg.norm(r_self - r_target)
    return distance


@statemanager(agent="Body2", consumed="(velocity,)", produced="timeStep")
def body2_timestep(_):
    """Compute the length of the next simulation timeStep for the agent"""
    return 100.0


@statemanager("Body2", consumed="(prev!(time), timeStep)", produced="time")
def prop_body2_time(time, step):
    """Compute the time for the next simulation step for the agent"""
    return time + step


#
# Body 1
#


@statemanager(agent="Body1", consumed="(prev!(velocity),)", produced="velocity")
def prop_body1_velocity(arg):
    return arg


@statemanager(agent="Body1", consumed="(prev!(position), )", produced="position")
def prop_body1_position(arg):
    return arg


@statemanager(agent="Body1", consumed="(prev!(mass), )", produced="mass")
def prop_body1_mass(arg):
    return arg


@statemanager(agent="Body1", consumed="(velocity,)", produced="timeStep")
def body1_timestep(_):
    """Compute the length of the next simulation timeStep for the agent"""
    return 100.0


@statemanager("Body1", consumed="(prev!(time), timeStep)", produced="time")
def prop_body1_time(time, timeStep):
    """Compute the time for the next simulation step for the agent"""
    return time + timeStep


#
# Initial state values
# NOTE: we intentionally separate the data from the functions operating on it.
#
data = {
    "Body1": {
        "timeStep": 0.01,
        "time": 0.0,
        "position": {"x": -0.73, "y": 0, "z": 0},
        "velocity": {"x": 0, "y": -0.0015, "z": 0},
        "mass": 1,
    },
    "Body2": {
        "timeStep": 0.01,
        "time": 0.0,
        "position": {"x": 60.34, "y": 0, "z": 0},
        "velocity": {"x": 0, "y": 0.13, "z": 0},
        "mass": 0.123,
    },
}
