# The nano query language

In Sedaro Nano, the simulation is made up of *agents* that have state and *managers* that define how that state evolves over time. The managers are Python functions that consume arguments from the simulator and produce a result value. The agent specifies what state should be passed into those arguments and where the result should be stored through a simple query language.

The queries are parsed and interpreted by the simulation runtime in `simulator/`, which is Rust; the managers themselves stay here in Python. A query that will not parse, a model that cannot make progress, and a manager that raises all reach you as a `RuntimeError`, and when a manager is at fault it is named. One that raised inside its own body brings its Python traceback with it, pointing at the line in this file. A missing `consumed`, `produced` or `function` key is a `KeyError` instead.

At the starting point for the exercise, the velocity of the first agent is bound to the `identity` statemanager by the following block:
```python
        {
            'consumed': '''(
                prev!(velocity),
            )''',
            'produced': '''velocity''',
            'function': identity,
        }
```

We want to bind it to the `propagate_velocity` statemanager, which takes into account the gravitational force of the other agent. First, we'll replace `identity` with `propagate_velocity`, but this is insufficient:

```text
app  | RuntimeError: state manager `propagate_velocity` raised TypeError: propagate_velocity() missing 4 required positional arguments: 'position', 'velocity', 'other_position', and 'm_other'
```

We'll need to extend the query to include the remaining arguments. We can get our other state from the same agent:
```python
            'consumed': '''(
                timeStep,
                position,
                prev!(velocity),
            )''',
```

This results in a deadlock, since the current position depends on the current velocity, which depends on the current position. The runtime reports every manager that is stuck, and what each one is waiting for:

```text
app  | RuntimeError: no progress made while evaluating the state managers of agent `Body1`: `propagate_velocity` is waiting on `timeStep`, `propagate_position` is waiting on `velocity`, `time_manager` is waiting on `timeStep`, `timestep_manager` is waiting on `velocity`. Consuming `prev!(..)` of a value breaks a cycle like this one.
```

To watch that happen rather than read it backwards from the error, run with `LOG_LEVEL=debug`. The runtime logs a line per round of state managers, naming what ran and what is still blocked, which for the deadlock above looks something like:

```text
DEBUG step{agent=Body1 start=0.0}: round round=1 ran=1 blocked=4 waiting=propagate_velocity propagate_position time_manager timestep_manager
DEBUG step{agent=Body1 start=0.0}: round round=2 ran=0 blocked=4 waiting=propagate_velocity propagate_position time_manager timestep_manager
```

A second round that gets nothing further done is the point at which the runtime gives up. Worth knowing before you go hunting for a cycle: a misspelled field reports as a deadlock too, because a field that does not exist and a field that has not been produced yet are indistinguishable to the runtime. `prev!(velocty)` becomes a manager waiting forever on `prev!(velocty)`, so the names in that `waiting` list are worth reading carefully.

We can avoid this by wrapping our consumed queries in `prev!()`, which indicates that we want to read the value as computed in the most recent previous simulation step, rather than the value computed in the current simulation step.

```python
            'consumed': '''(
                prev!(timeStep),
                prev!(position),
                prev!(velocity),
            )''',
```

Which leaves the two arguments that describe the *other* agent:

```text
app  | RuntimeError: state manager `propagate_velocity` raised TypeError: propagate_velocity() missing 2 required positional arguments: 'other_position' and 'm_other'
```

Finally, we need to add the data we're reading from the other agent. This also needs special handling - we'll use `agent!(Body2)` to indicate which agent we want to read from. Every agent runs in parallel, and a step waits until no agent is behind the time we're stepping from, so this reads the most recently computed state of `Body2` as of that time or the step just after it. Two agents reading each other is fine and will never deadlock: whichever one is furthest behind in time is always free to run.

```python
            'consumed': '''(
                prev!(timeStep),
                prev!(position),
                prev!(velocity),
                agent!(Body2).position,
                agent!(Body2).mass,
            )''',
```
