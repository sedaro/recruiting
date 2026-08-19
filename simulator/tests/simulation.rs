//! Running a simulation: what a step commits, and what agents do to each other.
//!
//! These build agents and ask for steps the way `app/modsim.py` does, so they need only
//! the crate's public API. State managers are Python lambdas because that is what a
//! state manager is.
#![allow(clippy::unwrap_used)]

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use simulator::sim::{Agent, Simulator, StateManager};
use std::ffi::CStr;

/// Adds up whatever it is given: a clock, or a position that moves.
const SUM: &CStr = c_str!("lambda *arguments: sum(arguments)");
const HALF: &CStr = c_str!("lambda *arguments: 0.5");
const TWO: &CStr = c_str!("lambda *arguments: 2.0");
const NAN: &CStr = c_str!("lambda *arguments: float('nan')");
/// Hands back what it read, so that a test can assert on it.
const FIRST: &CStr = c_str!("lambda first: first");

/// A state manager running the function `body` evaluates to.
fn manager(py: Python<'_>, name: &str, body: &CStr, consumed: &str, produced: &str) -> StateManager {
    let function = py.eval(body, None, None).unwrap().unbind();
    StateManager::new(name.to_string(), function, consumed, produced).unwrap()
}

/// The smallest simulation there is: one agent whose only state is a clock.
///
/// Nothing here declares that `timeStep` has to be produced before `time` — the runtime
/// works that out from the queries, which is why every step takes two rounds.
fn clock(py: Python<'_>) -> Agent {
    Agent {
        id: "Body1".to_string(),
        state_managers: vec![
            manager(py, "time_manager", SUM, "(prev!(time), timeStep,)", "time"),
            manager(py, "timestep_manager", HALF, "(prev!(time),)", "timeStep"),
        ],
    }
}

/// A clock whose time step comes from another agent, so that it has to read one.
fn clock_reading(py: Python<'_>, id: &str, other: &str, pace: &CStr) -> Agent {
    Agent {
        id: id.to_string(),
        state_managers: vec![
            manager(py, "time_manager", SUM, "(prev!(time), timeStep,)", "time"),
            manager(
                py,
                "timestep_manager",
                pace,
                &format!("(agent!({other}).time,)"),
                "timeStep",
            ),
        ],
    }
}

/// Every named agent, starting at time zero and holding nothing else.
fn initial<'py>(py: Python<'py>, agents: &[&str]) -> Bound<'py, PyDict> {
    let initial = PyDict::new(py);
    for agent in agents {
        let state = PyDict::new(py);
        state.set_item("time", 0.0).unwrap();
        initial.set_item(agent, state).unwrap();
    }
    initial
}

#[test]
fn steps_an_agent_and_frames_what_it_committed() {
    Python::initialize();
    Python::attach(|py| {
        let mut simulator = Simulator::new(&initial(py, &["Body1"]), vec![clock(py)]).unwrap();
        simulator.run(py, 2).unwrap();

        // The agent's initial state, and then one frame per step.
        let frames: Vec<(f64, f64)> = simulator
            .frames()
            .iter()
            .map(|frame| (frame.start, frame.end))
            .collect();
        assert_eq!(frames, [(0.0, 0.0), (0.0, 0.5), (0.5, 1.0)]);
    });
}

#[test]
fn produces_state_at_the_query_that_named_it() {
    // NOTE: `produced: position.x` writes into a `position` nobody produced, and the
    // step after reads it back through `prev!(position).x`. A query names a place in
    // state, not just a field.
    Python::initialize();
    Python::attach(|py| {
        let agent = Agent {
            id: "Body1".to_string(),
            state_managers: vec![
                manager(py, "time_manager", SUM, "(prev!(time), timeStep,)", "time"),
                manager(py, "timestep_manager", HALF, "(prev!(time),)", "timeStep"),
                manager(py, "move_x", SUM, "(prev!(position).x, timeStep,)", "position.x"),
            ],
        };

        let initial = PyDict::new(py);
        let state = PyDict::new(py);
        let position = PyDict::new(py);
        position.set_item("x", 0.0).unwrap();
        state.set_item("time", 0.0).unwrap();
        state.set_item("position", position).unwrap();
        initial.set_item("Body1", state).unwrap();

        let mut simulator = Simulator::new(&initial, vec![agent]).unwrap();
        simulator.run(py, 3).unwrap();

        let x = simulator
            .frames()
            .last()
            .unwrap()
            .state
            .bind(py)
            .get_item("position")
            .unwrap()
            .unwrap()
            .get_item("x")
            .unwrap()
            .extract::<f64>()
            .unwrap();
        assert_eq!(x, 1.5);
    });
}

#[test]
fn rejects_a_time_that_is_not_finite() {
    // NOTE: A `NaN` time compares false against everything, so it would otherwise pass
    // every check a step makes and produce nothing meaningful.
    Python::initialize();
    Python::attach(|py| {
        let agent = Agent {
            id: "Body1".to_string(),
            state_managers: vec![manager(py, "time_manager", NAN, "(prev!(time),)", "time")],
        };

        let mut simulator = Simulator::new(&initial(py, &["Body1"]), vec![agent]).unwrap();

        assert_eq!(
            simulator.run(py, 1).unwrap_err(),
            "the `time` of agent `Body1` is NaN, and a simulation time has to be a finite number"
        );
    });
}

#[test]
fn a_model_that_raises_reports_the_traceback() {
    // NOTE: A model is Python, so a model bug is a Python exception. It has to arrive as
    // an error naming the state manager and the line it came from, rather than as a
    // panic or an exception with no idea where it happened.
    Python::initialize();
    Python::attach(|py| {
        let agent = Agent {
            id: "Body1".to_string(),
            state_managers: vec![manager(
                py,
                "time_manager",
                c_str!("lambda time: 1 / 0"),
                "(prev!(time),)",
                "time",
            )],
        };

        let mut simulator = Simulator::new(&initial(py, &["Body1"]), vec![agent]).unwrap();
        let error = simulator.run(py, 1).unwrap_err();

        assert!(
            error.starts_with("state manager `time_manager` raised ZeroDivisionError: division by zero"),
            "{error}"
        );
        assert!(error.contains("Traceback (most recent call last)"), "{error}");
    });
}

#[test]
fn reports_state_managers_that_cannot_make_progress() {
    // NOTE: Two managers each consuming what the other produces, with no `prev!` to
    // break the cycle. The error has to say which managers were stuck and on what.
    Python::initialize();
    Python::attach(|py| {
        let agent = Agent {
            id: "Body1".to_string(),
            state_managers: vec![
                manager(py, "time_manager", SUM, "(timeStep,)", "time"),
                manager(py, "timestep_manager", HALF, "(time,)", "timeStep"),
            ],
        };

        let mut simulator = Simulator::new(&initial(py, &["Body1"]), vec![agent]).unwrap();
        let error = simulator.run(py, 1).unwrap_err();

        assert!(error.contains("`time_manager` is waiting on `timeStep`"), "{error}");
        assert!(error.contains("`timestep_manager` is waiting on `time`"), "{error}");
        assert!(error.contains("prev!"), "{error}");
    });
}

#[test]
fn agents_that_read_each_other_both_finish() {
    // NOTE: Neither agent can step until the other has reached the time it is stepping
    // from, so this is the check that waiting lets both through rather than leaving them
    // waiting on each other.
    Python::initialize();
    Python::attach(|py| {
        let agents = vec![
            clock_reading(py, "Body1", "Body2", HALF),
            clock_reading(py, "Body2", "Body1", HALF),
        ];

        let mut simulator = Simulator::new(&initial(py, &["Body1", "Body2"]), agents).unwrap();
        simulator.run(py, 4).unwrap();

        // An initial state each, and then four steps from each of the two agents.
        assert_eq!(simulator.frames().len(), 10);
    });
}

#[test]
fn an_agent_can_read_one_that_has_run_out_of_steps() {
    // NOTE: Given the same number of steps the fast agent ends at time 8 and the slow
    // one at 2, so the fast one's later steps ask for a time the slow one never reaches.
    // Unless an agent says when it has finished, this hangs instead of ending.
    Python::initialize();
    Python::attach(|py| {
        let agents = vec![
            clock_reading(py, "Fast", "Slow", TWO),
            clock_reading(py, "Slow", "Fast", HALF),
        ];

        let mut simulator = Simulator::new(&initial(py, &["Fast", "Slow"]), agents).unwrap();
        simulator.run(py, 4).unwrap();

        assert_eq!(simulator.frames().len(), 10);
        assert_eq!(simulator.frames().last().unwrap().end, 8.0);
    });
}

#[test]
fn reads_another_agent_within_a_step_of_its_own_time() {
    // NOTE: `Fast` reads nobody, so the only thing stopping it running to the end of the
    // simulation while `Slow` is still on its first steps is that a step waits for the
    // agents behind it. What `Slow` sees of it is therefore never more than one of
    // `Fast`'s steps from the time `Slow` is stepping from.
    Python::initialize();
    Python::attach(|py| {
        let racer = Agent {
            id: "Fast".to_string(),
            state_managers: vec![
                manager(py, "time_manager", SUM, "(prev!(time), timeStep,)", "time"),
                manager(py, "timestep_manager", TWO, "(prev!(time),)", "timeStep"),
            ],
        };
        let watcher = Agent {
            id: "Slow".to_string(),
            state_managers: vec![
                manager(py, "time_manager", SUM, "(prev!(time), timeStep,)", "time"),
                manager(py, "timestep_manager", HALF, "(prev!(time),)", "timeStep"),
                manager(py, "watch_fast", FIRST, "(agent!(Fast).time,)", "seen"),
            ],
        };

        let mut simulator = Simulator::new(&initial(py, &["Fast", "Slow"]), vec![racer, watcher]).unwrap();
        simulator.run(py, 10).unwrap();

        for frame in simulator.frames() {
            if frame.agent != "Slow" || frame.start == frame.end {
                continue;
            }
            let seen = frame
                .state
                .bind(py)
                .get_item("seen")
                .unwrap()
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert!(
                seen >= frame.start && seen <= frame.start + 2.0,
                "stepping from {}, `Slow` saw `Fast` at {seen}",
                frame.start
            );
        }
    });
}
