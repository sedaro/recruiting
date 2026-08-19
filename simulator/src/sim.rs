//! The simulation runtime.
//!
//! Running a simulation gives every agent a thread of its own, and no agent may leave
//! the others behind: a step waits until every agent still advancing has reached the
//! time it is stepping from. So the agents stay within a step of each other, and what
//! `agent!` reads of one of them is the state it held at that time or the one just
//! after — never a state from a time the reader has not simulated yet.
//!
//! Models cannot deadlock: whichever agent is furthest behind is waiting for nobody,
//! because waiting is only ever for an agent that has further to go than you do.

use crate::Result;
use crate::interp::Step;
use crate::python::describe;
use crate::query::Query;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, debug_span, error, info, trace};

/// How long an agent waits before looking again at an agent it is ahead of.
const POLL: Duration = Duration::from_micros(100);

/// A state manager: a Python function, plus the queries binding it to agent state.
pub struct StateManager {
    /// The function's `__name__` for error messages and logs.
    name: String,
    function: Py<PyAny>,
    /// The manager's arguments: a tuple query, whose elements are them in order.
    consumed: Query,
    produced: Query,
}

/// Whether a state manager ran, or is still waiting on something.
enum Progress {
    Produced,
    /// The query that came back empty.
    Blocked(Query),
}

impl StateManager {
    /// Build a state manager by parsing the queries binding `function` to agent state.
    pub fn new(name: String, function: Py<PyAny>, consumed: &str, produced: &str) -> Result<Self> {
        let consumed = Query::parse(consumed)?;
        let produced = Query::parse(produced)?;

        Ok(Self {
            name,
            function,
            consumed,
            produced,
        })
    }

    /// Take the manager's step, if everything it consumes is available.
    fn step<'py>(&self, py: Python<'py>, step: &Step<'_, 'py>) -> Result<Progress> {
        let Query::Tuple(queries) = &self.consumed else {
            return Err(format!(
                "the consumed query of state manager `{}` must be a tuple of arguments, \
                 like `(prev!(time), timeStep,)`",
                self.name
            ));
        };

        let mut arguments = Vec::with_capacity(queries.len());
        for query in queries {
            let Some(value) = step.read(query)? else {
                return Ok(Progress::Blocked(query.clone()));
            };
            arguments.push(value);
        }

        let produced = self.call(py, &arguments)?;
        step.write(&self.produced, &produced)?;
        Ok(Progress::Produced)
    }

    /// Call the model, turning whatever it raises into an error that names it.
    fn call<'py>(&self, py: Python<'py>, arguments: &[Bound<'py, PyAny>]) -> Result<Bound<'py, PyAny>> {
        let arguments =
            PyTuple::new(py, arguments).map_err(|err| format!("cannot call state manager `{}`: {err}", self.name))?;

        self.function
            .bind(py)
            .call1(arguments)
            .map_err(|err| format!("state manager `{}` raised {}", self.name, describe(py, &err)))
    }
}

/// An agent: an id and the state managers that advance its state.
pub struct Agent {
    pub id: String,
    pub state_managers: Vec<StateManager>,
}

impl Agent {
    /// Take `steps` steps, one after another.
    fn run(&self, universe: &Universe, steps: usize) -> Result<Vec<Frame>> {
        let span = debug_span!("agent", id = %self.id);
        let _entered = span.enter();

        let mut frames = Vec::with_capacity(steps);
        for _ in 0..steps {
            match Python::attach(|py| self.step(py, universe)) {
                Ok(frame) => frames.push(frame),
                Err(error) => {
                    // Told to the universe as well as returned, so an agent waiting on this
                    // one gives up instead of polling for a step that is not coming.
                    universe.fail(&error);
                    return Err(error);
                }
            }
        }

        universe.done(&self.id)?;
        Ok(frames)
    }

    /// Run the state managers to completion and commit the result.
    fn step(&self, py: Python<'_>, universe: &Universe) -> Result<Frame> {
        let (start, previous) = universe.own(py, &self.id)?;

        let span = debug_span!("step", start);
        let _entered = span.enter();

        // Before anything is read: this is what keeps the agents together, and so what
        // bounds how far from `start` a read of another agent can be.
        universe.wait_for_the_others(py, &self.id, start)?;

        // Managers declare what they read, not when they run, so run whatever can run
        // and go around again. Each pass has to produce something, or the remaining
        // managers are waiting on each other and always will be.
        let step = Step::new(py, universe, previous.into_bound(py));
        let mut pending: Vec<&StateManager> = self.state_managers.iter().collect();
        let mut round: usize = 0;
        while !pending.is_empty() {
            round += 1;
            let mut blocked = Vec::new();
            for manager in &pending {
                let name = &manager.name;
                match manager.step(py, &step) {
                    Ok(Progress::Produced) => {
                        trace!(manager = name, produced = %manager.produced, "ran");
                    }
                    Ok(Progress::Blocked(query)) => {
                        trace!(manager = name, waiting_on = %query, "blocked");
                        blocked.push((*manager, query));
                    }
                    Err(error) => {
                        // Logged as well as returned
                        error!(manager = name, round, error = %error, "a state manager failed");
                        return Err(error);
                    }
                }
            }

            debug!(
                round,
                ran = pending.len() - blocked.len(),
                blocked = blocked.len(),
                waiting = %waiting(&blocked),
                "round"
            );

            if blocked.len() == pending.len() {
                return Err(deadlock(&self.id, &blocked));
            }
            pending = blocked.into_iter().map(|(manager, _)| manager).collect();
        }

        let state = step.finish();
        let end = time_of(&state, &self.id)?;
        debug!(rounds = round, end, fields = state.len(), "stepped");
        if end <= start {
            return Err(format!(
                "agent `{}` stepped from time {start} to {end}, which would not advance the simulation",
                self.id
            ));
        }

        universe.commit(&self.id, end, &state)?;

        Ok(Frame {
            start,
            end,
            agent: self.id.clone(),
            state: state.unbind(),
        })
    }
}

/// One committed step: the state an agent held over `[start, end)`.
///
/// A simulation opens with one frame per agent, holding the state it started in. Those
/// are the only frames that begin and end at the same time, because nothing stepped to
/// them.
pub struct Frame {
    pub start: f64,
    pub end: f64,
    /// The agent whose state this is.
    pub agent: String,
    /// The state that became current at `start`.
    pub state: Py<PyDict>,
}

/// An agent's most recently committed state.
struct Committed {
    time: f64,
    state: Py<PyDict>,
    /// Whether later states are still coming. An agent with no state managers never
    /// had any, and one that has taken all its steps has no more.
    advancing: bool,
}

struct Shared {
    agents: HashMap<String, Committed>,
    /// The first error an agent hit, so that agents waiting on it stop waiting.
    failed: Option<String>,
}

/// Every agent's most recently committed state, and the rule for reading it.
pub struct Universe {
    shared: Mutex<Shared>,
}

impl Universe {
    fn new(initial: &Bound<'_, PyDict>, advancing: &[&str]) -> Result<Self> {
        let mut agents = HashMap::new();
        for (id, state) in initial {
            let id: String = id.extract().map_err(|_| format!("`{id}` is not an agent id"))?;
            let state = state
                .cast_into::<PyDict>()
                .map_err(|_| format!("the state of agent `{id}` is not a dictionary"))?;
            let time = time_of(&state, &id)?;
            agents.insert(
                id.clone(),
                Committed {
                    time,
                    state: state.unbind(),
                    advancing: advancing.contains(&id.as_str()),
                },
            );
        }

        Ok(Self {
            shared: Mutex::new(Shared { agents, failed: None }),
        })
    }

    /// A universe with no agents in it, for the interpreter's tests.
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            shared: Mutex::new(Shared {
                agents: HashMap::new(),
                failed: None,
            }),
        }
    }

    /// Read another agent's most recently committed state.
    pub fn read(&self, py: Python<'_>, agent: &str) -> Result<Py<PyDict>> {
        let shared = self.lock()?;
        let committed = shared
            .agents
            .get(agent)
            .ok_or_else(|| format!("there is no agent `{agent}` in this simulation"))?;
        Ok(committed.state.clone_ref(py))
    }

    /// Wait until no agent that is still advancing is behind `time`.
    fn wait_for_the_others(&self, py: Python<'_>, agent: &str, time: f64) -> Result<()> {
        loop {
            {
                let shared = self.lock()?;
                if let Some(error) = &shared.failed {
                    return Err(format!("another agent failed first: {error}"));
                }
                let behind = shared
                    .agents
                    .iter()
                    .any(|(id, committed)| id != agent && committed.advancing && committed.time < time);
                if !behind {
                    return Ok(());
                }
            }

            trace!(agent, time, "waiting for the others");
            // Avoid holding the GIL while sleeping, so that the agents being waited on can
            // run and advance.
            py.detach(|| thread::sleep(POLL));
        }
    }

    /// Read every agent's committed time and state, for the frames a simulation opens with.
    fn committed(&self, py: Python<'_>) -> Result<Vec<(String, f64, Py<PyDict>)>> {
        let shared = self.lock()?;
        Ok(shared
            .agents
            .iter()
            .map(|(id, committed)| (id.clone(), committed.time, committed.state.clone_ref(py)))
            .collect())
    }

    /// Read an agent's own time and state.
    fn own(&self, py: Python<'_>, agent: &str) -> Result<(f64, Py<PyDict>)> {
        let shared = self.lock()?;
        let committed = shared
            .agents
            .get(agent)
            .ok_or_else(|| format!("agent `{agent}` has no initial state"))?;
        Ok((committed.time, committed.state.clone_ref(py)))
    }

    /// Commit a new state for an agent at a given time.
    fn commit(&self, agent: &str, time: f64, state: &Bound<'_, PyDict>) -> Result<()> {
        let mut shared = self.lock()?;
        shared.agents.insert(
            agent.to_string(),
            Committed {
                time,
                state: state.clone().unbind(),
                advancing: true,
            },
        );
        Ok(())
    }

    /// Restart the simulation for the given agents.
    fn restart(&self, agents: &[Agent]) -> Result<()> {
        let mut shared = self.lock()?;
        for agent in agents {
            if let Some(committed) = shared.agents.get_mut(&agent.id) {
                committed.advancing = true;
            }
        }
        Ok(())
    }

    /// Say an agent will not advance further, so reads of it answer with its last
    /// state instead of waiting.
    fn done(&self, agent: &str) -> Result<()> {
        let mut shared = self.lock()?;
        if let Some(committed) = shared.agents.get_mut(agent) {
            committed.advancing = false;
        }
        Ok(())
    }

    /// Mark the simulation as failed with the given error message.
    fn fail(&self, error: &str) {
        if let Ok(mut shared) = self.shared.lock() {
            shared.failed.get_or_insert_with(|| error.to_string());
        }
    }

    /// Lock the shared simulation state, returning an error if it is poisoned.
    fn lock(&self) -> Result<MutexGuard<'_, Shared>> {
        self.shared
            .lock()
            .map_err(|_| "the simulation state is poisoned, because an agent panicked".to_string())
    }
}

pub struct Simulator {
    agents: Vec<Agent>,
    universe: Universe,
    frames: Vec<Frame>,
}

impl Simulator {
    /// Build a simulator over an initial universe.
    ///
    /// `initial` maps agent id to that agent's starting state, which must include a
    /// numeric `time`. It may name agents that have no state managers: they never
    /// advance, but the rest of the simulation can still read them through `agent!`.
    pub fn new(initial: &Bound<'_, PyDict>, agents: Vec<Agent>) -> Result<Self> {
        let started = Instant::now();

        let advancing: Vec<&str> = agents.iter().map(|agent| agent.id.as_str()).collect();
        let universe = Universe::new(initial, &advancing)?;

        // The initial states, as a frame each. Taken from the universe rather than from
        // `initial` so that they are the states it checked, and so that an agent with no
        // state managers is framed too: it never steps, but it is part of the run.
        let mut frames: Vec<Frame> = Python::attach(|py| universe.committed(py))?
            .into_iter()
            .map(|(agent, time, state)| Frame {
                start: time,
                end: time,
                agent,
                state,
            })
            .collect();
        // A `HashMap` hands them over in no particular order, and a frame is easier to
        // find in the output when the earliest one comes first.
        frames.sort_by(|left, right| {
            left.end
                .total_cmp(&right.end)
                .then_with(|| left.agent.cmp(&right.agent))
        });

        // Counted before the agents are handed over, because that moves them.
        let managers: usize = agents.iter().map(|agent| agent.state_managers.len()).sum();
        info!(agents = agents.len(), managers, elapsed = ?started.elapsed(), "built");

        Ok(Self {
            agents,
            universe,
            frames,
        })
    }

    /// Take `steps` steps with every agent, each running as fast as it can.
    pub fn run(&mut self, py: Python<'_>, steps: usize) -> Result<()> {
        let started = Instant::now();
        self.universe.restart(&self.agents)?;

        let agents = &self.agents;
        let universe = &self.universe;
        // Without this the worker threads could never attach to the interpreter, and
        // the first state manager call would wait on a GIL this thread still holds.
        let produced: Vec<Result<Vec<Frame>>> = py.detach(|| {
            thread::scope(|scope| {
                let running: Vec<_> = agents
                    .iter()
                    .map(|agent| scope.spawn(move || agent.run(universe, steps)))
                    .collect();

                running
                    .into_iter()
                    .map(|agent| {
                        agent
                            .join()
                            .unwrap_or_else(|_| Err("an agent's thread panicked".to_string()))
                    })
                    .collect()
            })
        });

        for frames in produced {
            self.frames.extend(frames?);
        }
        // Frames arrived interleaved so we sort them by their end time.
        self.frames.sort_by(|left, right| left.end.total_cmp(&right.end));

        info!(
            agents = self.agents.len(),
            steps,
            frames = self.frames.len(),
            elapsed = ?started.elapsed(),
            "simulated"
        );
        Ok(())
    }

    /// Every frame committed so far, in the order they end.
    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }
}

/// The `time` in a state, which has to be there and has to be finite.
///
/// `NaN` compares false against everything, so without the check a step could be seen
/// neither to move forward nor to stand still, and the simulation would run every step
/// it was asked for having produced nothing meaningful.
fn time_of(state: &Bound<'_, PyDict>, agent: &str) -> Result<f64> {
    let time = state
        .get_item("time")
        .map_err(|err| format!("cannot read the `time` of agent `{agent}`: {err}"))?
        .ok_or_else(|| format!("agent `{agent}` has no `time`, so the simulation cannot advance"))?
        .extract::<f64>()
        .map_err(|_| format!("the `time` of agent `{agent}` is not a number"))?;

    if time.is_finite() {
        Ok(time)
    } else {
        Err(format!(
            "the `time` of agent `{agent}` is {time}, and a simulation time has to be a \
             finite number"
        ))
    }
}

/// The state managers a round left blocked, for a log line. `tracing` only evaluates
/// a field when something is listening, so at the default level this never runs.
fn waiting(blocked: &[(&StateManager, Query)]) -> String {
    if blocked.is_empty() {
        return "-".to_string();
    }
    blocked
        .iter()
        .map(|(manager, _)| manager.name.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// State managers that can no longer make progress, and what each waits for — which
/// is where the cycle is.
fn deadlock(agent: &str, blocked: &[(&StateManager, Query)]) -> String {
    let waiting: Vec<String> = blocked
        .iter()
        .map(|(manager, query)| format!("`{}` is waiting on `{query}`", manager.name))
        .collect();

    format!(
        "no progress made while evaluating the state managers of agent `{agent}`: {}. \
         Consuming `prev!(..)` of a value breaks a cycle like this one.",
        waiting.join(", ")
    )
}
