//! The Python boundary: the `simulator` extension module.

use crate::sim;
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};
use std::fmt::Write as _;

/// How many steps [`Simulator::run`] takes with each agent if it is not told.
const DEFAULT_STEPS: usize = 1000;

/// The simulator, as `app/` sees it.
#[pyclass(module = "simulator")]
pub struct Simulator {
    inner: sim::Simulator,
}

#[pymethods]
impl Simulator {
    /// Build a simulator.
    ///
    /// - `initial` is `{agent_id: {field: value}}`, and each state must include a numeric `time`.
    /// - `agents` is `{agent_id: [{"consumed": str, "produced": str, "function": callable}]}`.
    ///
    /// An agent may appear in `initial` and not in `agents`: it never advances,
    /// but the other agents can read it with `agent!`.
    #[new]
    fn new(initial: &Bound<'_, PyDict>, agents: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut definitions = Vec::new();
        for (id, managers) in agents {
            let id: String = id.extract()?;
            definitions.push(sim::Agent {
                state_managers: state_managers(&id, &managers)?,
                id,
            });
        }

        Ok(Self {
            inner: sim::Simulator::new(initial, definitions).map_err(simulation_error)?,
        })
    }

    /// Take `steps` steps with every agent and return every frame, including the
    /// initial states, as `[[start, end, agent_id, state], ..]`. Each frame is the state
    /// one agent held over `[start, end)`.
    #[pyo3(signature = (steps = DEFAULT_STEPS))]
    fn run<'py>(&mut self, py: Python<'py>, steps: usize) -> PyResult<Bound<'py, PyList>> {
        self.inner.run(py, steps).map_err(simulation_error)?;

        let frames = PyList::empty(py);
        for frame in self.inner.frames() {
            // The agent is named in the frame rather than keyed by it: a frame is one
            // agent's state, and a reader that wants them grouped can group them.
            frames.append(PyTuple::new(
                py,
                [
                    PyFloat::new(py, frame.start).into_any(),
                    PyFloat::new(py, frame.end).into_any(),
                    PyString::new(py, &frame.agent).into_any(),
                    frame.state.bind(py).clone().into_any(),
                ],
            )?)?;
        }

        Ok(frames)
    }
}

/// Read one agent's state manager definitions.
fn state_managers(agent: &str, definitions: &Bound<'_, PyAny>) -> PyResult<Vec<sim::StateManager>> {
    let mut managers = Vec::new();
    for definition in definitions.try_iter()? {
        let definition = definition?;
        let consumed: String = field(agent, &definition, "consumed")?.extract()?;
        let produced: String = field(agent, &definition, "produced")?.extract()?;
        let function = field(agent, &definition, "function")?;

        let name = function
            .getattr("__name__")
            .and_then(|name| name.extract::<String>())
            .unwrap_or_else(|_| function.to_string());

        managers.push(sim::StateManager::new(name, function.unbind(), &consumed, &produced).map_err(simulation_error)?);
    }
    Ok(managers)
}

fn field<'py>(agent: &str, definition: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    definition
        .get_item(name)
        .map_err(|_| PyKeyError::new_err(format!("a state manager of agent `{agent}` has no `{name}`")))
}

/// Describe a Python exception the way the interpreter would.
pub(crate) fn describe(py: Python<'_>, error: &PyErr) -> String {
    let mut described = error.to_string();
    if let Some(traceback) = error.traceback(py)
        && let Ok(formatted) = traceback.format()
    {
        let _ = write!(described, "\n{}", formatted.trim_end());
    }
    described
}

/// Simulation errors are strings; they reach Python as `RuntimeError`.
fn simulation_error(error: String) -> PyErr {
    PyRuntimeError::new_err(error)
}

/// Send the simulator's logs to stderr, at the level `$LOG_LEVEL` asks for.
#[pyfunction]
fn init_tracing() {
    crate::trace::init();
}

#[pymodule]
fn simulator(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Simulator>()?;
    module.add_function(wrap_pyfunction!(init_tracing, module)?)?;
    Ok(())
}
