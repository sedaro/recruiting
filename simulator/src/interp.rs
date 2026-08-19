//! The query interpreter.
//!
//! One [`Step`] is one agent taking one step: it reads what a state manager consumes
//! out of simulation state, and collects what the manager produces into the state that
//! step will commit. State is the Python dictionaries the models build, held as they
//! are.
//!
//! A read returns `Ok(None)` when the value has not been produced *yet*, which is the
//! whole scheduling mechanism within a step — [`crate::sim`] retries the managers that
//! came back empty. Nothing here blocks: a step waits for the other agents before it
//! runs a manager, so by the time `agent!(..)` is read they are all far enough along.
//! See [`crate::sim::Universe`].

use crate::Result;
use crate::query::Query;
use crate::sim::Universe;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// One agent's in-progress step.
pub struct Step<'a, 'py> {
    py: Python<'py>,
    /// Every agent's most recently committed state. This is what `agent!` reads.
    universe: &'a Universe,
    /// This agent's own state as of the end of its previous step: what `prev!` reads.
    previous: Bound<'py, PyDict>,
    /// What this agent has produced so far during this step.
    next: Bound<'py, PyDict>,
}

impl<'a, 'py> Step<'a, 'py> {
    pub fn new(py: Python<'py>, universe: &'a Universe, previous: Bound<'py, PyDict>) -> Self {
        Self {
            py,
            universe,
            previous,
            next: PyDict::new(py),
        }
    }

    /// Resolve a consumed query.
    ///
    /// `Ok(None)` means the query names something this step has not produced yet —
    /// ask again once another manager has run.
    pub fn read(&self, query: &Query) -> Result<Option<Bound<'py, PyAny>>> {
        self.read_at(query, false)
    }

    /// Store a produced value at `query`.
    pub fn write(&self, query: &Query, value: &Bound<'py, PyAny>) -> Result<()> {
        match query {
            Query::Base(field) => set(&self.next, field, value),
            Query::Access { base, field } => set(&self.container(base)?, field, value),
            // A tuple names one place per element, so the manager returns one value per
            // element, in the order it wrote them — the mirror of a consumed tuple
            // becoming the manager's arguments.
            Query::Tuple(queries) => {
                let values = value
                    .cast::<PyTuple>()
                    .map_err(|_| format!("cannot produce into `{query}`: `{value}` is not a tuple"))?;
                if values.len() != queries.len() {
                    return Err(format!(
                        "cannot produce into `{query}`: it names {} values, and the state manager returned {}",
                        queries.len(),
                        values.len()
                    ));
                }

                for (query, value) in queries.iter().zip(values.iter()) {
                    self.write(query, &value)?;
                }
                Ok(())
            }
            // `prev!` and `agent!` name state that is already committed, which is not
            // somewhere a manager can put a value.
            Query::Prev(_) | Query::Agent(_) => Err(format!("cannot produce into `{query}`")),
        }
    }

    /// The state this step produced, ready to commit.
    pub fn finish(self) -> Bound<'py, PyDict> {
        self.next
    }

    /// `prev` tracks whether we are inside a `prev!(..)`, which switches bare field
    /// queries from the state being built to the committed one.
    fn read_at(&self, query: &Query, prev: bool) -> Result<Option<Bound<'py, PyAny>>> {
        match query {
            Query::Base(field) => {
                let state = if prev { &self.previous } else { &self.next };
                get(state, field)
            }
            Query::Prev(query) => self.read_at(query, true),
            // Another agent is readable as of the last step it committed. The runtime has
            // already made sure it is not behind this step; it may be one step ahead of
            // it, but never mid-step.
            Query::Agent(agent) => Ok(Some(self.universe.read(self.py, agent)?.into_bound(self.py).into_any())),
            Query::Access { base, field } => {
                let Some(base) = self.read_at(base, prev)? else {
                    return Ok(None);
                };
                let base = base
                    .cast_into::<PyDict>()
                    .map_err(|_| format!("cannot read `{query}`: it is not a dictionary"))?;
                get(&base, field)
            }
            Query::Tuple(queries) => {
                let mut values = Vec::with_capacity(queries.len());
                for query in queries {
                    let Some(value) = self.read_at(query, prev)? else {
                        return Ok(None);
                    };
                    values.push(value);
                }
                Ok(Some(
                    PyTuple::new(self.py, values)
                        .map_err(|err| format!("cannot read `{query}`: {err}"))?
                        .into_any(),
                ))
            }
        }
    }

    /// The dictionary that `query` names, creating it if it is not there yet.
    ///
    /// This is what makes `produced: position.x` work without anyone having produced
    /// a `position` dictionary first.
    fn container(&self, query: &Query) -> Result<Bound<'py, PyDict>> {
        let field = match query {
            Query::Base(field) => field,
            Query::Access { base, field } => {
                let base = self.container(base)?;
                return existing_or_new(&base, field);
            }
            Query::Prev(_) | Query::Agent(_) | Query::Tuple(_) => {
                return Err(format!("cannot produce into `{query}`"));
            }
        };
        existing_or_new(&self.next, field)
    }
}

/// The dictionary at `field` of `state`, put there now if it was not there already.
fn existing_or_new<'py>(state: &Bound<'py, PyDict>, field: &str) -> Result<Bound<'py, PyDict>> {
    if let Some(value) = get(state, field)? {
        return value
            .cast_into::<PyDict>()
            .map_err(|_| format!("cannot produce into `{field}`: it is not a dictionary"));
    }

    let created = PyDict::new(state.py());
    set(state, field, created.as_any())?;
    Ok(created)
}

fn get<'py>(state: &Bound<'py, PyDict>, field: &str) -> Result<Option<Bound<'py, PyAny>>> {
    state
        .get_item(field)
        .map_err(|err| format!("cannot read `{field}`: {err}"))
}

fn set(state: &Bound<'_, PyDict>, field: &str, value: &Bound<'_, PyAny>) -> Result<()> {
    state
        .set_item(field, value)
        .map_err(|err| format!("cannot produce into `{field}`: {err}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::sim::Universe;
    use pyo3::types::PyFloat;

    #[test]
    fn interprets_one_state_managers_queries() {
        Python::initialize();
        Python::attach(|py| {
            let previous = PyDict::new(py);
            let position = PyDict::new(py);
            position.set_item("x", 1.0).unwrap();
            previous.set_item("position", &position).unwrap();

            let universe = Universe::empty();
            let step = Step::new(py, &universe, previous);

            let consumed = Query::parse("(prev!(position).x,)").unwrap();
            let read = step.read(&consumed).unwrap().unwrap();
            assert_eq!(read.extract::<(f64,)>().unwrap(), (1.0,));

            // Nothing has produced a `position` during this step yet, so reading one
            // comes back empty rather than falling back to the previous step.
            assert!(step.read(&Query::parse("position").unwrap()).unwrap().is_none());

            let produced = Query::parse("position.x").unwrap();
            step.write(&produced, &PyFloat::new(py, 2.0).into_any()).unwrap();
            let finished = step.finish();
            assert_eq!(
                finished
                    .get_item("position")
                    .unwrap()
                    .unwrap()
                    .get_item("x")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                2.0
            );
        });
    }

    #[test]
    fn produces_a_tuple_into_every_query_it_names() {
        Python::initialize();
        Python::attach(|py| {
            let universe = Universe::empty();
            let step = Step::new(py, &universe, PyDict::new(py));

            // One element names a field and the other a field of a dictionary that
            // nothing has produced yet, the same as a produced query that is not a tuple.
            let produced = Query::parse("(time, position.x,)").unwrap();
            let values = PyTuple::new(py, [PyFloat::new(py, 1.0), PyFloat::new(py, 2.0)]).unwrap();
            step.write(&produced, values.as_any()).unwrap();

            // A manager that returns the wrong number of values is told which it was.
            let short = PyTuple::new(py, [PyFloat::new(py, 1.0)]).unwrap();
            let error = step.write(&produced, short.as_any()).unwrap_err();
            assert!(
                error.contains("it names 2 values, and the state manager returned 1"),
                "{error}"
            );

            let error = step.write(&produced, &PyFloat::new(py, 1.0).into_any()).unwrap_err();
            assert!(error.contains("is not a tuple"), "{error}");

            let finished = step.finish();
            assert_eq!(
                finished.get_item("time").unwrap().unwrap().extract::<f64>().unwrap(),
                1.0
            );
            assert_eq!(
                finished
                    .get_item("position")
                    .unwrap()
                    .unwrap()
                    .get_item("x")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                2.0
            );
        });
    }
}
