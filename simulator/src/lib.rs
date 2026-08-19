//! The Sedaro Nano simulation runtime.
//!
//! A simulation is a set of *agents*. Each owns a bag of state and a list of *state
//! managers*: Python functions (see `app/modsim.py`) that consume some of that state
//! and produce a new piece of it. An agent says what each manager reads and writes, in
//! the query language, and never what order they run in — the runtime works that out.
//!
//! - [`query`] — the query language: syntax, AST, and parser.
//! - [`interp`] — resolving queries against agent state.
//! - [`sim`] — the runtime: it builds agents and runs each in its own thread.
//! - [`python`] — the pyo3 boundary, exported as the `sedaro_nano_simulator` module.
//! - [`trace`] — what the runtime logs, and where.

pub mod interp;
pub mod python;
pub mod query;
pub mod sim;
pub mod trace;

pub type Result<T> = std::result::Result<T, String>;
