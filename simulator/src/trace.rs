//! Simulator tracing and logging.

use std::env;
use std::io::{self, IsTerminal};
use tracing_subscriber::filter::LevelFilter;

/// Initialize the simulator's tracing and logging system idempotently based on `$LOG_LEVEL`, which is one of `OFF`, `ERROR`, `WARN`, `INFO`, `DEBUG`, or `TRACE`. The default is `INFO`.
pub fn init() {
    let level = env::var("LOG_LEVEL")
        .ok()
        .and_then(|level| level.parse::<LevelFilter>().ok())
        .unwrap_or(LevelFilter::INFO);

    let _ = tracing_subscriber::fmt()
        .with_max_level(level)
        .with_writer(io::stderr)
        // Colour when a person is watching, and not when this is being captured.
        .with_ansi(io::stderr().is_terminal())
        .try_init();
}
