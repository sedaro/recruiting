fn main() {
    // A Python extension module deliberately leaves CPython's symbols undefined:
    // whichever interpreter imports the module resolves them. macOS's linker has
    // to be told that explicitly (`-undefined dynamic_lookup`), and only the crate
    // actually being linked can tell it — `cargo::rustc-cdylib-link-arg` does not
    // propagate from a dependency's build script, so pyo3 cannot do this for us.
    // A no-op everywhere else.
    pyo3_build_config::add_extension_module_link_args();
}
