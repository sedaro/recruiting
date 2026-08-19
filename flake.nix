{
  description = "Sedaro Nano: the tiniest possible mockup of our system";

  # The current release branch, not `nixos-unstable`: every revision on it has been
  # built by Hydra, so entering this shell downloads binaries instead of compiling
  # them. Worth moving on at each release — an end-of-life branch stops receiving
  # fixes, and packages it pins start being marked insecure.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";

  outputs =
    inputs:
    let
      lib = inputs.nixpkgs.lib;

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      overlays = [ (import inputs.rust-overlay) ];

      forEachSupportedSystem =
        f:
        lib.genAttrs supportedSystems (
          system:
          let
            pkgs = import inputs.nixpkgs { inherit system overlays; };
          in
          f {
            inherit system pkgs;
            # One toolchain definition, shared with non-nix setups via rustup.
            toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
            # The interpreter the simulator is built against and imported from.
            # It has to stay in step with the `abi3-py314` feature in
            # `simulator/Cargo.toml` and with the image in `app/Dockerfile`.
            #
            # Bare, with no packages: those go into the `.venv` the shell hook
            # creates, from `app/requirements.txt`, so that the versions here are
            # the ones Docker installs and there is only one list of them.
            python = pkgs.python314;
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs, toolchain, python, ... }:
        {
          default = pkgs.mkShell {
            # Everything `bin/main --dev` needs: the simulator and its tests, the
            # API, and the frontend.
            packages = [
              toolchain
              python
              pkgs.git
              # Vite, for the frontend. Node 24 is the current LTS line.
              pkgs.nodejs_24
              # Runs the API and the frontend together, for `bin/main --dev`.
              pkgs.process-compose
              # Builds the venv below, and the way to add a dependency: `uv pip
              # install`. It resolves and installs the app's requirements in about
              # the time pip spends working out that it has nothing to do.
              pkgs.uv
              # No docker client: `bin/main` wants an engine as well, so that is a
              # thing you install rather than something a shell can hand you.
            ]
            # macOS gets its linker from the stdenv's clang; adding gcc there is
            # both a large build and a way to hand rustc a compiler that rejects
            # Apple's flags.
            ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux [
              pkgs.gcc
              pkgs.mold
            ];

            env = {
              RUST_BACKTRACE = "1";
              # The interpreter is nix's, above, and it is the one the simulator is
              # built against — so uv must use it rather than helpfully fetching a
              # CPython of its own, which would be a second Python in the shell and
              # the wrong one for an `abi3-py314` extension module to be imported
              # from.
              UV_NO_MANAGED_PYTHON = "1";
              UV_PYTHON_DOWNLOADS = "never";
              # `cargo test` links libpython, unlike the extension module build, so
              # the test binary needs to find it at run time. The other two are
              # what numpy's wheel dynamically loads, which nix does not patch.
              LD_LIBRARY_PATH = lib.makeLibraryPath [
                python
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ];
            }
            // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS = "-C link-arg=-fuse-ld=mold";
              CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS = "-C link-arg=-fuse-ld=mold";
            };

            shellHook = ''
              # Where this flake lives, found by walking up rather than taken from the
              # working directory, so `nix develop` works from a subdirectory too.
              root="$PWD"
              while [ ! -e "$root/flake.nix" ] && [ "$root" != / ]; do
                root="$(dirname "$root")"
              done

              export PATH="$root/bin:$PATH"

              # `bin/build` links the extension module into `build/`, so
              # `import sedaro_nano_simulator` works from anywhere in the repo.
              export PYTHONPATH="$root/build:$PYTHONPATH"

              # Set up (and activate) a local virtualenv. The Python packages come
              # from wheels rather than from nix, which is the point: nix builds
              # them from source, and running SQLAlchemy's test suite to enter a
              # shell is not a good trade.
              #
              # `bin/setup` is the same script the non-nix setup in the README
              # runs, so there is one definition of what the virtualenv contains,
              # and it does nothing on a shell you have already entered.
              "$root/bin/setup"
              export VIRTUAL_ENV="$root/.venv"
              # shellcheck disable=SC1091
              . "$VIRTUAL_ENV/bin/activate"
            '';
          };
        }
      );
    };
}
