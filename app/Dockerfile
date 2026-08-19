FROM rust:1.97.1 AS builder

RUN mkdir /simulator
WORKDIR /simulator
COPY simulator .

# The `abi3-py314` build targets Python's stable ABI, so it needs no interpreter
# of its own — which is just as well, because this image has none.
ENV PYO3_NO_PYTHON=1
# `--lib`: the extension module is the only thing this crate builds that anything
# imports, and it is the only thing that can be built here — a test binary links
# libpython, which this stage has no interpreter to provide.
RUN cargo build --release --lib --features extension-module

# Running python
FROM python:3.14

# Create and set working directory
RUN mkdir /app
WORKDIR /app

COPY app/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app .

# The simulation runtime, importable as `sedaro_nano_simulator`. Cargo names a
# `cdylib` after the library; Python needs it named after the module. It lives
# outside `/app` because docker compose bind-mounts the host's `app/` over that,
# which would hide anything the image put there.
COPY --from=builder \
  /simulator/target/release/libsedaro_nano_simulator.so \
  /build/sedaro_nano_simulator.so
ENV PYTHONPATH=/build

CMD [ "python3", "-m" , "flask", "run", "--debug", "--host=0.0.0.0", "--port=8000" ]
