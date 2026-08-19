FROM rust:1.97.1 AS builder

RUN mkdir /queries
WORKDIR /queries
COPY queries .

RUN cargo clean
RUN cargo build --release

# Running python
FROM python:3.14

# Create and set working directory
RUN mkdir /app
WORKDIR /app

COPY app/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app .

RUN mkdir /queries

COPY --from=builder /queries/target/release/sedaro-nano-queries /queries/target/release/sedaro-nano-queries

CMD [ "python3", "-m" , "flask", "run", "--debug", "--host=0.0.0.0", "--port=8000" ]