# HTTP SERVER

import json
import os

from flask import Flask, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from modsim import AGENTS
# NOTE: the simulation runtime is Rust. See `simulator/` for the query language,
# the interpreter, and the runtime; `modsim.py` for the models it runs.
from sedaro_nano_simulator import Simulator, init_tracing
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import logging

class Base(DeclarativeBase):
    pass


############################## Logging ##############################

# $LOG_LEVEL sets the level of both halves of this app: Python's `logging` here, and
# the Rust runtime's `tracing` in `init_tracing`. Both write to stderr. `trace` and
# `off` are the runtime's names, and mean the nearest thing `logging` has.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
LOG_LEVEL = {"TRACE": "DEBUG", "OFF": "CRITICAL"}.get(LOG_LEVEL, LOG_LEVEL)

# Nothing may log before this: the first call to `logging.warning` on a root logger
# with no handlers installs one itself, and `basicConfig` then does nothing.
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
init_tracing()
logging.info(f"Logging at $LOG_LEVEL={LOG_LEVEL}")


############################## Application Configuration ##############################

app = Flask(__name__)
CORS(app, origins=["http://localhost:3030"])

db = SQLAlchemy(model_class=Base)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
db.init_app(app)

############################## Database Models ##############################


class Simulation(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    data: Mapped[str]


with app.app_context():
    db.create_all()


############################## API Endpoints ##############################


@app.get("/")
def health():
    return "<p>Sedaro Nano API - running!</p>"


@app.get("/simulation")
def get_data():
    # Get most recent simulation from database
    simulation: Simulation = Simulation.query.order_by(Simulation.id.desc()).first()
    return simulation.data if simulation else []


@app.post("/simulation")
def simulate():
    # Get data from request in this form
    # init = {
    #     "Body1": {"position": {"x": 0, "y": 0.1, "z": 0}, "velocity": {...}, "mass": 1},
    #     "Body2": {"position": {"x": 0, "y": 1, "z": 0}, "velocity": {...}, "mass": 0.1},
    # }

    # Define time and timeStep for each agent
    init: dict = request.json
    for key in init.keys():
        init[key]["time"] = 0
        init[key]["timeStep"] = 100

    # Create simulator. This parses every query the agents declare.
    simulator = Simulator(init, AGENTS)

    # Run simulation. Each frame is `[start, end, {agentId: state}]`, and holds the
    # one agent that stepped to it.
    frames = simulator.run()

    # Simulation state is whatever the models in `modsim.py` built, so `default`
    # covers a number that is not one Python's json knows — a `numpy.float32`, say.
    # `sort_keys` writes the fields of a state in the same order however the agents'
    # threads happened to fill them in.
    data = json.dumps(frames, default=float, sort_keys=True)

    # Save data to database
    simulation = Simulation(data=data)
    db.session.add(simulation)
    db.session.commit()

    return app.response_class(data, mimetype="application/json")
