# HTTP SERVER

import json
import os

from flask import Flask, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from modsim import AGENTS
from simulator import Simulator, init_tracing
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import logging

class Base(DeclarativeBase):
    pass


############################## Logging ##############################

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
LOG_LEVEL = {"TRACE": "DEBUG", "OFF": "CRITICAL"}.get(LOG_LEVEL, LOG_LEVEL)

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
        init[key]["timeStep"] = 0.01

    # Build a simulator
    simulator = Simulator(init, AGENTS)

    # Run simulation. Each frame is `[start, end, agentId, state]`, and holds the
    # one agent that stepped to it.
    frames = simulator.run()

    # Save data to database
    data = json.dumps(frames, default=float, sort_keys=True)
    simulation = Simulation(data=data)
    db.session.add(simulation)
    db.session.commit()

    return app.response_class(data, mimetype="application/json")
