#!/bin/bash

# Create virtual env
python3 -m venv quantum_venv

# Activate venv
source quantum_venv/bin/activate

# Install dependencies
pip install qutip numpy matplotlib

# Run the simulation code (assumes file is simulate_quantum_comm.py)
python sim.py

# Deactivate venv (optional, but good practice)
deactivate
