#!/bin/bash

# Script to run CartPole RL with the correct Python version
# Usage: ./run_cartpole.sh [arguments]

# Check if rl_venv virtual environment exists and activate it
if [ -d "rl_venv" ]; then
    echo "Activating RL virtual environment..."
    source rl_venv/bin/activate
    PYTHON_CMD="python"
else
    # Fall back to system Python 3.10 if venv doesn't exist
    echo "No RL virtual environment found, using system Python 3.10..."
    PYTHON_CMD="python3.10"
fi

# Run CartPole with any provided arguments
$PYTHON_CMD cartpole.py "$@"

# Deactivate virtual environment if activated
if [ -d "rl_venv" ]; then
    deactivate 2>/dev/null
fi
