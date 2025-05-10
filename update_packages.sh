#!/bin/bash

# Update all required packages for the project

# Check if we should use virtual environment for RL components
if [[ "$1" == "--venv" ]]; then
    # Create virtual environment if it doesn't exist
    if [ ! -d "rl_venv" ]; then
        echo "Creating virtual environment for RL components..."
        python3 -m venv rl_venv
    fi

    # Activate virtual environment
    echo "Activating RL virtual environment..."
    source rl_venv/bin/activate

    # Install all requirements
    echo "Installing requirements in virtual environment..."
    pip install -r requirements.txt

    # Deactivate virtual environment
    echo "Deactivating virtual environment..."
    deactivate

    echo "Done! All packages installed in virtual environment."
    echo "Use ./run_cartpole.sh to run the CartPole environment."
else
    # Install for system Python 3.10
    echo "Installing required packages for Python 3.10..."
    python3.10 -m pip install --user -r requirements.txt

    # Add additional packages that might be needed but not in requirements.txt
    echo "Installing additional packages for visualization..."
    python3.10 -m pip install --user matplotlib pandas numpy seaborn

    echo "Done! All packages installed for Python 3.10"
    echo "To use a virtual environment instead, run with --venv flag"
fi
