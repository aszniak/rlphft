#!/bin/bash

# Script to run Python scripts with the correct Python version (3.10)
# Usage: ./run.sh script.py [arguments]

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script.py> [arguments]"
    exit 1
fi

SCRIPT=$1
shift # Remove the script name from the arguments

# Run with Python 3.10 which has all the required packages installed
python3.10 $SCRIPT "$@"
