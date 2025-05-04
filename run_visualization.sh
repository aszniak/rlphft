#!/bin/bash

# Script to visualize cryptocurrency technical indicators
# Usage: ./run_visualization.sh [symbol] [days]

ARGS=""

# Process arguments
if [ ! -z "$1" ]; then
    ARGS="--symbol $1"
fi

if [ ! -z "$2" ]; then
    ARGS="$ARGS --days $2"
fi

# Make sure to use Python 3.10 which has all required packages
python3.10 visualize_indicators.py $ARGS
