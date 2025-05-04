#!/bin/bash

# Update all required packages for our Python 3.10 installation

echo "Installing required packages for Python 3.10..."
python3.10 -m pip install --user -r requirements.txt

# Add additional packages that might be needed but not in requirements.txt
echo "Installing additional packages for visualization..."
python3.10 -m pip install --user matplotlib pandas numpy seaborn

echo "Done! All packages should be installed for Python 3.10"
