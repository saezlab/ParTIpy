import os
import sys

# NOTE: this is a tempory solution, when this project is more mature
# we will rather use pip install -e

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append project root to sys.path
sys.path.insert(0, PROJECT_ROOT)
