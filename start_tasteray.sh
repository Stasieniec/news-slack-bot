#!/bin/bash

# Navigate to the bot directory
cd /path/to/tasteray_bot

# Activate virtual environment
source tasteray_env/bin/activate

# Start the scheduler in the background
python scheduler.py &

# Start the main bot application
python app.py