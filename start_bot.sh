#!/bin/bash

# Directory where your bot code lives
BOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$BOT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

# Ensure we're in the right directory
cd $BOT_DIR

# Kill any existing instances
pkill -f "python scheduler.py"
pkill -f "python app.py"

# Activate virtual environment
source venv/bin/activate

echo "Starting bot processes..."
echo "----------------------------------------"

# Start both processes and combine their output
(python scheduler.py | tee "$LOG_DIR/scheduler.log") & \
(python app.py | tee "$LOG_DIR/app.log") &

# Store PIDs for potential later use
echo $! > "$LOG_DIR/app.pid"

echo "Bot processes started. Showing live logs below:"
echo "----------------------------------------"

# Use tail to follow both log files in real-time
tail -f "$LOG_DIR/scheduler.log" "$LOG_DIR/app.log"