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

# Start both processes with nohup to keep them running after terminal closes
nohup python scheduler.py > "$LOG_DIR/scheduler.log" 2>&1 &
SCHEDULER_PID=$!
echo $SCHEDULER_PID > "$LOG_DIR/scheduler.pid"

nohup python app.py > "$LOG_DIR/app.log" 2>&1 &
APP_PID=$!
echo $APP_PID > "$LOG_DIR/app.pid"

echo "Bot processes started with PIDs:"
echo "Scheduler PID: $SCHEDULER_PID"
echo "App PID: $APP_PID"
echo "----------------------------------------"
echo "You can check the logs in $LOG_DIR"
echo "To view logs in real-time, use:"
echo "tail -f $LOG_DIR/scheduler.log $LOG_DIR/app.log"