#!/bin/bash

# Directory where your bot code lives
BOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$BOT_DIR/logs"

echo "Stopping bot processes..."

# Kill processes using their PIDs if available
if [ -f "$LOG_DIR/scheduler.pid" ]; then
    SCHEDULER_PID=$(cat "$LOG_DIR/scheduler.pid")
    if ps -p $SCHEDULER_PID > /dev/null; then
        kill $SCHEDULER_PID
        echo "Stopped scheduler process (PID: $SCHEDULER_PID)"
    fi
    rm "$LOG_DIR/scheduler.pid"
fi

if [ -f "$LOG_DIR/app.pid" ]; then
    APP_PID=$(cat "$LOG_DIR/app.pid")
    if ps -p $APP_PID > /dev/null; then
        kill $APP_PID
        echo "Stopped app process (PID: $APP_PID)"
    fi
    rm "$LOG_DIR/app.pid"
fi

# As a backup, also try to kill any remaining processes
pkill -f "python scheduler.py"
pkill -f "python app.py"

echo "Bot processes stopped" 