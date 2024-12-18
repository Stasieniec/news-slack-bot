#!/bin/bash

# Directory where your bot code lives
BOT_DIR="/home/wasil/Desktop/news-slack-bot"
SESSION_NAME="newsbot"

# Ensure we're in the right directory
cd $BOT_DIR

# Function to check if tmux session exists
tmux_session_exists() {
    tmux has-session -t $SESSION_NAME 2>/dev/null
}

# Kill existing session if it exists
if tmux_session_exists; then
    tmux kill-session -t $SESSION_NAME
fi

# Create new session
tmux new-session -d -s $SESSION_NAME

# Navigate to bot directory and activate virtual environment
tmux send-keys -t $SESSION_NAME "cd $BOT_DIR && source tasteray_env/bin/activate && python scheduler.py" C-m

# Split window horizontally
tmux split-window -h -t $SESSION_NAME

# Navigate to bot directory and activate virtual environment in second pane
tmux send-keys -t $SESSION_NAME.1 "cd $BOT_DIR && source tasteray_env/bin/activate && python app.py" C-m

echo "Bot started in tmux session '$SESSION_NAME'"
echo "To view the session, use: tmux attach -t $SESSION_NAME"