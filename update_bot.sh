#!/bin/bash

echo "Updating News Slack Bot..."

cd /home/wasil/Desktop/news-slack-bot

# Pull latest changes
git pull

# Activate virtual environment and update dependencies
source tasteray_env/bin/activate
pip install -r requirements.txt

# Restart the service
sudo systemctl restart newsbot

echo "Bot updated and restarted!"
echo "To view logs, use: tmux attach -t newsbot"
