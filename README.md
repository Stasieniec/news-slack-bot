# Insights Supplier Bot

A Slack bot that collects and shares news articles about streaming services and movie recommendations. The bot runs daily digests and can respond to direct mentions.

## Features

- Daily news digests about streaming services and recommendations
- Responds to mentions and commands in Slack
- Provides personalized analysis for Tasteray's needs
- Runs in background using systemd service

## Setup

### Prerequisites

```bash
# Install required system packages
sudo apt update
sudo apt install -y python3-venv tmux git
```

### Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd news-slack-bot
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `apis.py` file with your API keys:
```python
SLACK_TOKEN = "xoxb-your-bot-token"  # Bot User OAuth Token
SLACK_APP_TOKEN = "xapp-your-app-token"  # App-Level Token
OPEN_AI_TR = "your-openai-key"
NEWS = "your-news-api-key"
```

### Running the Bot

#### Local Development
```bash
# Run the test version
python test_mention.py

# View the response in your Slack channel
@Insights Supplier hello
```

#### Production (Raspberry Pi)

1. Start the service:
```bash
sudo systemctl start newsbot
```

2. Check status:
```bash
sudo systemctl status newsbot
```

3. View logs:
```bash
tmux attach -t newsbot
# (Use Ctrl+B then D to detach)
```

### Common Commands

```bash
# Start the bot
sudo systemctl start newsbot

# Stop the bot
sudo systemctl stop newsbot

# Restart the bot
sudo systemctl restart newsbot

# View service status
sudo systemctl status newsbot

# View logs
tmux attach -t newsbot

# Update the bot (after code changes)
./update_bot.sh
```

## Development

### File Structure
```
news-slack-bot/
├── app.py              # Main bot application
├── scheduler.py        # Daily digest scheduler
├── bot_handler.py      # Bot logic and commands
├── news_collector.py   # News fetching and processing
├── apis.py            # API keys (not in git)
├── requirements.txt    # Python dependencies
├── start_bot.sh       # Bot startup script
└── update_bot.sh      # Update script
```

### Making Changes

1. Edit files locally
2. Test changes:
```bash
python test_mention.py  # For testing mentions
```

3. When ready, commit and push:
```bash
git add .
git commit -m "Your change description"
git push
```

4. On Raspberry Pi, update the bot:
```bash
./update_bot.sh
# Or manually:
git pull
sudo systemctl restart newsbot
```

### Slack Commands

The bot responds to these commands:
- `@Insights Supplier help` - Show available commands
- `@Insights Supplier news` - Get latest news
- `@Insights Supplier news from YYYY-MM-DD to YYYY-MM-DD` - Get news for specific dates

### Troubleshooting

1. Bot not responding:
```bash
# Check service status
sudo systemctl status newsbot

# View logs
tmux attach -t newsbot

# Restart the service
sudo systemctl restart newsbot
```

2. API errors:
- Verify tokens in `apis.py`
- Check Slack App settings at api.slack.com/apps
- Ensure all required scopes are enabled

3. Common issues:
- If bot crashes, check logs for errors
- Make sure virtual environment is activated when installing packages
- Verify all required Slack events are configured

## Maintenance

### Regular Tasks
1. Check logs periodically
2. Update dependencies monthly
3. Monitor API usage
4. Backup `apis.py` file

### Updating Dependencies
```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
sudo systemctl restart newsbot
```

## Security Notes

- Keep `apis.py` secure and never commit it to git
- Regularly rotate API keys
- Monitor bot activity for unusual patterns
- Keep system and dependencies updated

## Contact

For issues or questions, contact:
[Your Contact Information]