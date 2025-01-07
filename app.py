# app.py

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import apis
from bot_handler import TasterayBot
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = App(token=apis.SLACK_TOKEN)
bot = TasterayBot()

@app.event("app_home_opened")
def handle_app_home_opened(event, say):
    """Handle when a user opens the app home tab."""
    user_id = event["user"]
    logger.info(f"User {user_id} opened the home tab")
    bot.publish_home_tab(user_id)

@app.event("app_mention")
def handle_mentions(event, say):
    """Handle when the bot is mentioned in channels."""
    bot.handle_message(event, is_mention=True)

@app.event("message")
def handle_message(event, say):
    """Handle direct messages and other message events."""
    # Only handle direct messages (IM) and not mentions or other message subtypes
    if event.get('channel_type') == 'im' and not event.get('subtype'):
        logger.info(f"Handling DM from user {event.get('user')}")
        bot.handle_message(event, is_mention=False)

def main():
    """Main function to start the bot."""
    logger.info("Starting Ray bot...")
    handler = SocketModeHandler(app, apis.SLACK_APP_TOKEN)
    handler.start()

if __name__ == "__main__":
    main()