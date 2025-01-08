# app.py

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from bot_handler import TasterayBot
import apis
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize the Slack app
app = AsyncApp(token=apis.SLACK_TOKEN)

# Initialize the bot
bot = TasterayBot()

@app.event("app_mention")
async def handle_mention(event, say):
    """Handle mentions of the bot."""
    # Validate event
    if not event or 'text' not in event or 'channel' not in event:
        logger.error(f"Invalid mention event: {event}")
        return
        
    logger.info(f"Handling mention from user {event.get('user')}")
    try:
        # Remove the bot mention from the text
        text = event['text']
        text = text.split('>', 1)[1].strip() if '>' in text else text
        
        await bot.handle_message(
            text=text,
            channel=event['channel'],
            thread_ts=event.get('thread_ts', event.get('ts')),
            sender_id=event.get('user'),
            is_mention=True
        )
    except Exception as e:
        logger.error(f"Error handling mention: {e}")
        await say("I'm sorry, I encountered an error while processing your request.")

@app.event("message")
async def handle_message(event, say):
    """Handle direct messages to the bot."""
    # Validate event
    if not event or 'text' not in event or 'channel' not in event:
        logger.error(f"Invalid message event: {event}")
        return
        
    # Ignore messages from the bot itself
    if event.get('user') == bot.bot_id:
        return
        
    # Only handle DMs (messages in the bot's DM channel)
    if event.get('channel_type') != 'im':
        return
        
    logger.info(f"Handling DM from user {event.get('user')}")
    try:
        await bot.handle_message(
            text=event['text'],
            channel=event['channel'],
            thread_ts=event.get('thread_ts', event.get('ts')),
            sender_id=event.get('user'),
            is_mention=False
        )
    except Exception as e:
        logger.error(f"Error handling DM: {e}")
        await say("I'm sorry, I encountered an error while processing your request.")

async def main():
    # Initialize the bot first
    logger.info("Initializing bot...")
    await bot.initialize()
    logger.info("Bot initialized successfully!")
    
    # Then start the socket mode handler
    handler = AsyncSocketModeHandler(app, apis.SLACK_APP_TOKEN)
    await handler.start_async()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())