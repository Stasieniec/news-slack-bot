# app.py

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import apis
from bot_handler import TasterayBot

app = App(token=apis.SLACK_TOKEN)
bot = TasterayBot()

@app.event("app_mention")
def handle_mentions(event, say):
    bot.handle_mention(event)

if __name__ == "__main__":
    handler = SocketModeHandler(app, apis.SLACK_APP_TOKEN)
    handler.start()