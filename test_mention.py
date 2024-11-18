from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import apis

# Initialize your app with your bot token
app = App(token=apis.SLACK_TOKEN)

@app.event("app_mention")
def handle_mention(event, say):
    print(f"Got mention event: {event}")  # This will show in your console
    say("Hello! I'm here and responding to your mention!")

@app.message("hello")
def handle_message(message, say):
    print(f"Got message: {message}")  # This will show in your console
    say("Hey there!")

if __name__ == "__main__":
    # Start your app with Socket Mode
    handler = SocketModeHandler(app, apis.SLACK_APP_TOKEN)
    print("⚡️ Bolt app is running!")
    handler.start()
