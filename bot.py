from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import schedule
import time
from flask import Flask, jsonify
from main import main_function
import apis

# Flask app for Slack events if needed in the future
app = Flask(__name__)

# Initialize Slack client with your bot token
client = WebClient(token=apis.SLACK_TOKEN)

# Define the function you want to call daily
def daily_task():
    # Replace this function with the actual task you want to perform

    result = "Testowa codzienna wiadomość"  # or call your custom function here

    # Post the result to the specified Slack channel
    try:
        client.chat_postMessage(
            channel="#insights-bot",  # Replace with the channel name
            text=result
        )
        print("Message posted successfully!")
    except SlackApiError as e:
        print(f"Error posting message: {e.response['error']}")


schedule.every().day.at("9:30").do(daily_task)

# Run the schedule in a background thread
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Run the scheduled task in a separate thread
import threading
threading.Thread(target=run_schedule).start()

# Start Flask server if you want to expand to handle events in the future
@app.route("/slack/events", methods=["POST"])
def slack_events():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
