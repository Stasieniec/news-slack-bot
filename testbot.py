from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import apis
from news_collector import main_function

# Initialize the Slack client with your bot token
client = WebClient(token=apis.SLACK_TOKEN)

# Function to send a message to a specified Slack channel
def send_message_to_slack():
    message = main_function(from_date='2024-11-15', to_date='2024-11-18')  # Call the test function and get its output
    try:
        # Send the message to the specified channel
        response = client.chat_postMessage(
            channel="#insights-bot",  # Replace with the actual Slack channel name
            text=message,
            unfurl_links=False  # Disable link preview
        )
        print(f"Message posted successfully: {response['ts']}")
    except SlackApiError as e:
        print(f"Error posting message: {e.response['error']}")

# Run the test message function
if __name__ == "__main__":
    send_message_to_slack()
