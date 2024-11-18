from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import apis
from news_collector import main_function

def send_daily_digest():
    """Send daily digest of yesterday's articles."""
    client = WebClient(token=apis.SLACK_TOKEN)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        message = main_function(from_date=yesterday, to_date=yesterday)
        response = client.chat_postMessage(
            channel="#insights-bot",
            text=message,
            unfurl_links=False
        )
        print(f"Daily digest sent successfully: {response['ts']}")
    except Exception as e:
        print(f"Error sending daily digest: {e}")

def run_scheduler():
    scheduler = BlockingScheduler()
    # Run every day at 9:00 AM
    scheduler.add_job(send_daily_digest, 'cron', hour=9, minute=0)
    
    print("Scheduler started. Daily digest will be sent at 9:00 AM.")
    scheduler.start()