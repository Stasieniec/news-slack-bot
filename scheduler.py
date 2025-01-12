import time
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
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
    """Run the scheduler with proper error handling."""
    scheduler = BlockingScheduler()
    
    # Schedule the daily digest to run at 9:00 AM every day
    scheduler.add_job(
        send_daily_digest,
        CronTrigger(hour=9, minute=0),
        name='daily_digest'
    )
    
    print("Scheduler started. Daily digest will be sent at 09:00 AM.")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopping...")
        scheduler.shutdown()
    except Exception as e:
        print(f"Scheduler error: {e}")

if __name__ == "__main__":
    run_scheduler()