# bot_handler.py

import json
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import apis
from news_collector import main_function
from typing import List, Dict
from openai import OpenAI

client = OpenAI(api_key=apis.OPEN_AI_TR)

class TasterayBot:
    def __init__(self):
        self.client = WebClient(token=apis.SLACK_TOKEN)
        self.functions = {
            'news': self._get_news,
            'help': self._get_help,
        }

    def _get_context(self, channel: str, thread_ts: str = None) -> List[Dict]:
        """Retrieve last 5 messages from the conversation for context."""
        try:
            # If in a thread
            if thread_ts:
                response = self.client.conversations_replies(
                    channel=channel,
                    ts=thread_ts,
                    limit=5
                )
            else:
                response = self.client.conversations_history(
                    channel=channel,
                    limit=5
                )
            
            messages = []
            for msg in response['messages'][:5]:
                messages.append({
                    'user': msg.get('user', 'unknown'),
                    'text': msg.get('text', ''),
                    'ts': msg.get('ts', '')
                })
            return messages
        except SlackApiError as e:
            print(f"Error getting context: {e}")
            return []

    def _analyze_command(self, text: str, context: List[Dict]) -> dict:
        """Use LLM to analyze the command and determine the response."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Tasteray's bot assistant. You can understand user requests and either respond directly "
                    "or call appropriate functions. Available functions are:\n"
                    "1. news: Get news articles (parameters: from_date, to_date in YYYY-MM-DD format)\n"
                    "2. help: Show available commands\n\n"
                    "Respond with a JSON object containing:\n"
                    "- 'function': name of function to call (or 'direct_response' for simple replies)\n"
                    "- 'parameters': dictionary of parameters if calling a function\n"
                    "- 'response': text to respond with if direct_response\n"
                )
            },
            {
                "role": "user",
                "content": f"Recent context:\n{context}\n\nCurrent message:\n{text}"
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error analyzing command: {e}")
            return {
                'function': 'direct_response',
                'response': "I'm sorry, I couldn't process that request. Try 'help' for available commands."
            }

    def _get_news(self, from_date=None, to_date=None):
        """Get news articles for specified dates."""
        if not from_date:
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = from_date
        
        return main_function(from_date=from_date, to_date=to_date)

    def _get_help(self):
        """Return help message with available commands."""
        return (
            "Available commands:\n"
            "• `news`: Get news articles (default: yesterday's articles)\n"
            "  - Example: `@tasteray news`\n"
            "  - With dates: `@tasteray news from 2024-11-01 to 2024-11-05`\n"
            "• `help`: Show this help message\n"
            "\nJust mention me (@tasteray) with any of these commands!"
        )

    def handle_mention(self, event: Dict):
        """Handle mentions and execute appropriate commands."""
        try:
            channel = event['channel']
            thread_ts = event.get('thread_ts', event.get('ts'))
            
            # Get conversation context
            context = self._get_context(channel, thread_ts)
            
            # Analyze the command
            command_analysis = self._analyze_command(event['text'], context)
            
            # Execute command or respond directly
            if command_analysis['function'] == 'direct_response':
                response_text = command_analysis['response']
            else:
                func = self.functions.get(command_analysis['function'])
                if func:
                    response_text = func(**command_analysis.get('parameters', {}))
                else:
                    response_text = "I'm sorry, I don't know how to do that yet."
            
            # Send response
            self.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts if thread_ts != event['ts'] else None,
                text=response_text
            )
            
        except Exception as e:
            print(f"Error handling mention: {e}")
            self.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts if thread_ts != event['ts'] else None,
                text="I encountered an error processing your request."
            )