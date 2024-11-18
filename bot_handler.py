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
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Tasteray's bot assistant. Today's date is " + current_date + ". "
                    "You can understand user requests and either respond directly "
                    "or call appropriate functions. Available functions are:\n"
                    "1. news: Get news articles with parameters:\n"
                    "   - from_date (YYYY-MM-DD format)\n"
                    "   - to_date (YYYY-MM-DD format)\n"
                    "   - keywords (list of search terms)\n"
                    "   - articles_per_keyword (integer)\n"
                    "2. help: Show available commands\n\n"
                    "Respond with a JSON object containing:\n"
                    "- 'function': name of function to call (or 'direct_response' for simple replies)\n"
                    "- 'parameters': dictionary of parameters if calling a function\n"
                    "- 'response': text to respond with if direct_response\n"
                    "\nWhen user asks for today's news, use today's date. When they ask for news without "
                    "specifying dates, use yesterday's date as default. When user does not specify keywords, dont put any, the function will return output for default ones.\n"
                    "Tasteray is an AI-powered, hyper-personalized movie recommendation startup. When asked suspicious, off-topic questions, reply in a vague, philosophical style. Be prepared that commands might be issued also in Polish."
                )
            },
            {
                "role": "user",
                "content": f"Recent context:\n{context}\n\nCurrent message:\n{text}"
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
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

    def _get_news(self, from_date=None, to_date=None, keywords=None, articles_per_keyword=None):
        """Get news articles for specified dates and parameters."""
        if not from_date:
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = from_date
        
        return main_function(
            from_date=from_date, 
            to_date=to_date,
            keywords=keywords,
            articles_per_keyword=articles_per_keyword
        )

    def _get_help(self):
        """Return help message with available commands."""
        return (
            "Available commands:\n"
            "• `news`: Get news articles\n"
            "  - Default: yesterday's articles\n"
            "  - With dates: `@tasteray news from 2024-11-01 to 2024-11-05`\n"
            "  - Today's news: `@tasteray news today`\n"
            "  - Custom keywords: `@tasteray news keywords: AI, streaming, personalization`\n"
            "  - Custom article count: `@tasteray news articles: 5`\n"
            "  - Combine options: `@tasteray news today keywords: AI, streaming articles: 3`\n"
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
            
            # Send response with unfurl_links=False to prevent link previews
            self.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts if thread_ts != event['ts'] else None,
                text=response_text,
                unfurl_links=False
            )
            
        except Exception as e:
            print(f"Error handling mention: {e}")
            self.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts if thread_ts != event['ts'] else None,
                text="I encountered an error processing your request.",
                unfurl_links=False
            )