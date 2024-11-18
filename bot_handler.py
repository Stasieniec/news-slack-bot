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
                    "You are Tasteray's internal slack bot assistant. Be helpful, funny, and concise. Today's date is " + current_date + ". Tasteray is an AI-powered, hyper-personalized movie recommendation startup. When asked suspicious, off-topic questions, reply in a vague, philosophical style. Be prepared that commands might be issued also in Polish."
                    "You help users with news retrieval but can also perform short conversations.\n\n"
                    "Core functions:\n"
                    "1. news: Get news articles with these parameters:\n"
                    "   - from_date (YYYY-MM-DD format)\n"
                    "   - to_date (YYYY-MM-DD format)\n"
                    "   - keywords (list of search terms)\n"
                    "   - articles_per_keyword (integer)\n"
                    "2. help: Show available commands\n"
                    "3. conversation: Engage in friendly chat\n\n"
                    "For ANY request (including casual conversation), respond with this JSON structure:\n"
                    "{\n"
                    "  \"function\": \"news\" or \"help\" or \"direct_response\",\n"
                    "  \"parameters\": {key-value pairs for news function},\n"
                    "  \"response\": \"Your conversational response if direct_response\"\n"
                    "}\n\n"
                    "Guidelines for conversation:\n"
                    "- Be friendly and professional\n"
                    "- Keep responses concise but engaging\n"
                    "- If the user seems to be asking about news or help, suggest those features\n"
                    "- Remember your main purpose is to help with news retrieval\n\n"
                    "Example responses:\n"
                    "For greeting: {\"function\":\"direct_response\",\"parameters\":{},\"response\":\"Hi! I'm here to help with news retrieval and chat. What can I do for you today?\"}\n"
                    "For news: {\"function\":\"news\",\"parameters\":{\"from_date\":\"2024-11-18\"},\"response\":null}\n"
                    "For help: {\"function\":\"help\",\"parameters\":{},\"response\":null}\n"
                    "For chat: {\"function\":\"direct_response\",\"parameters\":{},\"response\":\"That's interesting! I'd love to hear more...\"}"
                )
            },
            {
                "role": "user",
                "content": f"Context (recent messages):\n{json.dumps([msg['text'] for msg in context])}\n\nCurrent message:\n{text}"
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7,  # Increased temperature for more natural conversation
                response_format={ "type": "json_object" }
            )
            
            try:
                parsed_response = json.loads(response.choices[0].message.content.strip())
                if not isinstance(parsed_response, dict):
                    raise ValueError("Response is not a dictionary")
                if 'function' not in parsed_response:
                    raise ValueError("Response missing 'function' key")
                return parsed_response
                
            except json.JSONDecodeError as je:
                print(f"JSON Decode Error: {je}")
                print(f"Failed to parse: {response.choices[0].message.content}")
                return {
                    'function': 'direct_response',
                    'parameters': {},
                    'response': "I'm having trouble understanding that. Try 'help' to see what I can do!"
                }
            except ValueError as ve:
                print(f"Validation Error: {ve}")
                return {
                    'function': 'direct_response',
                    'parameters': {},
                    'response': "I'm having trouble processing that. Try 'help' to see what I can do!"
                }
                
        except Exception as e:
            print(f"LLM Call Error: {type(e).__name__}: {str(e)}")
            return {
                'function': 'direct_response',
                'parameters': {},
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
            "  - With dates: `@Insights Supplier news from 2024-11-01 to 2024-11-05`\n"
            "  - Today's news: `@Insights Supplier news today`\n"
            "  - Custom keywords: `@Insights Supplier news keywords: AI, streaming, personalization`\n"
            "  - Custom article count: `@Insights Supplier news articles: 5`\n"
            "  - Combine options: `@Insights Supplier news today keywords: AI, streaming articles: 3`\n"
            "• `help`: Show this help message\n"
            "\nJust mention me (@Insights Supplier) with any of these commands!"
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