# bot_handler.py

import json
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import apis
from news_collector import main_function
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import logging
import re
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=apis.OPEN_AI_TR)

class TasterayBot:
    def __init__(self):
        self.client = WebClient(token=apis.SLACK_TOKEN)
        self.functions = {
            'news': self._get_news,
            'help': self._get_help,
            'summarize': self._get_summary,
            'task': self._create_clickup_task,
            'delete_last': self._delete_last_message,
        }
        self.channel_cache = {}
        # Context limits
        self.DM_CONTEXT_LIMIT = 15  # Increased from 10 to get more context
        self.MENTION_CONTEXT_LIMIT = 8  # Increased from 5 to get more context
        # Chunk sizes for summarization
        self.DM_CHUNK_SIZE = 10  # Increased from 5 for better context in chunks
        self.CHANNEL_CHUNK_SIZE = 25  # Increased from 20 for better summaries
        # Token limits
        self.DM_TOKEN_LIMIT = 2000  # Increased from 500 for complete responses
        self.MENTION_TOKEN_LIMIT = 1000  # Increased from 300 for complete responses
        
        # Initialize workspace access and Home tab
        self._initialize_workspace_access()
        self._initialize_home_tab()
        self.last_message_ts = {}  # Track last message timestamp per channel

    def _initialize_workspace_access(self):
        """Initialize workspace access by joining all accessible channels."""
        try:
            print("\nInitializing workspace access...")
            # Get list of all public channels
            response = self.client.conversations_list(
                types="public_channel",
                exclude_archived=True,
                limit=1000
            )
            
            channels = response['channels']
            joined = 0
            already_in = 0
            
            print(f"Found {len(channels)} public channels")
            
            for channel in channels:
                channel_id = channel['id']
                channel_name = channel['name']
                
                # Cache the channel
                self.channel_cache[channel_name] = channel_id
                
                # Try to join if not already a member
                if not channel.get('is_member', False):
                    try:
                        self.client.conversations_join(
                            channel=channel_id,
                            no_notify=True  # Join silently without notifying channel members
                        )
                        print(f"Silently joined #{channel_name}")
                        joined += 1
                    except SlackApiError as e:
                        if "already_in_channel" in str(e):
                            already_in += 1
                        else:
                            print(f"Failed to join #{channel_name}: {e}")
                else:
                    already_in += 1
            
            print(f"\nWorkspace access initialized:")
            print(f"- Silently joined {joined} new channels")
            print(f"- Already in {already_in} channels")
            print(f"- Total channels cached: {len(self.channel_cache)}")
            
        except SlackApiError as e:
            print(f"Error initializing workspace access: {e}")

    def _ensure_channel_access(self, channel_id: str) -> bool:
        """Ensure the bot has access to the specified channel."""
        try:
            # First check if we can get channel info - this verifies access
            try:
                channel_info = self.client.conversations_info(channel=channel_id)
                # If we can get info, we have access
                logger.info(f"Already have access to channel {channel_id}")
                return True
            except SlackApiError as e:
                if "channel_not_found" in str(e):
                    logger.warning(f"No access to channel {channel_id}")
                    return False
            
            # If we get here, try joining the channel
            try:
                self.client.conversations_join(
                    channel=channel_id,
                    no_notify=True  # Join silently without notifying channel members
                )
                logger.info(f"Successfully joined channel {channel_id}")
                return True
            except SlackApiError as e:
                if "already_in_channel" in str(e):
                    logger.info(f"Already in channel {channel_id}")
                    return True
                elif "cant_invite_self" in str(e) or "is_archived" in str(e):
                    logger.warning(f"Cannot join channel {channel_id}: {e}")
                    return False
                else:
                    logger.error(f"Failed to join channel {channel_id}: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error ensuring channel access: {e}")
            return False

    def _resolve_channel_reference(self, channel_ref: str) -> Optional[Tuple[str, str]]:
        """
        Resolve a channel reference to a channel ID and name.
        Returns tuple of (channel_id, channel_name) or None if not found.
        """
        logger.info(f"Resolving channel reference: {channel_ref}")
        
        try:
            # If it's already a channel ID format
            if channel_ref.startswith('C'):
                # Get channel info to verify it exists and get the name
                try:
                    response = self.client.conversations_info(channel=channel_ref)
                    channel_name = response['channel']['name']
                    logger.info(f"Resolved channel ID {channel_ref} to #{channel_name}")
                    return channel_ref, channel_name
                except SlackApiError as e:
                    logger.error(f"Failed to get info for channel ID {channel_ref}: {e}")
                    return None
            
            # If it's a channel name (with or without #)
            channel_name = channel_ref.lstrip('#')
            
            # Check cache first
            if channel_name in self.channel_cache:
                channel_id = self.channel_cache[channel_name]
                # Verify we can access the channel
                try:
                    self.client.conversations_info(channel=channel_id)
                    logger.info(f"Found channel #{channel_name} in cache with ID {channel_id}")
                    return channel_id, channel_name
                except SlackApiError:
                    # If we can't access it, remove from cache and continue searching
                    logger.warning(f"Cached channel {channel_name} is no longer accessible")
                    del self.channel_cache[channel_name]
            
            # List all channels the bot has access to
            response = self.client.conversations_list(
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=1000
            )
            
            for channel in response['channels']:
                # Cache all channels for future use
                self.channel_cache[channel['name']] = channel['id']
                
                if channel['name'] == channel_name:
                    channel_id = channel['id']
                    # Double check we can access this channel
                    try:
                        self.client.conversations_info(channel=channel_id)
                        logger.info(f"Resolved channel #{channel_name} to ID {channel_id}")
                        return channel_id, channel_name
                    except SlackApiError as e:
                        logger.warning(f"Found channel {channel_name} but cannot access it: {e}")
                        continue
            
            logger.warning(f"Could not find accessible channel matching reference: {channel_ref}")
            return None
            
        except SlackApiError as e:
            logger.error(f"Failed to list channels: {e}")
            return None

    def _get_context(self, channel: str, thread_ts: str = None, is_mention: bool = True, event: Dict = None) -> List[Dict]:
        """
        Retrieve conversation context.
        For mentions: last MENTION_CONTEXT_LIMIT messages
        For DMs: last DM_CONTEXT_LIMIT messages to maintain more context
        
        Args:
            channel: Channel ID to get context from
            thread_ts: Thread timestamp if in a thread
            is_mention: Whether this is a channel mention or DM
            event: The full Slack event object for additional context
        """
        limit = self.MENTION_CONTEXT_LIMIT if is_mention else self.DM_CONTEXT_LIMIT
        print("\n" + "="*50)
        print(f"RETRIEVING CONTEXT FROM {'CHANNEL' if is_mention else 'DM'}:")
        print(f"Channel: {channel}")
        print(f"Thread: {thread_ts if thread_ts else 'None'}")
        print(f"Limit: {limit}")
        print("="*50)
        
        logger.info(f"Getting context for {'mention in channel' if is_mention else 'DM'} {channel}" + 
                   (f" thread {thread_ts}" if thread_ts else "") +
                   f" with limit {limit}")
        try:
            # For DMs, always get the conversation history
            if not is_mention:
                print("\nGetting DM history...")
                current_ts = datetime.now().timestamp()
                
                # Get messages before the current timestamp
                response = self.client.conversations_history(
                    channel=channel,
                    limit=limit + 1,  # Get one extra message to include current
                    inclusive=True,
                    include_all_metadata=True
                )
                
                # If we didn't get enough messages and there are more
                if len(response.get('messages', [])) < limit and response.get('has_more', False):
                    print(f"\nFound only {len(response.get('messages', []))} messages, fetching more from history...")
                    cursor = response['response_metadata'].get('next_cursor')
                    while cursor and len(response.get('messages', [])) < limit:
                        historical_response = self.client.conversations_history(
                            channel=channel,
                            cursor=cursor,
                            limit=limit - len(response.get('messages', [])),
                            inclusive=True
                        )
                        # Combine messages, avoiding duplicates
                        all_messages = response.get('messages', [])
                        seen_ts = {msg['ts'] for msg in all_messages}
                        for msg in historical_response.get('messages', []):
                            if msg['ts'] not in seen_ts:
                                all_messages.append(msg)
                        response['messages'] = all_messages
                        
                        if not historical_response.get('has_more'):
                            break
                        cursor = historical_response['response_metadata'].get('next_cursor')
                    
                    print(f"Total messages after fetching more: {len(response.get('messages', []))}")
            
            # For channel mentions with thread
            elif thread_ts:
                print("\nGetting thread replies...")
                response = self.client.conversations_replies(
                    channel=channel,
                    ts=thread_ts,
                    limit=limit,
                    inclusive=True
                )
            
            # For channel mentions without thread
            else:
                print("\nGetting channel history...")
                # Get messages before the current message
                response = self.client.conversations_history(
                    channel=channel,
                    limit=limit + 1,  # Get one extra for current message
                    latest=event.get('ts'),  # Get messages before current message
                    inclusive=True,
                    include_all_metadata=True
                )
                
                # If we didn't get enough messages and there are more
                if len(response.get('messages', [])) < limit and response.get('has_more', False):
                    print(f"\nFound only {len(response.get('messages', []))} messages, fetching more from history...")
                    cursor = response['response_metadata'].get('next_cursor')
                    while cursor and len(response.get('messages', [])) < limit:
                        historical_response = self.client.conversations_history(
                            channel=channel,
                            cursor=cursor,
                            limit=limit - len(response.get('messages', [])),
                            inclusive=True
                        )
                        # Combine messages, avoiding duplicates
                        all_messages = response.get('messages', [])
                        seen_ts = {msg['ts'] for msg in all_messages}
                        for msg in historical_response.get('messages', []):
                            if msg['ts'] not in seen_ts:
                                all_messages.append(msg)
                        response['messages'] = all_messages
                        
                        if not historical_response.get('has_more'):
                            break
                        cursor = historical_response['response_metadata'].get('next_cursor')
                    
                    print(f"Total messages after fetching more: {len(response.get('messages', []))}")
            
            messages = []
            # Process messages in reverse order (oldest first)
            msg_list = response.get('messages', [])[:limit + 1]  # Include current message
            
            # Sort messages by timestamp to ensure correct order
            msg_list.sort(key=lambda x: float(x.get('ts', 0)))
            
            print("\nProcessing messages:")
            for msg in msg_list:
                # Skip bot's own messages in channel mentions to focus on user messages
                if msg.get('bot_id') and is_mention:
                    logger.debug("Skipping bot message in channel mention")
                    continue
                
                # Get user info for better context
                user_id = msg.get('user', 'unknown')
                is_bot = bool(msg.get('bot_id'))
                user_name = self._get_user_name(user_id)
                
                # Convert user mentions in the message text
                text = self._convert_user_mentions(msg.get('text', ''))
                
                # Include all messages for proper context
                message_data = {
                    'user': user_name,
                    'text': text,
                    'ts': msg.get('ts', ''),
                    'is_bot': is_bot
                }
                messages.append(message_data)
                print(f"[{datetime.fromtimestamp(float(message_data['ts'])).strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"{user_name}: {message_data['text']}")
            
            print(f"\nRetrieved {len(messages)} messages for context")
            print("="*50 + "\n")
            
            logger.debug(f"Retrieved {len(messages)} context messages")
            logger.debug(f"Context messages: {json.dumps(messages, indent=2)}")
            
            # Return all messages except the current one
            return messages[:-1] if len(messages) > 1 else []
            
        except SlackApiError as e:
            logger.error(f"Error getting context: {e}")
            print(f"\nError getting context: {e}")
            print("="*50 + "\n")
            return []

    def _get_channel_history(self, channel: str, oldest: str = None, latest: str = None) -> List[Dict]:
        """Retrieve channel history with pagination support."""
        all_messages = []
        cursor = None
        page = 1
        
        logger.info(f"Fetching history for channel {channel}" + 
                   (f" from {oldest}" if oldest else "") +
                   (f" to {latest}" if latest else ""))
        
        while True:
            try:
                params = {
                    'channel': channel,
                    'limit': 100
                }
                if oldest:
                    params['oldest'] = oldest
                if latest:
                    params['latest'] = latest
                if cursor:
                    params['cursor'] = cursor
                
                logger.debug(f"Fetching page {page} with params: {params}")
                response = self.client.conversations_history(**params)
                messages = response['messages']
                
                # Process messages to include user names
                for msg in messages:
                    user_id = msg.get('user', 'unknown')
                    msg['user_name'] = self._get_user_name(user_id)
                    # Convert user mentions in the message text and ensure user_name is used
                    msg['text'] = self._convert_user_mentions(msg.get('text', ''))
                    msg['user'] = msg['user_name']  # Replace user ID with name
                
                all_messages.extend(messages)
                
                logger.info(f"Retrieved {len(messages)} messages on page {page}")
                
                if not response['has_more']:
                    break
                    
                cursor = response['response_metadata']['next_cursor']
                page += 1
                
            except SlackApiError as e:
                logger.error(f"Error getting channel history on page {page}: {e}")
                break
        
        logger.info(f"Total messages retrieved: {len(all_messages)}")
        return all_messages

    def _get_summary(self, channel: str = None, from_date: str = None, to_date: str = None) -> str:
        """Get channel summary for the specified time range."""
        logger.info(f"Getting summary for channel {channel}" +
                   (f" from {from_date}" if from_date else "") +
                   (f" to {to_date}" if to_date else ""))

        # Detect language from the request (if channel contains Polish words, use Polish)
        request_language = 'pl' if any(word in str(channel).lower() for word in ['kanał', 'kanal', 'ten']) else 'en'
        logger.info(f"Detected request language: {request_language}")

        if not channel:
            logger.warning("No channel specified")
            return "Please specify a channel to summarize." if request_language == 'en' else "Proszę określić kanał do podsumowania."

        # Resolve channel reference
        channel_info = self._resolve_channel_reference(channel)
        if not channel_info:
            return (
                "Could not access channel: {channel}. For private channels, please invite me first using `/invite @Ray`"
                if request_language == 'en' else
                f"Nie mogę uzyskać dostępu do kanału: {channel}. W przypadku kanałów prywatnych, najpierw zaproś mnie używając `/invite @Ray`"
            )
        
        channel_id, channel_name = channel_info
        
        # Convert dates to timestamps if provided
        oldest = None
        latest = None
        if from_date:
            try:
                oldest = datetime.strptime(from_date, '%Y-%m-%d').timestamp()
            except ValueError:
                logger.error(f"Invalid from_date format: {from_date}")
                return (
                    "Invalid from_date format. Please use YYYY-MM-DD format."
                    if request_language == 'en' else
                    "Nieprawidłowy format daty początkowej. Proszę użyć formatu RRRR-MM-DD."
                )
        if to_date:
            try:
                latest = (datetime.strptime(to_date, '%Y-%m-%d') + timedelta(days=1)).timestamp()
            except ValueError:
                logger.error(f"Invalid to_date format: {to_date}")
                return (
                    "Invalid to_date format. Please use YYYY-MM-DD format."
                    if request_language == 'en' else
                    "Nieprawidłowy format daty końcowej. Proszę użyć formatu RRRR-MM-DD."
                )

        summary = self._summarize_channel(channel_id, oldest, latest, request_language)
        
        # Format the header with proper Slack markdown
        header_parts = []
        
        # Add channel name with icon
        if request_language == 'pl':
            header_parts.append(f":memo: *Podsumowanie Kanału: #{channel_name}*")
        else:
            header_parts.append(f":memo: *Channel Summary: #{channel_name}*")
        
        # Add date range if specified
        if from_date or to_date:
            date_range = []
            if from_date:
                date_range.append(f"{'od' if request_language == 'pl' else 'from'} {from_date}")
            if to_date:
                date_range.append(f"{'do' if request_language == 'pl' else 'to'} {to_date}")
            header_parts.append(f"_({' '.join(date_range)})_")
        
        # Combine header with summary
        return f"{' '.join(header_parts)}\n\n{summary}"

    def _analyze_command(self, text: str, context: List[Dict], is_mention: bool = True) -> dict:
        """Use LLM to analyze the command and determine the response."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Format context messages for better readability
        formatted_context = []
        for msg in context:
            ts = datetime.fromtimestamp(float(msg['ts'])).strftime('%Y-%m-%d %H:%M:%S')
            # Add special formatting for bot messages in the context
            if msg['is_bot']:
                formatted_context.append(f"[{ts}] Ray (assistant): {msg['text']}")
            else:
                formatted_context.append(f"[{ts}] {msg['user']} (user): {msg['text']}")
        
        print("\n" + "="*50)
        print("CONTEXT BEING SENT TO OPENAI:")
        print("="*50)
        print("\nFormatted conversation context:")
        for msg in formatted_context:
            print(msg)
        print("\nCurrent message:", text)
        print("="*50 + "\n")

        # Modify the prompt based on whether it's a DM or mention
        base_prompt = (
            f"You are Ray, an intelligent internal assistant for Tasteray - an AI-powered movie recommendation platform. "
            f"Today's date is {current_date}.\n\n"
            "Key Information:\n"
            "1. You are designed to help with various tasks including news gathering, channel summarization, and general assistance\n"
            "2. You should be helpful, clear, and informative in your responses.\n"
            "3. You are fully bilingual in English and Polish - you should:\n"
            "   - Understand commands in both languages\n"
            "   - Detect the language of the user's message and respond in the same language\n"
            "   - Handle Polish commands like 'podsumuj', 'wiadomości', 'pomoc'\n"
            "   - Understand Polish date formats and convert them appropriately\n"
            "4. You should maintain conversation context and refer to previous messages when relevant\n"
            "5. You should provide detailed explanations when asked\n\n"
            "6. Be friendly. Make jokes sometimes when there's a clear opportunity but don't go for cheesy ones. Try to not sound too robotic or corporate.\n\n"
            "7. For personal questions, answer as if you had a personality of a bored robot which sole purpose is to serve stupid people. You can be slighly mean. Don't be too obvious in that and don't mention that you are roleplaying. Keep the personal stuff short and to the point (unless user asks for your opinion).\n\n"
        )
        
        if is_mention:
            base_prompt += (
                "Channel Mode:\n"
                "- You're being mentioned in a channel, so keep responses concise and be focused.\n"
                "- Prioritize clear, actionable information\n"
                "- Use slack formatting for better readability in channel context\n"
                "- Respond in the same language as the user's message"
            )
        else:
            base_prompt += (
                "Direct Message Mode:\n"
                "- This is a private conversation, so you can be more detailed and conversational.\n"
                "- Maintain context of our ongoing discussion\n"
                "- Provide more comprehensive responses and explanations\n"
                "- Feel free to ask clarifying questions when needed\n"
                "- Respond in the same language as the user's message"
            )
        
        messages = [
            {
                "role": "system",
                "content": (
                    f"{base_prompt}\n\n"
                    "Available Functions:\n"
                    "1. news/wiadomości: Get news articles\n"
                    "   Parameters (always use English names):\n"
                    "   - from_date: Start date (YYYY-MM-DD)\n"
                    "   - to_date: End date (YYYY-MM-DD)\n"
                    "   - keywords: List of search terms\n"
                    "   - articles_per_keyword: Number of articles per keyword\n\n"
                    "2. summarize/podsumuj: Summarize channel content\n"
                    "   Parameters (always use English names):\n"
                    "   - channel: Channel to summarize (convert 'ten kanał' to 'this channel')\n"
                    "   - from_date: Start date (YYYY-MM-DD, optional)\n"
                    "   - to_date: End date (YYYY-MM-DD, optional)\n\n"
                    "3. task/zadanie: Create ClickUp task\n"
                    "   Parameters (always use English names):\n"
                    "   - list_id: ClickUp list ID (required)\n"
                    "   - task_name: Name of the task (required)\n"
                    "   - status: Task status (always in English: 'Open', 'In Progress', 'Done')\n"
                    "   - assignees: List of user IDs to assign\n"
                    "   - due_date: Due date (YYYY-MM-DD)\n"
                    "   - priority: Priority level (1: Urgent, 2: High, 3: Normal, 4: Low)\n"
                    "   - description: Task description\n\n"
                    "4. help/pomoc: Show available commands\n"
                    "5. delete_last/usuń ostatnią: Delete last message\n"
                    "   Parameters (always use English names):\n"
                    "   - channel: Current channel (automatically filled)\n"
                    "   - thread_ts: Current thread timestamp (automatically filled)\n\n"
                    "6. conversation: General chat and assistance\n\n"
                    "Response Format:\n"
                    "Always respond with this JSON structure, using ENGLISH function names and parameter names:\n"
                    "{\n"
                    "  \"function\": \"news\" or \"summarize\" or \"task\" or \"help\" or \"delete_last\" or \"direct_response\",\n"
                    "  \"parameters\": {key-value pairs for the function, always in English},\n"
                    "  \"response\": \"Your conversational response in the same language as the user's message\"\n"
                    "}\n\n"
                    "Response Guidelines:\n"
                    "1. Always consider the conversation context\n"
                    "2. Reference previous messages when relevant\n"
                    "3. Provide clear explanations for your actions\n"
                    "4. Use proper formatting for better readability\n"
                    "5. Ask for clarification if a request is ambiguous\n"
                    "6. Include relevant timestamps when referencing past messages\n"
                    "7. Detect and respond in the same language as the user\n"
                    "8. IMPORTANT: Always use English for function names and parameters\n\n"
                    "Example Responses:\n"
                    "1. Delete last message (Polish command -> English function):\n"
                    "   User: '@Ray usuń ostatnią wiadomość'\n"
                    "   Response: {\"function\":\"delete_last\",\"parameters\":{},\"response\":null}\n\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Previous conversation context (chronological order):\n" +
                    "\n".join(formatted_context) +
                    "\n\nCurrent message:\n" + text
                )
            }
        ]

        logger.info(f"Analyzing {'mention' if is_mention else 'DM'}: {text}")
        logger.debug(f"Context provided to LLM: {messages[1]['content']}")
        
        try:
            print("\nSending request to OpenAI...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=self.DM_TOKEN_LIMIT if not is_mention else self.MENTION_TOKEN_LIMIT,
                temperature=0.7,
                response_format={ "type": "json_object" }
            )
            
            try:
                parsed_response = json.loads(response.choices[0].message.content.strip())
                print("\nReceived response from OpenAI:")
                print(json.dumps(parsed_response, indent=2))
                print("="*50 + "\n")
                
                logger.debug(f"Parsed response: {parsed_response}")
                if not isinstance(parsed_response, dict):
                    raise ValueError("Response is not a dictionary")
                if 'function' not in parsed_response:
                    raise ValueError("Response missing 'function' key")
                return parsed_response
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON Decode Error: {je}")
                logger.error(f"Failed to parse: {response.choices[0].message.content}")
                return {
                    'function': 'direct_response',
                    'parameters': {},
                    'response': "I'm having trouble understanding that. Try 'help' to see what I can do!"
                }
            except ValueError as ve:
                logger.error(f"Validation Error: {ve}")
                return {
                    'function': 'direct_response',
                    'parameters': {},
                    'response': "I'm having trouble processing that. Try 'help' to see what I can do!"
                }
                
        except Exception as e:
            logger.error(f"LLM Call Error: {type(e).__name__}: {str(e)}")
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
            "Hi! I'm Ray, the internal Tasteray assistant. Here's what I can do:\n\n"
            "• `news`: Get news articles\n"
            "  - Default: yesterday's articles\n"
            "  - With dates: `@Ray news from 2024-11-01 to 2024-11-05`\n"
            "  - Today's news: `@Ray news today`\n"
            "  - Custom keywords: `@Ray news keywords: AI, streaming, personalization`\n"
            "  - Custom article count: `@Ray news articles: 5`\n"
            "  - Combine options: `@Ray news today keywords: AI, streaming articles: 3`\n\n"
            "• `summarize`: Get channel summaries\n"
            "  - Current channel: `@Ray summarize this channel`\n"
            "  - With dates: `@Ray summarize this channel from 2024-11-01 to 2024-11-05`\n"
            "  - Since date: `@Ray summarize this channel since 2024-11-01`\n"
            "  - Until date: `@Ray summarize this channel until 2024-11-05`\n"
            "  - Entire history: `@Ray summarize this channel history`\n\n"
            "• `task`: Create ClickUp tasks\n"
            "  - Basic: `@Ray task Create new feature`\n"
            "  - With status: `@Ray task Create new feature status: In Progress`\n"
            "  - With assignees: `@Ray task Create new feature assign: @john, @sarah`\n"
            "  - With due date: `@Ray task Create new feature due: 2024-02-01`\n"
            "  - With priority: `@Ray task Create new feature priority: high`\n"
            "  - Full example: `@Ray task Create new feature status: In Progress assign: @john due: 2024-02-01 priority: high`\n\n"
            "Just mention me (@Ray) with any of these commands!"
        )

    def handle_message(self, event: Dict, is_mention: bool = True):
        """Handle both mentions and direct messages."""
        try:
            channel = event['channel']
            thread_ts = event.get('thread_ts', event.get('ts'))
            current_ts = event.get('ts')
            
            logger.info(f"Handling {'mention' if is_mention else 'DM'} in channel {channel}" + 
                       (f" thread {thread_ts}" if thread_ts else ""))
            
            # Get conversation context (excluding current message)
            context = self._get_context(channel, thread_ts, is_mention, event)
            
            # For mentions, we need to remove the bot mention from the text
            text = event['text']
            if is_mention:
                # Remove the bot mention (everything up to the first space after the mention)
                text = re.sub(r'^<@[^>]+>\s*', '', text)
            
            # Add current message to debug log
            logger.debug(f"Current message [{current_ts}]: {text}")
            logger.debug(f"Context messages count: {len(context)}")
            
            # Analyze the command
            command_analysis = self._analyze_command(text, context, is_mention)
            
            # Execute command or respond directly
            if command_analysis['function'] == 'direct_response':
                response_text = command_analysis['response']
            else:
                func = self.functions.get(command_analysis['function'])
                if func:
                    # If it's a summarize command and channel parameter is "this channel", use current channel
                    if command_analysis['function'] == 'summarize':
                        params = command_analysis.get('parameters', {})
                        if params.get('channel') in ['this channel', 'this', 'here', None]:
                            params['channel'] = channel
                        response_text = func(**params)
                    # If it's a delete_last command, pass the current channel and thread_ts
                    elif command_analysis['function'] == 'delete_last':
                        response_text = func(
                            channel=channel,
                            thread_ts=thread_ts if thread_ts != event['ts'] else None
                        )
                    else:
                        response_text = func(**command_analysis.get('parameters', {}))
                else:
                    response_text = "I'm sorry, I don't know how to do that yet."
            
            # Skip sending message for delete_last command
            if command_analysis['function'] != 'delete_last':
                logger.info("Sending response to Slack")
                # Send response with unfurl_links=False to prevent link previews
                response = self.client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts if thread_ts != event['ts'] else None,
                    text=response_text,
                    unfurl_links=False
                )
                # Store the timestamp of this message
                channel_key = f"{channel}:{thread_ts if thread_ts else 'main'}"
                self.last_message_ts[channel_key] = response['ts']
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            if command_analysis['function'] != 'delete_last':
                self.client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts if thread_ts != event['ts'] else None,
                    text="I encountered an error processing your request.",
                    unfurl_links=False
                )

    def _chunk_messages(self, messages: List[Dict], chunk_size: int = 20) -> List[List[Dict]]:
        """Split messages into manageable chunks for the LLM."""
        return [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]

    def _summarize_chunk(self, messages: List[Dict], is_dm: bool = False, request_language: str = 'en') -> str:
        """
        Summarize a chunk of messages using LLM.
        
        Args:
            messages: List of messages to summarize
            is_dm: Whether this is a DM conversation
            request_language: Language to use for the summary ('en' or 'pl')
        """
        formatted_messages = []
        for msg in messages:
            user = msg.get('user_name', msg.get('user', 'unknown'))
            # Convert any user mentions in the text
            text = self._convert_user_mentions(msg.get('text', ''))
            ts = datetime.fromtimestamp(float(msg.get('ts', 0))).strftime('%Y-%m-%d %H:%M:%S')
            formatted_messages.append(f"[{ts}] {user}: {text}")

        # Choose the appropriate section format based on request language
        if request_language == 'pl':
            sections = (
                "*• Omówione Pomysły i Koncepcje:*\n"
                "  - Lista każdego unikalnego pomysłu z krótkim wyjaśnieniem\n"
                "  - Dla ważnych pomysłów, wskaż kto je zaproponował (np. '*Jan* zasugerował...')\n"
                "*• Decyzje i Wnioski:*\n"
                "  - Zanotuj podjęte decyzje i uzgodnione punkty\n"
                "  - Uwzględnij kto podjął lub poparł kluczowe decyzje\n"
                "*• Pytania i Punkty Dyskusji:*\n"
                "  - Lista zadanych pytań i kluczowych punktów dyskusji\n"
                "  - Zaznacz kto zadał ważne pytania lub wyraził obawy\n"
                "*• Zadania do Wykonania:*\n"
                "  - Zanotuj wspomniane zadania i follow-upy\n"
                "  - Uwzględnij kto jest odpowiedzialny lub zasugerował dane zadanie"
            )
        else:
            sections = (
                "*• Ideas & Concepts Discussed:*\n"
                "  - List each distinct idea with a brief explanation\n"
                "  - For significant ideas, include who proposed them (e.g., '*John* suggested...')\n"
                "*• Decisions & Conclusions:*\n"
                "  - Note any decisions made or points agreed upon\n"
                "  - Include who made or supported key decisions when relevant\n"
                "*• Questions & Discussion Points:*\n"
                "  - List questions raised and key points of discussion\n"
                "  - Note who raised important questions or concerns\n"
                "*• Action Items:*\n"
                "  - Note any tasks or follow-ups mentioned\n"
                "  - Include who is responsible for or suggested each action item"
            )

        system_content = (
            "You are Ray, Tasteray's intelligent assistant. Your task is to analyze conversations "
            "and identify ALL distinct ideas, concepts, and points of discussion. "
            f"Provide the summary in {'Polish' if request_language == 'pl' else 'English'}.\n\n"
            "Guidelines for analysis:"
            "\n1. Identify EVERY unique idea or concept mentioned, no matter how small"
            "\n2. Note any decisions made or conclusions reached"
            "\n3. Capture questions raised and points of discussion"
            "\n4. Pay attention to technical details and specifications"
            "\n5. Note any action items or future plans"
            "\n6. When an idea or suggestion is significant, attribute it to its author"
            "\n\nStructure your summary with these sections:\n"
            f"{sections}"
        )

        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": (
                    "Analyze this conversation and identify ALL distinct ideas and concepts. "
                    "Be thorough and don't miss any details. When an idea or suggestion is significant, "
                    f"mention who proposed it. Provide the summary in {'Polish' if request_language == 'pl' else 'English'}:\n\n" + 
                    "\n".join(formatted_messages)
                )
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1500 if is_dm else 1000,  # Increased from 500/300
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to summarize chunk: {e}")
            return ""

    def _summarize_channel(self, channel: str, oldest: str = None, latest: str = None, request_language: str = 'en') -> str:
        """
        Summarize channel history with specified time range.
        
        Args:
            channel: Channel ID to summarize
            oldest: Oldest timestamp to include
            latest: Latest timestamp to include
            request_language: Language to use for the summary ('en' or 'pl')
        """
        # Determine if this is a DM channel
        try:
            channel_info = self.client.conversations_info(channel=channel)
            is_dm = channel_info['channel']['is_im']
        except SlackApiError:
            is_dm = False
        
        logger.info(f"Summarizing {'DM' if is_dm else 'channel'} history in {request_language}")
        
        # Get channel history
        messages = self._get_channel_history(channel, oldest, latest)
        if not messages:
            return "No messages found in the specified time range." if request_language == 'en' else "Nie znaleziono wiadomości w podanym zakresie czasowym."

        # Sort messages by timestamp
        messages.sort(key=lambda x: float(x.get('ts', 0)))

        # Use appropriate chunk size based on channel type
        chunk_size = self.DM_CHUNK_SIZE if is_dm else self.CHANNEL_CHUNK_SIZE
        chunks = self._chunk_messages(messages, chunk_size=chunk_size)
        summaries = []

        # Process each chunk
        for i, chunk in enumerate(chunks, 1):
            summary = self._summarize_chunk(chunk, is_dm=is_dm, request_language=request_language)
            if summary:
                # Format each chunk summary with a timestamp range
                start_ts = datetime.fromtimestamp(float(chunk[0]['ts'])).strftime('%Y-%m-%d %H:%M')
                end_ts = datetime.fromtimestamp(float(chunk[-1]['ts'])).strftime('%Y-%m-%d %H:%M')
                summaries.append({
                    'period': f"{start_ts} to {end_ts}",
                    'summary': summary
                })

        # If we have multiple summaries, create a final summary
        if len(summaries) > 1:
            # Choose the appropriate section format based on request language
            if request_language == 'pl':
                sections = (
                    "*1. Kluczowe Decyzje i Rezultaty:*\n"
                    "   - Lista podjętych decyzji i osiągniętych wniosków\n"
                    "   - Uwzględnij kto podjął lub poparł kluczowe decyzje\n\n"
                    "*2. Główne Pomysły i Koncepcje:*\n"
                    "   - Lista omawianych pomysłów i koncepcji\n"
                    "   - Dla ważnych pomysłów, wskaż kto je zaproponował\n"
                    "   - Uwzględnij nawet drobne, ale istotne pomysły\n\n"
                    "*3. Zadania i Następne Kroki:*\n"
                    "   - Lista wspomnianych zadań i planów\n"
                    "   - Uwzględnij kto jest odpowiedzialny lub zaproponował dane zadanie\n\n"
                    "*4. Pytania i Otwarte Kwestie:*\n"
                    "   - Lista nierozwiązanych pytań i kwestii wymagających wyjaśnienia\n"
                    "   - Zaznacz kto zadał ważne pytania lub wyraził obawy"
                )
                instructions = (
                    "Bądź dokładny w identyfikowaniu *WSZYSTKICH* odrębnych pomysłów, nawet tych pozornie pobocznych. "
                    "Zachowaj informację o tym, kto jest autorem ważnych pomysłów i decyzji. "
                    "Użyj punktorów dla lepszej czytelności."
                )
            else:
                sections = (
                    "*1. Key Decisions & Outcomes:*\n"
                    "   - List each significant decision or conclusion reached\n"
                    "   - Include who made or supported key decisions when relevant\n\n"
                    "*2. Main Ideas & Concepts:*\n"
                    "   - List each unique idea or concept discussed\n"
                    "   - For significant ideas, include who proposed them\n"
                    "   - Include even minor ideas that might be relevant\n\n"
                    "*3. Action Items & Next Steps:*\n"
                    "   - List any tasks, follow-ups, or future plans mentioned\n"
                    "   - Include who is responsible for or suggested each action item\n\n"
                    "*4. Questions & Open Points:*\n"
                    "   - List any unresolved questions or points needing clarification\n"
                    "   - Note who raised important questions or concerns"
                )
                instructions = (
                    "Be thorough in identifying *ALL* distinct ideas, even if they seem tangential. "
                    "Maintain attribution of significant ideas and decisions to their authors. "
                    "Use bullet points for better readability."
                )

            final_summary_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are Ray, Tasteray's intelligent assistant. Your task is to create a comprehensive summary "
                        "that identifies and catalogs all distinct ideas, concepts, and decisions from the conversation. "
                        f"Provide the summary in {'Polish' if request_language == 'pl' else 'English'}.\n\n"
                        "Structure your response with these sections:\n\n"
                        f"{sections}\n\n"
                        f"{instructions}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Create a comprehensive summary from these conversation summaries, "
                        "identifying ALL distinct ideas and concepts:\n\n" +
                        "\n\n".join(s['summary'] for s in summaries)
                    )
                }
            ]

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=final_summary_prompt,
                    max_tokens=2000 if is_dm else 1500,  # Increased from DM_TOKEN_LIMIT/MENTION_TOKEN_LIMIT
                    temperature=0.7
                )
                final_summary = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Failed to create final summary: {e}")
                final_summary = None
        else:
            final_summary = summaries[0]['summary'] if summaries else None

        # Format the response using Slack's markdown
        response_parts = []
        
        # Add the main summary
        if final_summary:
            # Ensure proper Slack formatting for bold text
            formatted_summary = final_summary.replace('**', '*')  # Replace any markdown bold with Slack bold
            response_parts.append(formatted_summary)
        
        # Add chronological timeline if we have multiple chunks
        if len(summaries) > 1:
            timeline_header = "*Chronological Timeline*" if request_language == 'en' else "*Oś Czasowa*"
            response_parts.append(f"\n:clock1: {timeline_header}")
            for i, summary in enumerate(summaries, 1):
                response_parts.append(f"\n:small_blue_diamond: *{summary['period']}*")
                # Format the summary with proper indentation and Slack formatting
                formatted_lines = []
                for line in summary['summary'].split('\n'):
                    if line.strip():  # Skip empty lines
                        # Replace any markdown bold with Slack bold
                        line = line.replace('**', '*')
                        if line.startswith('•') or line.startswith('-'):
                            formatted_lines.append(f">{line}")
                        else:
                            formatted_lines.append(f">{line}")
                response_parts.append('\n'.join(formatted_lines))
        
        return '\n'.join(response_parts)

    def _initialize_home_tab(self):
        """Initialize the Home tab template."""
        self.home_tab_template = {
            "type": "home",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "👋 Welcome to Ray!",
                        "emoji": True
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*I'm your AI assistant for Tasteray. Here's what I can do:*"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "🔍 *Get News*\n• Fetch relevant news about streaming and movies\n• Customize date ranges and keywords\n• Get personalized insights"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "📝 *Summarize Channels*\n• Get summaries of any channel\n• Specify time ranges\n• Extract key insights"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "💬 *Chat & Assist*\n• Ask questions in DMs\n• Get help with tasks\n• Maintain conversation context"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Quick Start Commands:*"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "• `@Ray news` - Get latest news\n• `@Ray summarize this channel` - Summarize current channel\n• `@Ray help` - Show all commands"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "🤖 Ray is always learning and improving. Your feedback helps make me better!"
                        }
                    ]
                }
            ]
        }

    def publish_home_tab(self, user_id: str):
        """Publish the Home tab for a specific user."""
        try:
            # Get user's interaction stats
            stats = self._get_user_stats(user_id)
            
            # Create a copy of the template
            home_view = dict(self.home_tab_template)
            
            # Add user-specific stats if available
            if stats:
                stats_block = {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Your Activity:*\n• Messages: {stats['messages']}\n• Channels summarized: {stats['summaries']}\n• News queries: {stats['news_queries']}"
                    }
                }
                home_view["blocks"].insert(3, stats_block)
                home_view["blocks"].insert(4, {"type": "divider"})
            
            # Publish the view
            self.client.views_publish(
                user_id=user_id,
                view=home_view
            )
            logger.info(f"Published home tab for user {user_id}")
            
        except SlackApiError as e:
            logger.error(f"Error publishing home tab: {e}")

    def _get_user_stats(self, user_id: str) -> Optional[Dict]:
        """Get user interaction statistics."""
        # This is a placeholder - you might want to implement actual stats tracking
        return {
            "messages": 0,
            "summaries": 0,
            "news_queries": 0
        }

    def _create_clickup_task(self, 
                        list_id: str,
                        task_name: str,
                        status: str = None,
                        assignees: List[str] = None,
                        due_date: str = None,
                        priority: int = None,
                        description: str = None) -> str:
        """
        Create a task in ClickUp.
        
        Args:
            list_id: The ID of the ClickUp list (required)
            task_name: Name/title of the task
            status: Task status (e.g., 'Open', 'In Progress', 'Done')
            assignees: List of ClickUp user IDs to assign
            due_date: Due date in YYYY-MM-DD format
            priority: Priority level (1: Urgent, 2: High, 3: Normal, 4: Low)
            description: Task description
            
        Returns:
            Response message with task details and URL
        """
        logger.info(f"Creating ClickUp task: {task_name}")
        
        try:
            headers = {
                'Authorization': apis.CLICKUP_TOKEN,
                'Content-Type': 'application/json'
            }
            
            # Convert due date to timestamp if provided
            due_date_ts = None
            if due_date:
                try:
                    due_date_ts = int(datetime.strptime(due_date, '%Y-%m-%d').timestamp() * 1000)
                except ValueError:
                    return "Invalid due_date format. Please use YYYY-MM-DD format."
            
            # Prepare task data
            task_data = {
                'name': task_name,
                'description': description or '',
            }
            
            if status:
                task_data['status'] = status
            if assignees:
                task_data['assignees'] = assignees
            if due_date_ts:
                task_data['due_date'] = due_date_ts
            if priority:
                task_data['priority'] = priority
            
            # Create task using ClickUp API
            response = requests.post(
                f'https://api.clickup.com/api/v2/list/{list_id}/task',
                headers=headers,
                json=task_data
            )
            
            if response.status_code == 200:
                task = response.json()
                return (
                    f"✅ Task created successfully!\n"
                    f"*Task:* {task['name']}\n"
                    f"*Status:* {task.get('status', {}).get('status', 'Not set')}\n"
                    f"*URL:* {task['url']}"
                )
            else:
                error_msg = response.json().get('err', response.text)
                logger.error(f"ClickUp API error: {error_msg}")
                return f"Failed to create task. Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error creating ClickUp task: {e}")
            return f"Error creating task: {str(e)}"

    def _get_user_name(self, user_id: str) -> str:
        """Get user's display name from their ID."""
        try:
            if not user_id or user_id == 'unknown':
                return 'Unknown User'
            
            # Check if it's our bot
            if user_id == self.client.auth_test()['user_id']:
                return 'Ray'
                
            user_info = self.client.users_info(user=user_id)
            # Prefer display name, fall back to real name, then username
            return (
                user_info['user'].get('profile', {}).get('display_name') or
                user_info['user'].get('real_name') or
                user_info['user'].get('name') or
                'Unknown User'
            )
        except SlackApiError as e:
            logger.error(f"Error getting user info: {e}")
            return 'Unknown User'

    def _delete_last_message(self, channel: str = None, thread_ts: str = None) -> None:
        """Delete the last message sent by the bot in the specified channel/thread."""
        try:
            # Get the key for the last message timestamp
            channel_key = f"{channel}:{thread_ts if thread_ts else 'main'}"
            
            # Get the timestamp of the last message
            last_ts = self.last_message_ts.get(channel_key)
            
            if last_ts:
                # Delete the message
                self.client.chat_delete(
                    channel=channel,
                    ts=last_ts
                )
                # Remove the timestamp from tracking
                del self.last_message_ts[channel_key]
                logger.info(f"Deleted last message in {channel_key}")
            else:
                logger.warning(f"No last message found to delete in {channel_key}")
                
        except SlackApiError as e:
            logger.error(f"Error deleting last message: {e}")
            raise

    def _convert_user_mentions(self, text: str) -> str:
        """Convert user IDs in message text to display names."""
        if not text:
            return text
            
        # Match both direct mentions <@U1234567> and user references in text U1234567
        user_mentions = set(re.finditer(r'<@(U[A-Z0-9]+)>', text))
        user_mentions.update(re.finditer(r'\b(U[A-Z0-9]{8,})\b', text))
        new_text = text
        
        for match in user_mentions:
            user_id = match.group(1)
            user_name = self._get_user_name(user_id)
            # Replace both formats with the display name
            new_text = new_text.replace(f'<@{user_id}>', f'*{user_name}*')
            new_text = new_text.replace(user_id, f'*{user_name}*')
        
        return new_text