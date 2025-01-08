# bot_handler.py

import json
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import apis
from news_collector import main_function
from typing import List, Dict, Optional, Tuple, Any
from openai import OpenAI
import logging
import re
import requests
import aiohttp
import asyncio
import unicodedata

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
        """Initialize basic bot attributes."""
        from slack_sdk.web.async_client import AsyncWebClient
        self.client = AsyncWebClient(token=apis.SLACK_TOKEN)
        self.last_message_ts = {}
        self.bot_id = None
        self.channel_cache = {}  # Initialize channel cache
        self.user_cache = {}  # Cache for ClickUp user IDs
        
        self.functions = {
            'news': self._get_news,
            'help': self._get_help,
            'summarize': self._get_summary,
            'task': self._create_clickup_task,
            'delete_last': self._delete_last_message,
        }
        
        # Context limits
        self.DM_CONTEXT_LIMIT = 15
        self.MENTION_CONTEXT_LIMIT = 8
        # Chunk sizes for summarization
        self.DM_CHUNK_SIZE = 10
        self.CHANNEL_CHUNK_SIZE = 25
        # Token limits - significantly increased for long summaries
        self.DM_TOKEN_LIMIT = 4000  # Increased from 2000
        self.MENTION_TOKEN_LIMIT = 3000  # Increased from 1000
        # Maximum chunks per timeline section
        self.MAX_CHUNKS_PER_SECTION = 10
        
        # Initialize Home tab template
        self._initialize_home_tab()
        
    async def initialize(self):
        """Async initialization of the bot."""
        try:
            # Get bot's user ID
            auth_response = await self.client.auth_test()
            self.bot_id = auth_response['user_id']
            logger.info(f"Bot initialized with ID: {self.bot_id}")
            
            # Initialize workspace access
            await self._initialize_workspace_access()
            
            # Initialize ClickUp user cache
            await self._initialize_clickup_users()
            
        except SlackApiError as e:
            logger.error(f"Error during bot initialization: {e}")
            raise

    async def _initialize_workspace_access(self):
        """Initialize workspace access by joining all accessible channels."""
        try:
            print("\nInitializing workspace access...")
            # Get list of all public channels
            response = await self.client.conversations_list(
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
                        await self.client.conversations_join(
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

    async def _initialize_clickup_users(self):
        """Initialize ClickUp user cache by fetching actual user IDs."""
        url = f"https://api.clickup.com/api/v2/team/{apis.CLICKUP_TEAM_ID}"
        headers = {
            "Authorization": apis.CLICKUP_TOKEN,
            "Content-Type": "application/json"
        }
        
        def normalize_name(name):
            """Normalize name by removing diacritical marks and converting to lowercase."""
            # Normalize unicode characters and remove diacritical marks
            normalized = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
            # Convert to lowercase and remove spaces
            return normalized.lower().replace(' ', '')
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        team = data.get('team', {})
                        members = team.get('members', [])
                        
                        logger.info(f"Found {len(members)} members in ClickUp team")
                        
                        # Build user cache
                        for member in members:
                            user = member.get('user', {})
                            user_id = user.get('id')
                            email = user.get('email', '').lower()
                            username = user.get('username', '')
                            
                            logger.info(f"Processing user: {username} (ID: {user_id}, Email: {email})")
                            
                            if user_id and (email or username):
                                # Cache by email
                                if email:
                                    self.user_cache[email] = user_id
                                    logger.debug(f"Cached by email: {email} -> {user_id}")
                                
                                # Cache by username
                                if username:
                                    # Cache original username
                                    self.user_cache[username.lower()] = user_id
                                    logger.debug(f"Cached by original username: {username.lower()} -> {user_id}")
                                    
                                    # Cache normalized username (without spaces)
                                    normalized = normalize_name(username)
                                    self.user_cache[normalized] = user_id
                                    logger.debug(f"Cached by normalized username: {normalized} -> {user_id}")
                                    
                                    # Also cache common name variations
                                    name_parts = username.split()
                                    if len(name_parts) > 1:
                                        # First name variations
                                        first_name = name_parts[0]
                                        self.user_cache[first_name.lower()] = user_id
                                        self.user_cache[normalize_name(first_name)] = user_id
                                        logger.debug(f"Cached by first name: {first_name.lower()} -> {user_id}")
                                        
                                        # Last name variations
                                        last_name = name_parts[-1]
                                        self.user_cache[last_name.lower()] = user_id
                                        self.user_cache[normalize_name(last_name)] = user_id
                                        logger.debug(f"Cached by last name: {last_name.lower()} -> {user_id}")
                                        
                                        # Full name variations (with and without spaces)
                                        full_name = ' '.join(name_parts)
                                        self.user_cache[full_name.lower()] = user_id
                                        self.user_cache[normalize_name(full_name)] = user_id
                                        logger.debug(f"Cached by full name: {full_name.lower()} -> {user_id}")
                        
                        logger.info(f"Initialized ClickUp user cache with {len(self.user_cache)} entries")
                        logger.info("User cache contents:")
                        for key, value in sorted(self.user_cache.items()):
                            logger.info(f"  {key} -> {value}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Error fetching ClickUp users: {error_text}")
                        
        except Exception as e:
            logger.error(f"Error initializing ClickUp users: {e}")

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

    async def _analyze_command(self, text: str, sender_id: str = None) -> dict:
        """Use LLM to analyze the command and determine the response."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get sender's info for self-references
        sender_name = None
        sender_email = None
        if sender_id:
            try:
                response = await self.client.users_info(user=sender_id)
                user_info = response['user']
                sender_name = user_info.get('real_name', '')
                sender_email = user_info.get('profile', {}).get('email', '')
                logger.info(f"Got sender info: name={sender_name}, email={sender_email}")
            except SlackApiError as e:
                logger.error(f"Error getting sender info: {e}")
        
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are Jane, an intelligent internal assistant for Tasteray - an AI-powered movie recommendation platform. "
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
                    "   - list_name: Name of the list (e.g., 'o:produkt')\n"
                    "   - task_name: Name of the task (required)\n"
                    "   - status: Task status (always in English: 'Open', 'In Progress', 'Done')\n"
                    "   - assignees: List of people to assign (can be Slack mentions, names, or emails)\n"
                    "   - due_date: Due date (YYYY-MM-DD)\n"
                    "   - priority: Priority level (1: Urgent, 2: High, 3: Normal, 4: Low)\n"
                    "   - description: Task description\n\n"
                    "4. help/pomoc: Show available commands\n"
                    "5. delete_last/usuń ostatnią: Delete last message\n"
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
                    "8. IMPORTANT: Always use English for function names and parameters\n"
                    "9. For tasks, try to infer assignees from context when not explicitly specified\n"
                    "10. Be forgiving with typos and informal references\n\n"
                    f"Current User Info:\n"
                    f"- Name: {sender_name}\n"
                    f"- Email: {sender_email}\n"
                )
            },
            {
                "role": "user",
                "content": text
            }
        ]

        logger.info(f"Analyzing message: {text}")
        
        try:
            print("\nSending request to OpenAI...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
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

    async def _get_help(self) -> str:
        """Get help text with available commands."""
        return (
            "*Available Commands:*\n\n"
            "*1. News/Wiadomości:*\n"
            "• `news` - Get latest news\n"
            "• `news from 2023-12-01 to 2023-12-31` - Get news for specific date range\n"
            "• `news about AI and movies` - Get news about specific topics\n\n"
            "*2. Summarize/Podsumuj:*\n"
            "• `summarize` - Summarize current channel\n"
            "• `summarize #general` - Summarize specific channel\n"
            "• `summarize from 2023-12-01` - Summarize from date\n"
            "• `summarize to 2023-12-31` - Summarize until date\n\n"
            "*3. Tasks/Zadania:*\n"
            "• `task Create example task` - Create task in default list\n"
            "• `task Create bug report list: bugtracker-www` - Create task in specific list\n"
            "• `task Review PR for @user due: 2024-01-15` - Create task with due date\n"
            "• `task Urgent feature request priority: 1` - Create high priority task\n\n"
            "*4. Other Commands:*\n"
            "• `help` - Show this help message\n"
            "• `delete_last` - Delete my last message\n\n"
            "*Notes:*\n"
            "• You can use commands in English or Polish\n"
            "• For tasks, you can specify: list, assignees, due date, priority, status\n"
            "• For private channels, invite me first using `/invite @Jane`"
        )
        
    async def _get_news(self, from_date: str = None, to_date: str = None, keywords: List[str] = None, articles_per_keyword: int = 3) -> str:
        """Get news articles based on specified criteria."""
        # This is a placeholder - implement actual news gathering logic
        return "News gathering not implemented yet."
        
    async def _get_summary(self, channel: str = None, from_date: str = None, to_date: str = None) -> str:
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
        channel_info = await self._resolve_channel_reference(channel)
        if not channel_info:
            return (
                "Could not access channel: {channel}. For private channels, please invite me first using `/invite @Jane`"
                if request_language == 'en' else
                f"Nie mogę uzyskać dostępu do kanału: {channel}. W przypadku kanałów prywatnych, najpierw zaproś mnie używając `/invite @Jane`"
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

        summary = await self._summarize_channel(channel_id, oldest, latest, request_language)
        
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

    async def handle_message(self, text: str, channel: str, thread_ts: str = None, sender_id: str = None, is_mention: bool = False) -> None:
        """
        Handle a message sent to the bot.
        
        Args:
            text: The message text
            channel: The channel ID where the message was sent
            thread_ts: The thread timestamp if the message is in a thread
            sender_id: The Slack user ID of the message sender
            is_mention: Whether the message was a direct mention of the bot
        """
        try:
            # Validate input
            if not text or not channel:
                logger.error(f"Invalid message: text='{text}', channel='{channel}'")
                return
                
            # Get command analysis from LLM
            analysis = await self._analyze_command(text, sender_id)
            logger.info(f"Command analysis: {analysis}")
            
            if not analysis:
                return
            
            function = analysis.get('function')
            parameters = analysis.get('parameters', {})
            response = analysis.get('response')
            
            if function == 'task':
                # Create task in ClickUp
                task_result = await self._create_clickup_task(
                    list_name=parameters.get('list_name', 'o:produkt'),
                    task_name=parameters.get('task_name'),
                    status=parameters.get('status'),
                    assignees=parameters.get('assignees'),
                    due_date=parameters.get('due_date'),
                    priority=parameters.get('priority'),
                    description=parameters.get('description')
                )
                
                # Format the response
                list_name = parameters.get('list_name', 'o:produkt')
                status_text = task_result.get('status', {}).get('status', 'Not set')
                priority_text = task_result.get('priority', {}).get('priority', 'Not set')
                
                # Format assignee names
                assignee_names = []
                if parameters.get('assignees'):
                    for assignee_id in parameters['assignees']:
                        user_info = apis.CLICKUP_USER_INFO.get(assignee_id)
                        if user_info:
                            assignee_names.append(user_info['name'])
                
                response_text = (
                    f"✅ Task created successfully!\n"
                    f"*Task:* {task_result['name']}\n"
                    f"*List:* {list_name}\n"
                    f"*Status:* {status_text}\n"
                    f"*Priority:* {priority_text}\n"
                )
                
                if assignee_names:
                    response_text += f"*Assigned to:* {', '.join(assignee_names)}\n"
                
                response_text += f"*URL:* {task_result['url']}"
                
                await self._post_message(channel, response_text, thread_ts)
                
            elif function == 'summarize':
                # Handle channel summary
                summary = await self._get_summary(
                    channel=parameters.get('channel', channel),
                    from_date=parameters.get('from_date'),
                    to_date=parameters.get('to_date')
                )
                await self._post_message(channel, summary, thread_ts)
                
            elif function == 'news':
                # Handle news request
                news = await self._get_news(
                    from_date=parameters.get('from_date'),
                    to_date=parameters.get('to_date'),
                    keywords=parameters.get('keywords', []),
                    articles_per_keyword=parameters.get('articles_per_keyword', 3)
                )
                await self._post_message(channel, news, thread_ts)
                
            elif function == 'help':
                # Show help message
                help_text = await self._get_help()
                await self._post_message(channel, help_text, thread_ts)
                
            elif function == 'delete_last':
                # Delete last message
                await self._delete_last_message(channel, thread_ts)
                
            elif function == 'direct_response':
                # Just post the response
                if response:
                    await self._post_message(channel, response, thread_ts)
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            logger.error(f"Error handling message: {e}")
            await self._post_message(channel, error_message, thread_ts)

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
                "> - Lista każdego unikalnego pomysłu z krótkim wyjaśnieniem\n"
                "> - Dla ważnych pomysłów, wskaż kto je zaproponował (np. *Jan* zasugerował...)\n"
                "*• Decyzje i Wnioski:*\n"
                "> - Zanotuj podjęte decyzje i uzgodnione punkty\n"
                "> - Uwzględnij kto podjął lub poparł kluczowe decyzje\n"
                "*• Pytania i Punkty Dyskusji:*\n"
                "> - Lista zadanych pytań i kluczowych punktów dyskusji\n"
                "> - Zaznacz kto zadał ważne pytania lub wyraził obawy\n"
                "*• Zadania do Wykonania:*\n"
                "> - Zanotuj wspomniane zadania i follow-upy\n"
                "> - Uwzględnij kto jest odpowiedzialny lub zasugerował dane zadanie"
            )
        else:
            sections = (
                "*• Ideas & Concepts Discussed:*\n"
                "> - List each distinct idea with a brief explanation\n"
                "> - For significant ideas, include who proposed them (e.g., *John* suggested...)\n"
                "*• Decisions & Conclusions:*\n"
                "> - Note any decisions made or points agreed upon\n"
                "> - Include who made or supported key decisions when relevant\n"
                "*• Questions & Discussion Points:*\n"
                "> - List questions raised and key points of discussion\n"
                "> - Note who raised important questions or concerns\n"
                "*• Action Items:*\n"
                "> - Note any tasks or follow-ups mentioned\n"
                "> - Include who is responsible for or suggested each action item"
            )

        system_content = (
            "You are Jane, Tasteray's intelligent assistant. Your task is to analyze conversations "
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
                    "> - Lista podjętych decyzji i osiągniętych wniosków\n"
                    "> - Uwzględnij kto podjął lub poparł kluczowe decyzje\n\n"
                    "*2. Główne Pomysły i Koncepcje:*\n"
                    "> - Lista omawianych pomysłów i koncepcji\n"
                    "> - Dla ważnych pomysłów, wskaż kto je zaproponował\n"
                    "> - Uwzględnij nawet drobne, ale istotne pomysły\n\n"
                    "*3. Zadania i Następne Kroki:*\n"
                    "> - Lista wspomnianych zadań i planów\n"
                    "> - Uwzględnij kto jest odpowiedzialny lub zaproponował dane zadanie\n\n"
                    "*4. Pytania i Otwarte Kwestie:*\n"
                    "> - Lista nierozwiązanych pytań i kwestii wymagających wyjaśnienia\n"
                    "> - Zaznacz kto zadał ważne pytania lub wyraził obawy"
                )
                instructions = (
                    "Bądź dokładny w identyfikowaniu *WSZYSTKICH* odrębnych pomysłów, nawet tych pozornie pobocznych. "
                    "Zachowaj informację o tym, kto jest autorem ważnych pomysłów i decyzji. "
                    "Używaj '>' na początku każdej linii dla lepszej czytelności."
                )
            else:
                sections = (
                    "*1. Key Decisions & Outcomes:*\n"
                    "> - List each significant decision or conclusion reached\n"
                    "> - Include who made or supported key decisions when relevant\n\n"
                    "*2. Main Ideas & Concepts:*\n"
                    "> - List each unique idea or concept discussed\n"
                    "> - For significant ideas, include who proposed them\n"
                    "> - Include even minor ideas that might be relevant\n\n"
                    "*3. Action Items & Next Steps:*\n"
                    "> - List any tasks, follow-ups, or future plans mentioned\n"
                    "> - Include who is responsible for or suggested each action item\n\n"
                    "*4. Questions & Open Points:*\n"
                    "> - List any unresolved questions or points needing clarification\n"
                    "> - Note who raised important questions or concerns"
                )
                instructions = (
                    "Be thorough in identifying *ALL* distinct ideas, even if they seem tangential. "
                    "Maintain attribution of significant ideas and decisions to their authors. "
                    "Use '>' at the start of each line for better readability."
                )

            # Split summaries into sections if there are too many
            timeline_sections = []
            for i in range(0, len(summaries), self.MAX_CHUNKS_PER_SECTION):
                section_summaries = summaries[i:i + self.MAX_CHUNKS_PER_SECTION]
                section_start = section_summaries[0]['period'].split(' to ')[0]
                section_end = section_summaries[-1]['period'].split(' to ')[1]
                
                final_summary_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are Jane, Tasteray's intelligent assistant. Your task is to create a comprehensive summary "
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
                            "\n\n".join(s['summary'] for s in section_summaries)
                        )
                    }
                ]

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=final_summary_prompt,
                        max_tokens=3000,  # Increased token limit for section summaries
                        temperature=0.7
                    )
                    section_summary = response.choices[0].message.content.strip()
                    timeline_sections.append({
                        'period': f"{section_start} to {section_end}",
                        'summary': section_summary
                    })
                except Exception as e:
                    logger.error(f"Failed to create section summary: {e}")
                    continue

            # Create overall summary from section summaries
            if timeline_sections:
                try:
                    overall_summary_prompt = [
                        {
                            "role": "system",
                            "content": (
                                "Create a high-level summary of the entire conversation period, "
                                "focusing on the most important developments, decisions, and trends. "
                                f"Provide the summary in {'Polish' if request_language == 'pl' else 'English'}"
                            )
                        },
                        {
                            "role": "user",
                            "content": "\n\n".join(s['summary'] for s in timeline_sections)
                        }
                    ]
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=overall_summary_prompt,
                        max_tokens=4000,  # Maximum tokens for overall summary
                        temperature=0.7
                    )
                    final_summary = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Failed to create final summary: {e}")
                    final_summary = timeline_sections[0]['summary']
            else:
                final_summary = None
        else:
            final_summary = summaries[0]['summary'] if summaries else None
            timeline_sections = []

        # Format the response using Slack's markdown
        response_parts = []
        
        # Add the main summary
        if final_summary:
            response_parts.append(self._format_summary_section(final_summary))
        
        # Add chronological timeline if we have multiple sections
        if timeline_sections:
            timeline_header = "*Chronological Timeline*" if request_language == 'en' else "*Oś Czasowa*"
            response_parts.append(f"\n:clock1: {timeline_header}")
            
            for section in timeline_sections:
                response_parts.append(f"\n:calendar: *{section['period']}*")
                formatted_section = self._format_summary_section(section['summary'])
                response_parts.append(formatted_section)
                response_parts.append("\n─────────────────────\n")
        
        # If we have individual summaries and they're not too many, add them as well
        elif len(summaries) > 1 and len(summaries) <= self.MAX_CHUNKS_PER_SECTION:
            timeline_header = "*Detailed Timeline*" if request_language == 'en' else "*Szczegółowa Oś Czasowa*"
            response_parts.append(f"\n:clock1: {timeline_header}")
            
            for summary in summaries:
                response_parts.append(f"\n:small_blue_diamond: *{summary['period']}*")
                formatted_section = self._format_summary_section(summary['summary'])
                response_parts.append(formatted_section)
        
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
                        "text": "👋 Welcome to Jane!",
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
                        "text": "• `@Jane news` - Get latest news\n• `@Jane summarize this channel` - Summarize current channel\n• `@Jane help` - Show all commands"
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
                            "text": "🤖 Jane is always learning and improving. Your feedback helps make me better!"
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

    def _resolve_user_reference(self, user_ref: str, sender_id: str = None) -> Optional[str]:
        """
        Resolve a user reference (mention, name, or email) to a ClickUp user ID.
        Handles typos and variations in names.
        
        Args:
            user_ref: The user reference (mention, name, email, or self-reference)
            sender_id: The Slack user ID of the message sender (for resolving self-references)
            
        Returns:
            ClickUp user ID or None if no match is found
        """
        try:
            # Handle self-references if sender_id is provided
            self_refs = ['me', 'my', 'i', 'mnie', 'mi', 'ja', 'moje', 'moim', 'mną']
            if sender_id and user_ref.lower() in self_refs:
                try:
                    user_info = self.client.users_info(user=sender_id)
                    email = user_info['user'].get('profile', {}).get('email', '').lower()
                    if email in apis.CLICKUP_USERS:
                        return apis.CLICKUP_USERS[email]
                except SlackApiError:
                    pass
            
            # If it's a Slack mention, extract the user ID and get their email
            if user_ref.startswith('<@') and user_ref.endswith('>'):
                user_id = user_ref[2:-1]
                try:
                    user_info = self.client.users_info(user=user_id)
                    email = user_info['user'].get('profile', {}).get('email', '').lower()
                    if email in apis.CLICKUP_USERS:
                        return apis.CLICKUP_USERS[email]
                except SlackApiError:
                    pass

            # Try direct lookup in CLICKUP_USERS
            user_ref_lower = user_ref.lower().strip()
            if user_ref_lower in apis.CLICKUP_USERS:
                return apis.CLICKUP_USERS[user_ref_lower]
            
            # Map of informal names, typos, and variations
            name_mappings = {
                # Stanisław variations
                'stasiek': 'stanislaw',
                'staszek': 'stanislaw',
                'stachu': 'stanislaw',
                'stach': 'stanislaw',
                'stanisław': 'stanislaw',
                'stanislaw': 'stanislaw',
                'stan': 'stanislaw',
                'stas': 'stanislaw',
                
                # Michał Górecki variations
                'goo': 'michal gorecki',
                'gorecki': 'michal gorecki',
                'górecki': 'michal gorecki',
                'goorecki': 'michal gorecki',
                'gurecki': 'michal gorecki',
                
                # Konrad variations
                'konrad': 'konrad traczyk',
                'konradem': 'konrad traczyk',
                'konradek': 'konrad traczyk',
                'traczyk': 'konrad traczyk',
                
                # Kordian variations
                'kordian': 'kordian klecha',
                'kordianem': 'kordian klecha',
                'kordi': 'kordian klecha',
                'klecha': 'kordian klecha',
                
                # Iwo variations
                'iwo': 'iwo tuleya',
                'tuleya': 'iwo tuleya',
                'iwem': 'iwo tuleya',
                
                # Mariusz variations
                'mariusz': 'mariusz labudda',
                'labudda': 'mariusz labudda',
                'mariuszem': 'mariusz labudda',
                'mariu': 'mariusz labudda',
                
                # Michał Jaskólski variations
                'jaskolski': 'michal jaskolski',
                'jaskólski': 'michal jaskolski',
                'jaskulski': 'michal jaskolski',
                
                # Michał Wolniak variations
                'wolniak': 'michal wolniak',
                'wolny': 'michal wolniak'
            }
            
            # Check name mappings
            if user_ref_lower in name_mappings:
                mapped_name = name_mappings[user_ref_lower]
                if mapped_name in apis.CLICKUP_USERS:
                    return apis.CLICKUP_USERS[mapped_name]
            
            # Try fuzzy matching for close typos
            for ref, mapped_name in name_mappings.items():
                # Check if the reference is very similar to any mapping
                if (
                    # Same first 3 letters (common abbreviation)
                    (len(user_ref_lower) >= 3 and len(ref) >= 3 and user_ref_lower[:3] == ref[:3]) or
                    # One character difference
                    (abs(len(user_ref_lower) - len(ref)) <= 1 and sum(1 for a, b in zip(user_ref_lower, ref) if a != b) <= 1) or
                    # Contains the main part of the name
                    (len(user_ref_lower) >= 4 and (user_ref_lower in ref or ref in user_ref_lower))
                ):
                    if mapped_name in apis.CLICKUP_USERS:
                        return apis.CLICKUP_USERS[mapped_name]

            return None
        except Exception as e:
            logger.error(f"Error resolving user reference: {e}")
            return None

    def _resolve_list_reference(self, text: str) -> Optional[str]:
        """
        Resolve a list reference from natural language to a list name.
        Handles typos and variations in list names.
        Returns the list name or None if no match is found.
        """
        # Map of common references and typos to list names
        list_mappings = {
            # Backlog variations
            'backlog': 'deep-backlog',
            'deep backlog': 'deep-backlog',
            'backl': 'deep-backlog',
            'baklog': 'deep-backlog',
            'baclog': 'deep-backlog',
            
            # Product variations
            'product': 'o:produkt',
            'produkt': 'o:produkt',
            'prod': 'o:produkt',
            'produktu': 'o:produkt',  # Polish case
            'produkcie': 'o:produkt',  # Polish case
            
            # Growth variations
            'growth': 'o:growth',
            'growt': 'o:growth',
            'grow': 'o:growth',
            'groth': 'o:growth',
            
            # Bug tracker variations
            'bugs': 'bugtracker-www',
            'bug': 'bugtracker-www',
            'website bugs': 'bugtracker-www',
            'www bugs': 'bugtracker-www',
            'web bugs': 'bugtracker-www',
            'website': 'bugtracker-www',
            'www': 'bugtracker-www',
            'bugi': 'bugtracker-www',  # Polish
            'błędy': 'bugtracker-www',  # Polish
            'bledy': 'bugtracker-www',  # Polish without diacritics
            
            # App bug tracker variations
            'app bugs': 'bugtracker-app',
            'mobile bugs': 'bugtracker-app',
            'app': 'bugtracker-app',
            'mobile': 'bugtracker-app',
            'aplikacja': 'bugtracker-app',  # Polish
            'apka': 'bugtracker-app',  # Polish informal
            
            # Operations variations
            'operations': 'operacyjna-back',
            'ops': 'operacyjna-back',
            'operation': 'operacyjna-back',
            'operational': 'operacyjna-back',
            'operacyjny': 'operacyjna-back',  # Polish
            'operacyjne': 'operacyjna-back',  # Polish
            'operacje': 'operacyjna-back',  # Polish
            
            # Insights variations
            'insights': 'insights',
            'insight': 'insights',
            'insighty': 'insights',  # Polish
            'insajty': 'insights'  # Polish phonetic
        }
        
        text_lower = text.lower()
        
        # First check for exact list name matches
        for list_name in apis.CLICKUP_LISTS.keys():
            if list_name.lower() in text_lower:
                return list_name
        
        # Then check for variations and typos
        for ref, list_name in list_mappings.items():
            if ref in text_lower:
                return list_name
        
        # If no exact match, try fuzzy matching for close typos
        words = text_lower.split()
        for word in words:
            for ref, list_name in list_mappings.items():
                # Check if the word is very similar to any reference
                if (
                    # Same first 3 letters (common abbreviation)
                    (len(word) >= 3 and len(ref) >= 3 and word[:3] == ref[:3]) or
                    # One character difference
                    (abs(len(word) - len(ref)) <= 1 and sum(1 for a, b in zip(word, ref) if a != b) <= 1) or
                    # Contains the main part of the word
                    (len(word) >= 4 and word in ref or ref in word)
                ):
                    return list_name
                
        return None

    def _parse_task_command(self, text: str, sender_id: str = None) -> Dict[str, Any]:
        """
        Parse task command text to extract task parameters.
        Relies on LLM's natural language understanding for resolving references.
        
        Args:
            text: The command text to parse
            sender_id: The Slack user ID of the message sender (for resolving self-references)
        """
        # Remove the command prefix if present
        text = re.sub(r'^task\s+', '', text.strip())
        
        # Get sender's info for self-references
        sender_name = None
        sender_email = None
        if sender_id:
            try:
                user_info = self.client.users_info(user=sender_id)
                sender_name = user_info['user'].get('real_name', '')
                sender_email = user_info['user'].get('profile', {}).get('email', '')
            except SlackApiError as e:
                logger.error(f"Error getting sender info: {e}")
        
        # Initialize parameters
        params = {
            'task_name': '',
            'list_name': None,
            'status': None,
            'assignees': None,
            'due_date': None,
            'priority': None,
            'description': None
        }
        
        # First try to parse formal command format
        param_matches = re.finditer(r'(list|status|assign|due|priority|description):\s*([^:]+?)(?=\s+\w+:|$)', text)
        param_positions = [(m.start(), m.group(1), m.group(2).strip()) for m in param_matches]
        
        if param_positions:
            # Formal command format detected
            first_param_pos = param_positions[0][0]
            params['task_name'] = text[:first_param_pos].strip()
            
            # Process each parameter
            for _, param_name, param_value in param_positions:
                if param_name == 'list':
                    params['list_name'] = param_value
                elif param_name == 'status':
                    params['status'] = param_value
                elif param_name == 'assign':
                    # Handle self-references
                    if any(ref.lower() in ['me', 'my', 'i', 'ja', 'mnie', 'mi', 'mój', 'moja', 'moje'] for ref in param_value.split(',')):
                        if sender_email:
                            params['assignees'] = [sender_email]
                    else:
                        params['assignees'] = [ref.strip() for ref in param_value.split(',')]
                elif param_name == 'due':
                    params['due_date'] = param_value
                elif param_name == 'priority':
                    params['priority'] = param_value
                elif param_name == 'description':
                    params['description'] = param_value
        else:
            # For informal requests, let the LLM handle everything
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a task parser that extracts structured information from informal task requests. "
                        "Return your response in JSON format.\n\n"
                        "Available information about the sender:\n"
                        f"- Name: {sender_name}\n"
                        f"- Email: {sender_email}\n\n"
                        "When the sender refers to themselves ('me', 'my', 'I', etc.), "
                        f"use their email: {sender_email}\n\n"
                        "Return JSON with these fields:\n"
                        "1. task_name: The name/title of the task\n"
                        "2. list_name: The target list (if mentioned)\n"
                        "3. assignees: Array of emails for people to assign\n"
                        "4. status: Task status (if mentioned)\n"
                        "5. priority: Priority level (if mentioned)\n"
                        "6. due_date: Due date (if mentioned)\n"
                        "7. description: Task description (if provided)\n\n"
                        "Example response for 'create a task for me':\n"
                        "{\n"
                        "  \"task_name\": \"New task\",\n"
                        f"  \"assignees\": [\"{sender_email}\"],\n"
                        "  \"status\": \"Open\"\n"
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.1,
                    response_format={ "type": "json_object" }
                )
                
                parsed = json.loads(response.choices[0].message.content)
                logger.info(f"LLM parsed task: {parsed}")
                
                # Update params with the LLM's understanding
                if 'task_name' in parsed:
                    params['task_name'] = parsed['task_name']
                if 'list_name' in parsed:
                    params['list_name'] = parsed['list_name']
                if 'assignees' in parsed:
                    params['assignees'] = parsed['assignees']
                if 'status' in parsed:
                    params['status'] = parsed['status']
                if 'priority' in parsed:
                    params['priority'] = parsed['priority']
                if 'due_date' in parsed:
                    params['due_date'] = parsed['due_date']
                if 'description' in parsed:
                    params['description'] = parsed['description']
                    
            except Exception as e:
                logger.error(f"Error parsing informal task request: {e}")
                # Fall back to using the entire text as task name
                params['task_name'] = text
        
        # If no task name was set, use the entire text
        if not params['task_name']:
            params['task_name'] = text
            
        # Ensure sender is included when they reference themselves
        if sender_email and params['assignees']:
            if isinstance(params['assignees'], list):
                if any(ref.lower() in ['me', 'my', 'i', 'ja', 'mnie', 'mi', 'mój', 'moja', 'moje'] for ref in params['assignees']):
                    if sender_email not in params['assignees']:
                        params['assignees'].append(sender_email)
            
        return params

    async def _get_list_statuses(self, list_id: str) -> List[str]:
        """Get available statuses for a ClickUp list."""
        url = f"https://api.clickup.com/api/v2/list/{list_id}"
        headers = {
            "Authorization": apis.CLICKUP_TOKEN,
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [status['status'] for status in data.get('statuses', [])]
                    else:
                        error_text = await response.text()
                        logger.error(f"Error getting list statuses: {error_text}")
                        return []
        except Exception as e:
            logger.error(f"Error getting list statuses: {e}")
            return []

    async def _create_clickup_task(self, list_name: str, task_name: str, status: str = None,
                               assignees: List[str] = None, due_date: str = None,
                               priority: int = None, description: str = None) -> Dict[str, Any]:
        """Create a task in ClickUp."""
        try:
            # Let the LLM match the list name to the correct one
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a list matcher that maps informal list references to their correct ClickUp list names. "
                        "Return your response in JSON format.\n\n"
                        "Available ClickUp lists:\n"
                        "- o:produkt: Main product development list\n"
                        "- o:growth: Growth and marketing initiatives\n"
                        "- bugtracker-www: Website bugs and issues\n"
                        "- bugtracker-app: Mobile app bugs and issues\n"
                        "- deep-backlog: General backlog for future tasks\n"
                        "- operacyjna-back: Backend operations tasks\n"
                        "- insights: Data and analytics tasks\n\n"
                        "Common variations:\n"
                        "- produkt, product, produktu → o:produkt\n"
                        "- growth, wzrost → o:growth\n"
                        "- bugs, website bugs, www → bugtracker-www\n"
                        "- app bugs, mobile → bugtracker-app\n"
                        "- backlog, deep backlog → deep-backlog\n"
                        "- operations, ops, backend → operacyjna-back\n"
                        "- insights, analityka → insights\n\n"
                        "Return JSON with a single field 'list_name' containing the matched list name.\n"
                        "If no match is found, use 'o:produkt' as the default.\n\n"
                        "Example response:\n"
                        "{\n"
                        "  \"list_name\": \"o:produkt\"\n"
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": list_name
                }
            ]
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=100,
                    temperature=0.1,
                    response_format={ "type": "json_object" }
                )
                
                parsed = json.loads(response.choices[0].message.content)
                matched_list = parsed.get('list_name', 'o:produkt')
                logger.info(f"LLM matched list '{list_name}' to '{matched_list}'")
                
                if matched_list not in apis.CLICKUP_LISTS:
                    logger.warning(f"LLM returned invalid list name: {matched_list}, falling back to o:produkt")
                    matched_list = 'o:produkt'
                
                list_name = matched_list
            except Exception as e:
                logger.error(f"Error matching list name: {e}")
                # Fall back to o:produkt if there's an error
                if list_name not in apis.CLICKUP_LISTS:
                    list_name = 'o:produkt'
            
            # Get the list ID
            list_id = apis.CLICKUP_LISTS.get(list_name)
            if not list_id:
                raise ValueError(f"List '{list_name}' not found")

            # Get available statuses for the list
            available_statuses = await self._get_list_statuses(list_id)
            logger.info(f"Available statuses for list {list_name}: {available_statuses}")

            # Determine the default status based on the list
            default_status = "current sprint" if list_name in ['o:produkt', 'o:growth'] else "backlog"

            # If a status was provided, try to match it to an available status
            task_status = None
            if status:
                status_lower = status.lower()
                # Try exact match first
                for available_status in available_statuses:
                    if status_lower == available_status.lower():
                        task_status = available_status
                        break
                # If no exact match, try partial match
                if not task_status:
                    for available_status in available_statuses:
                        if status_lower in available_status.lower():
                            task_status = available_status
                            break
            
            # Use the matched status, default status, or first available status
            task_status = task_status or default_status
            if task_status not in available_statuses and available_statuses:
                task_status = available_statuses[0]

            # Create the task
            url = f"https://api.clickup.com/api/v2/list/{list_id}/task"
            
            # Prepare the payload
            payload = {
                "name": task_name,
                "status": task_status
            }
            
            # Handle assignees
            if assignees:
                clickup_assignees = []
                # Create a list of all available users for the LLM
                available_users = []
                for key, value in self.user_cache.items():
                    if '@' in key:  # Only include email entries to avoid duplicates
                        user_info = {
                            'id': value,
                            'email': key,
                            'name': next((name for name, uid in self.user_cache.items() 
                                        if uid == value and ' ' in name), '')
                        }
                        if user_info not in available_users:
                            available_users.append(user_info)
                
                # Let the LLM match each assignee
                for assignee in assignees:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a user matcher that maps user references to their ClickUp IDs. "
                                "Return your response in JSON format with a single field 'user_id' containing the matched user's ID.\n\n"
                                "Available users:\n" +
                                "\n".join([f"- {user['name']} (ID: {user['id']}, Email: {user['email']})" 
                                         for user in available_users]) +
                                "\n\nExample response:\n"
                                "{\n"
                                "  \"user_id\": \"82602925\"\n"
                                "}"
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Find the ClickUp user ID for: {assignee}"
                        }
                    ]
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=100,
                            temperature=0.1,
                            response_format={ "type": "json_object" }
                        )
                        
                        parsed = json.loads(response.choices[0].message.content)
                        user_id = parsed.get('user_id')
                        
                        if user_id and str(user_id) in [str(u['id']) for u in available_users]:
                            clickup_id = int(user_id)
                            if clickup_id not in clickup_assignees:
                                clickup_assignees.append(clickup_id)
                                logger.info(f"LLM matched assignee '{assignee}' to user ID {clickup_id}")
                        else:
                            logger.warning(f"LLM could not confidently match user: {assignee}")
                            
                    except Exception as e:
                        logger.error(f"Error matching user with LLM: {e}")
                
                if clickup_assignees:
                    logger.info(f"Setting assignees: {clickup_assignees}")
                    payload["assignees"] = clickup_assignees
                else:
                    logger.warning(f"Could not map any assignees: {assignees}")
                
            if due_date:
                payload["due_date"] = int(datetime.strptime(due_date, "%Y-%m-%d").timestamp() * 1000)
            if priority:
                payload["priority"] = priority
            if description:
                payload["description"] = description
            
            headers = {
                "Authorization": apis.CLICKUP_TOKEN,
                "Content-Type": "application/json"
            }
            
            logger.info(f"Sending task creation request with payload: {payload}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                            # Verify the assignees were set correctly
                            actual_assignees = data.get('assignees', [])
                            if clickup_assignees and not actual_assignees:
                                logger.error(f"Task created but assignees not set. Response: {response_text}")
                                raise Exception("Task created but assignees were not set correctly")
                            return data
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON response: {response_text}")
                            raise Exception(f"Invalid response from ClickUp: {e}")
                    else:
                        logger.error(f"Error creating task. Status: {response.status}, Response: {response_text}")
                        raise Exception(f"Error creating task: {response_text}")
                        
        except ValueError as e:
            available_lists = "\n• ".join(apis.CLICKUP_LISTS.keys())
            raise ValueError(f"Error: {str(e)}.\nAvailable lists:\n• {available_lists}")
        except Exception as e:
            raise Exception(f"Error creating task: {str(e)}")

    def _get_user_name(self, user_id: str) -> str:
        """Get user's display name from their ID."""
        if not user_id or user_id == 'unknown':
            return 'Unknown User'
        
        try:
            # Check if it's our bot
            if user_id == self.client.auth_test()['user_id']:
                return 'Jane'
                
            # Get user info from Slack
            user_info = self.client.users_info(user=user_id)
            
            # Return the display name or real name or username, in that order of preference
            display_name = user_info['user'].get('profile', {}).get('display_name')
            real_name = user_info['user'].get('profile', {}).get('real_name')
            username = user_info['user'].get('name')
            
            return display_name or real_name or username or 'Unknown User'
            
        except SlackApiError as e:
            logger.error(f"Error getting user info: {e}")
            return f"User_{user_id}"

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

    def _format_slack_line(self, line: str) -> str:
        """Format a line of text with proper Slack markdown."""
        line = line.strip()
        if not line:
            return line
            
        # Convert markdown to Slack format
        line = re.sub(r'\*\*(.*?)\*\*', r'*\1*', line)  # Bold
        line = re.sub(r'__(.*?)__', r'_\1_', line)  # Italic
        line = re.sub(r'```(.*?)```', r'`\1`', line)  # Code blocks
        
        # Handle bullet points and blockquotes
        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
            # Remove the bullet point character and any extra spaces
            line = re.sub(r'^[•\-\*]\s*', '• ', line)
            return f">{line}"
        elif line.startswith('>'):
            # Ensure proper spacing after blockquote
            return f">{line[1:].strip()}"
        else:
            return line

    def _format_summary_section(self, text: str) -> str:
        """Format a summary section with proper Slack markdown."""
        if not text:
            return text
            
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            formatted_line = self._format_slack_line(line)
            if formatted_line:
                formatted_lines.append(formatted_line)
        
        return '\n'.join(formatted_lines)

    async def _post_message(self, channel: str, text: str, thread_ts: str = None) -> None:
        """
        Post a message to Slack.
        
        Args:
            channel: The channel ID to post to
            text: The message text
            thread_ts: The thread timestamp if posting in a thread
        """
        try:
            # Send response with unfurl_links=False to prevent link previews
            response = await self.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=text,
                unfurl_links=False
            )
            # Store the timestamp of this message
            channel_key = f"{channel}:{thread_ts if thread_ts else 'main'}"
            self.last_message_ts[channel_key] = response['ts']
        except Exception as e:
            logger.error(f"Error posting message: {e}")
            raise