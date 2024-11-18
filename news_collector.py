# news_collector.py

import requests
from newspaper import Article
from newspaper import Config
import openai
from openai import OpenAI
import apis
from datetime import datetime, timedelta
import re
from urllib.parse import urlparse
import random

# Initialize the OpenAI client
client = OpenAI(
    api_key=apis.OPEN_AI_TR,
)
NEWS_API_KEY = apis.NEWS
DEFAULT_KEYWORDS = ['"movie recommendations"', '"netflix recommendations"', '"streaming services"', '"netflix"', '"hbo max"']
DEFAULT_ARTICLES_PER_KEYWORD = 2

def clean_authors(authors_list):
    cleaned_authors = []
    for author in authors_list:
        author = author.strip()
        if not author:
            continue
        # Remove authors that are too long (more than 4 words)
        words = author.split()
        if len(words) > 4:
            continue
        # Remove authors that contain digits
        if re.search(r'\d', author):
            continue
        # Remove authors that contain too many special characters
        if re.search(r'[^a-zA-Z\s\-.]', author):
            continue
        # Remove authors that are common invalid entries
        invalid_authors = {'by', 'unknown', 'author'}
        if author.lower() in invalid_authors:
            continue
        cleaned_authors.append(author)
    return cleaned_authors

def get_articles(api_key, query, from_date, to_date, page_size=20, in_title=False):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': page_size,
        'apiKey': api_key,
    }
    if in_title:
        params['qInTitle'] = query
    else:
        params['q'] = query
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get('status') != 'ok':
            print(f"Error: {data.get('code')}, Message: {data.get('message')}")
            return []
        return data.get('articles', [])
    except Exception as e:
        print(f"Failed to get articles: {e}")
        return []

def scrape_article_content(url):
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' \
                                '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    config.request_timeout = 10
    article = Article(url, config=config)
    try:
        article.download()
        article.parse()
        cleaned_authors = clean_authors(article.authors)
        return {
            'title': article.title,
            'text': article.text,
            'authors': cleaned_authors,
            'publish_date': article.publish_date,
        }
    except Exception as e:
        print(f"Failed to download or parse article at {url}: {e}")
        return None

def extract_main_theses(text):
    messages = [
        {
            "role": "user",
            "content": (
                "Summarize the main theses of the following article in one concise sentence:\n\n"
                f"{text}"
            ),
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to extract main theses: {e}")
        return ""

def get_domain_from_url(url):
    """Extract the domain name from a URL."""
    try:
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        parts = domain.split('.')
        if len(parts) >= 2:
            main_part = parts[0]
            # Handle special cases
            if main_part == 'theguardian':
                return 'the Guardian'
            elif main_part == 'nytimes':
                return 'The New York Times'
            elif main_part == 'wsj':
                return 'The Wall Street Journal'
            else:
                return main_part.capitalize()
    except:
        return domain
    


    
def check_tasteray_relevance(text, title):
    """Check if the article is relevant enough for Tasteray's hyperpersonalized recommendation platform."""
    messages = [
        {
            "role": "user",
            "content": (
                "You are evaluating articles for Tasteray, a startup building a hyperpersonalized movie recommendation app. "
                "Based on the following article title and content, respond with a number from 0-10 indicating "
                "how relevant this article is for Tasteray's development or strategy.\n\n"
                "Use these criteria:\n"
                "10: Directly about personalization algorithms, recommendation systems, or user preference analysis\n"
                "7-9: About content discovery, user experience personalization, or viewer behavior analysis\n"
                "4-6: General streaming technology news that might affect personalized recommendations\n"
                "1-3: Basic streaming industry news\n"
                "0: Not relevant for a personalized recommendation platform\n\n"
                f"Title: {title}\n\n"
                f"Content: {text}\n\n"
                "Response format: Just the number (0-10)"
            ),
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=10,
            temperature=0.3,
        )
        relevance_score = int(response.choices[0].message.content.strip())
        return relevance_score >= 6  # Only consider articles with high relevance
    except Exception as e:
        print(f"Failed to check Tasteray relevance: {e}")
        return False
    

def analyze_tasteray_implications(text, title):
    """Analyze how the article could be relevant for Tasteray's hyperpersonalized recommendation platform."""
    messages = [
        {
            "role": "user",
            "content": (
                "You are an advisor to Tasteray, a startup building a hyperpersonalized movie recommendation app. "
                "Based on the following article title and content, explain in one concise sentence "
                "how this information could be valuable for Tasteray's development of ultra-personalized "
                "recommendation features or strategy. Focus on practical implications for personalization.\n\n"
                f"Title: {title}\n\n"
                f"Content: {text}"
            ),
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to analyze Tasteray implications: {e}")
        return ""
    


def format_articles_for_slack(articles_data):
    """Format articles data for Slack message with the new structure."""
    if not articles_data:
        return "No relevant articles found for the specified period."
    
    # Start with the friendly introduction
    message_parts = [generate_friendly_intro()]
    
    # Process each article
    for data in articles_data:
        # Format title (bold)
        title = f"*{data['title']}*"
        
        # Format description (main thesis)
        description = data['main_theses']
        
        # Format Tasteray implications (italics) if it exists
        tasteray_section = ""
        if data.get('tasteray_implications'):
            tasteray_section = f"\n_Tasteray implications: {data['tasteray_implications']}_"
        
        # Handle author formatting
        authors = data['author']
        if authors:
            authors_str = ', '.join(authors)
        else:
            authors_str = 'Unknown author'
            
        # Handle date formatting
        try:
            date_obj = datetime.fromisoformat(data['publish_date'].replace('Z', '+00:00'))
            date_str = date_obj.strftime('%B %d, %Y')
        except:
            date_str = data['publish_date']
        
        # Create source link with unfurl_links=False
        source_domain = get_domain_from_url(data['source'])
        source_link = f"<{data['source']}|read at {source_domain}>"
        
        # Combine metadata
        metadata = f"{date_str} • {authors_str} • {source_link}"
        
        # Add keyword information if custom keywords were used
        if data.get('keyword') not in [k.strip('"') for k in DEFAULT_KEYWORDS]:
            metadata += f" • Keyword: {data['keyword']}"
        
        # Combine all parts with proper spacing
        article_message = f"{title}\n{description}{tasteray_section}\n{metadata}"
        
        message_parts.append(article_message)
    
    # Join all parts with proper spacing
    return "\n\n---\n\n".join(message_parts)



def generate_friendly_intro():
    """Generate a friendly, varied introduction for the daily news digest."""
    intros = [
        "👋 Hi there! Here are today's interesting articles!",
        "🎬 Happy reading! Here's what's new in the streaming world today:",
        "📺 Good news! I found some interesting articles about streamings and recommendations:",
        "🎯 Hello! Check out these relevant articles I discovered today:",
        "🎥 Exciting updates! Here's what's happening in the streaming space:",
        "🌟 Greetings! I've gathered some interesting articles for Tasteray today:",
        "📱 Hi! Here's your daily dose of streaming industry insights:",
        "🎪 Welcome! I've curated these relevant articles for today:",
        "🎭 Hello there! Today's selection of streaming news is ready:",
        "🍿 Fresh news! Here's what caught my attention today:"
    ]
    return random.choice(intros)



def main_function(from_date='', to_date='', keywords=None, articles_per_keyword=None):
    """
    Main function that supports custom keywords and article counts.
    
    Args:
        from_date (str): Start date in YYYY-MM-DD format
        to_date (str): End date in YYYY-MM-DD format
        keywords (list): Optional list of keywords to search for
        articles_per_keyword (int): Optional number of articles per keyword
    """
    # Use default keywords if none provided
    keywords = keywords or DEFAULT_KEYWORDS
    # Use default article count if none provided
    x = articles_per_keyword or DEFAULT_ARTICLES_PER_KEYWORD
    
    # Handle dates
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    if not from_date and not to_date:
        from_date = yesterday
        to_date = yesterday

    articles_data = []

    for query in keywords:
        print(f"Processing keyword: {query}")
        articles = get_articles(NEWS_API_KEY, query, from_date, to_date, page_size=x, in_title=True)
        if not articles:
            print(f"No articles found for keyword: {query}")
            continue

        for item in articles:
            article_url = item['url']
            print(f"Processing article: {item['title']}")
            try:
                article_data = scrape_article_content(article_url)

                if not article_data or not article_data.get('text'):
                    print("No content found in the article.")
                    continue

                main_theses = extract_main_theses(article_data['text'])
                
                if check_tasteray_relevance(article_data['text'], article_data['title']):
                    tasteray_implications = analyze_tasteray_implications(article_data['text'], article_data['title'])
                else:
                    tasteray_implications = None

                data = {
                    'keyword': query.strip('"'),
                    'title': article_data['title'],
                    'author': article_data['authors'],
                    'publish_date': str(article_data['publish_date']),
                    'main_theses': main_theses,
                    'tasteray_implications': tasteray_implications,
                    'source': article_url,
                }

                articles_data.append(data)
            except Exception as e:
                print(f"An error occurred while processing article: {e}")
                continue
                
    return format_articles_for_slack(articles_data)