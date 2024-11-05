import json
import time
import requests
from newspaper import Article
from newspaper import Config
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import openai
from openai import OpenAI
import apis
from datetime import datetime, timedelta
import re
from requests_html import HTMLSession

# Initialize the OpenAI client
client = OpenAI(
    api_key=apis.OPEN_AI_TR,
)
NEWS_API_KEY = apis.NEWS


def fetch_page_content_requests_html(url):
    session = HTMLSession()
    try:
        response = session.get(url)
        response.html.render(timeout=20, chromium_args=["--no-sandbox"], executable_path="/usr/bin/chromium-browser")
        html_content = response.html.html
        return html_content
    except Exception as e:
        print(f"Failed to fetch page content at {url} using requests_html: {e}")
        return None


def clean_authors(authors_list):
    cleaned_authors = []
    for author in authors_list:
        author = author.strip()
        if not author:
            continue
        # Remove authors that are too long (e.g., more than 4 words)
        words = author.split()
        if len(words) > 4:
            continue
        # Remove authors that contain digits
        if re.search(r'\d', author):
            continue
        # Remove authors that contain too many special characters
        if re.search(r'[^a-zA-Z\s\-.]', author):
            continue
        # Remove authors that contain words unlikely to be in a name
        unwanted_words = {'class', 'display', 'inline', 'where', 'img', 'height', 'auto', 'max-width',
                          'vertical-align', 'middle', 'alignleft', 'alignright', 'wp-block', 'coauthors', 'plus', 'is', 'layout', 'flow'}
        if any(word.lower() in author.lower() for word in unwanted_words):
            continue
        # Remove authors that are common invalid entries
        invalid_authors = {'by', 'unknown', 'author'}
        if author.lower() in invalid_authors:
            continue
        cleaned_authors.append(author)
    return cleaned_authors



def extract_text_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text(separator='\n')
    return text_content

def fetch_page_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' \
                      '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
        return html_content
    except Exception as e:
        print(f"Failed to fetch page content at {url}: {e}")
        return None

def fetch_page_content_selenium(url):
    options = Options()
    options.headless = True
    options.add_argument('--no-sandbox')  # Necessary for running as root or in some environments
    options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(5)  # Wait for content to load if necessary
        html_content = driver.page_source
        return html_content
    except Exception as e:
        print(f"Failed to fetch page content at {url}: {e}")
        return None
    finally:
        driver.quit()

def extract_comments_from_text(text_content):
    # Truncate the text if necessary
    max_tokens = 3500  # Adjust as needed

    if len(text_content) > max_tokens * 4:
        text_content = text_content[:max_tokens * 4]

    messages = [
        {
            "role": "user",
            "content": (
                "From the following text, extract all user comments made on an article. "
                "The text may include the article content, navigation text, and other elements. "
                "If there are no comments in this text, output NO COMMENTS as your only output. "
                "Provide the comments as a JSON array of strings.\n\n"
                f"{text_content}"
            )
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0,
        )
        comments_text = response.choices[0].message.content.strip()

        if comments_text.lower() == "no comments":
            return []

        # Attempt to parse the response as JSON
        comments = json.loads(comments_text)
        return comments
    except Exception as e:
        print(f"Failed to extract comments using OpenAI API: {e}")
        return []

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
    config.request_timeout = 10  # Optional: Set a timeout for the request
    article = Article(url, config=config)
    try:
        article.download()
        article.parse()
        # Clean the authors list
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


def scrape_comments(url):
    # Fetch the page content (use Selenium if necessary)
    html_content = fetch_page_content_requests_html(url)

    if not html_content:
        print(f"No HTML content fetched for {url}")
        return []

    text_content = extract_text_content(html_content)

    comments = extract_comments_from_text(text_content)
    return comments

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
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to extract main theses: {e}")
        return ""

def categorize_comments(comments):
    categorized = {'interesting': [], 'funny': []}
    for comment in comments:
        messages = [
            {
                "role": "user",
                "content": (
                    "Categorize the following comment as 'interesting' or 'funny'. "
                    "Respond with only one word, which is the appropriate classification.\n\n"
                    f"{comment}"
                ),
            }
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=5,
                temperature=0.7,
            )
            category = response.choices[0].message.content.strip().lower()
            # Clean up the category string
            category = category.strip('.').strip('"').strip("'")
            if category in categorized:
                categorized[category].append(comment)
            else:
                # If the category is neither 'interesting' nor 'funny', you can choose to categorize it as 'other' or skip
                continue
        except Exception as e:
            print(f"Failed to categorize comment: {e}")
            continue
    return categorized


def format_articles_for_slack(articles_data):
    messages = []
    for data in articles_data:
        authors = data['author']
        if authors:
            authors_str = ', '.join(authors)
        else:
            authors_str = 'Unknown'
        message = f"*Title:* {data['title']}\n" \
                  f"*Author:* {authors_str}\n" \
                  f"*Published Date:* {data['publish_date']}\n" \
                  f"*Main Thesis:* {data['main_theses']}\n" \
                  f"*Source:* {data['source']}\n"
        # Include comments
        if data['comments']:
            comments_section = ""
            for category, comments_list in data['comments'].items():
                if comments_list:
                    # Limit the number of comments per category if necessary
                    max_comments = 3  # Adjust as needed
                    limited_comments = comments_list[:max_comments]
                    comments_text = '\n'.join([f"- {comment}" for comment in limited_comments])
                    comments_section += f"*{category.capitalize()} Comments:*\n{comments_text}\n"
            if comments_section:
                message += comments_section
        messages.append(message)
    return '\n---\n'.join(messages)


def main_function():
    keywords = ['"movie recommendations"', '"netflix recommendations"', '"streaming services"', '"netflix"', '"hbo max"']
    x = 2  # Number of articles per keyword
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
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
                comments = scrape_comments(article_url)
                if comments:
                    categorized_comments = categorize_comments(comments)
                else:
                    categorized_comments = {'interesting': [], 'funny': [], 'popular': []}

                data = {
                    'keyword': query.strip('"'),  # Remove quotes when storing
                    'title': article_data['title'],
                    'author': article_data['authors'],
                    'publish_date': str(article_data['publish_date']),
                    'main_theses': main_theses,
                    'comments': categorized_comments,
                    'source': article_url,
                }

                articles_data.append(data)
            except Exception as e:
                print(f"An error occurred while processing article: {e}")
                continue
    return format_articles_for_slack(articles_data)

if __name__ == "__main__":
    articles = main_function()
    print(articles)
