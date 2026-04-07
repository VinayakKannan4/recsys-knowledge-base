#!/usr/bin/env python3
"""
Clip a web URL to a clean markdown file.
Usage: python3 scripts/clip_url.py <url> <output_path>
Example: python3 scripts/clip_url.py "https://example.com/blog" raw/blogs/meta/sequence-learning.md
"""
import sys
import os
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from datetime import date

def clip_url(url, output_path):
    print(f"Fetching: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 403:
        # Fall back to Googlebot UA, which many sites (including Medium) allow
        headers['User-Agent'] = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
        response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove nav, footer, ads, scripts, styles
    for tag in soup.find_all(['nav', 'footer', 'script', 'style', 'aside',
                               'iframe', 'noscript']):
        tag.decompose()

    # Try to find main content
    main = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body

    if main is None:
        main = soup

    markdown_content = md(str(main), heading_style="ATX", strip=['img'])

    # Add metadata header
    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
    header = f"""---
source_url: {url}
title: "{title}"
clipped_date: {date.today().isoformat()}
type: blog
---

# {title}

"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + markdown_content)

    print(f"Saved to: {output_path}")
    print(f"Size: {len(markdown_content)} characters")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/clip_url.py <url> <output_path>")
        sys.exit(1)
    clip_url(sys.argv[1], sys.argv[2])
