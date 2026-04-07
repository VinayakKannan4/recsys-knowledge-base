#!/usr/bin/env python3
"""
Download an arXiv paper and create a metadata markdown file.
Usage: python3 scripts/download_paper.py <arxiv_id> <short_name>
Example: python3 scripts/download_paper.py 1906.00091 dlrm
"""
import sys
import os
import requests
from datetime import date

def download_paper(arxiv_id, short_name):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"

    # Create directory
    paper_dir = f"raw/papers/{short_name}"
    os.makedirs(paper_dir, exist_ok=True)

    # Download PDF
    pdf_path = f"{paper_dir}/{short_name}.pdf"
    print(f"Downloading: {pdf_url}")
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF saved to: {pdf_path}")

    # Fetch abstract page for metadata
    print(f"Fetching metadata from: {abs_url}")
    abs_response = requests.get(abs_url, timeout=30)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(abs_response.text, 'html.parser')

    title_tag = soup.find('h1', class_='title')
    title = title_tag.text.replace('Title:', '').strip() if title_tag else short_name

    abstract_tag = soup.find('blockquote', class_='abstract')
    abstract = abstract_tag.text.replace('Abstract:', '').strip() if abstract_tag else "No abstract found."

    authors_tag = soup.find('div', class_='authors')
    authors = authors_tag.text.replace('Authors:', '').strip() if authors_tag else "Unknown"

    # Create metadata markdown
    meta_path = f"{paper_dir}/{short_name}-meta.md"
    meta_content = f"""---
source_url: {abs_url}
pdf_path: {pdf_path}
arxiv_id: {arxiv_id}
title: "{title}"
authors: "{authors}"
downloaded_date: {date.today().isoformat()}
type: paper
---

# {title}

**Authors**: {authors}

**arXiv**: [{arxiv_id}]({abs_url})

## Abstract

{abstract}

## Key Takeaways

*(To be filled by the LLM agent during compilation)*

## Notes

*(To be filled by the LLM agent during compilation)*
"""

    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(meta_content)
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/download_paper.py <arxiv_id> <short_name>")
        sys.exit(1)
    download_paper(sys.argv[1], sys.argv[2])
