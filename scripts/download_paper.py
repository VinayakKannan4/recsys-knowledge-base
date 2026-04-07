#!/usr/bin/env python3
"""
Download a paper and create a metadata markdown file.

arXiv mode (existing):
  python3 scripts/download_paper.py <arxiv_id> <short_name>
  Example: python3 scripts/download_paper.py 1906.00091 dlrm

Direct mode (for papers not on arXiv):
  python3 scripts/download_paper.py <short_name> \
    --pdf-url <url_or_local_path> \
    --title "Paper Title" \
    --authors "Author One, Author Two" \
    --source-url https://dl.acm.org/doi/...
  Example:
  python3 scripts/download_paper.py youtube-dnn-recommendations \
    --pdf-url https://example.com/paper.pdf \
    --title "Deep Neural Networks for YouTube Recommendations" \
    --authors "Paul Covington, Jay Adams, Emre Sargin" \
    --source-url https://dl.acm.org/doi/10.1145/2959100.2959190
"""
import sys
import os
import shutil
import requests
from datetime import date


def download_arxiv(arxiv_id, short_name):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"

    paper_dir = f"raw/papers/{short_name}"
    os.makedirs(paper_dir, exist_ok=True)

    pdf_path = f"{paper_dir}/{short_name}.pdf"
    print(f"Downloading: {pdf_url}")
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF saved to: {pdf_path}")

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


def download_direct(short_name, pdf_url, title, authors, source_url):
    paper_dir = f"raw/papers/{short_name}"
    os.makedirs(paper_dir, exist_ok=True)

    pdf_path = f"{paper_dir}/{short_name}.pdf"

    if pdf_url.startswith("http://") or pdf_url.startswith("https://"):
        print(f"Downloading PDF from: {pdf_url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,*/*',
        }
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            raise ValueError(f"Got HTML instead of PDF — URL likely requires a login: {pdf_url}")
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Copying local PDF from: {pdf_url}")
        shutil.copy(pdf_url, pdf_path)

    print(f"PDF saved to: {pdf_path}")

    meta_path = f"{paper_dir}/{short_name}-meta.md"
    meta_content = f"""---
source_url: {source_url}
pdf_path: {pdf_path}
arxiv_id: N/A
title: "{title}"
authors: "{authors}"
downloaded_date: {date.today().isoformat()}
type: paper
---

# {title}

**Authors**: {authors}

**Source**: [{source_url}]({source_url})

## Abstract

*(To be filled by the LLM agent during compilation)*

## Key Takeaways

*(To be filled by the LLM agent during compilation)*

## Notes

*(To be filled by the LLM agent during compilation)*
"""
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(meta_content)
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    # Detect mode: if --pdf-url is present, use direct mode
    if "--pdf-url" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("short_name")
        parser.add_argument("--pdf-url", required=True)
        parser.add_argument("--title", required=True)
        parser.add_argument("--authors", required=True)
        parser.add_argument("--source-url", required=True)
        args = parser.parse_args()
        download_direct(args.short_name, args.pdf_url, args.title, args.authors, args.source_url)
    else:
        if len(sys.argv) != 3:
            print("Usage:")
            print("  arXiv:  python3 scripts/download_paper.py <arxiv_id> <short_name>")
            print("  Direct: python3 scripts/download_paper.py <short_name> --pdf-url <url_or_path> --title '...' --authors '...' --source-url '...'")
            sys.exit(1)
        download_arxiv(sys.argv[1], sys.argv[2])
