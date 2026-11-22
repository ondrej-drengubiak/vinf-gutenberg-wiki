"""

crawler.py

"""

import urllib.request
import urllib.error
import urllib.parse
import re
import time
import gzip
import sys
import json
import os

from config import Config


class Crawler:

    def __init__(self, config):
        self.config = config
        self.queue = self._load_queue()
        self.visited = self._load_visited()



    def _load_queue(self):
        if os.path.exists(self.config.QUEUE_FILE):
            with open(self.config.QUEUE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return list(self.config.SEED_URLS)

    def _load_visited(self):
        if os.path.exists(self.config.VISITED_FILE):
            with open(self.config.VISITED_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()

    def _save_state(self):
        with open(self.config.QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.queue, f, indent=2)
        with open(self.config.VISITED_FILE, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(self.visited)), f, indent=2)



    def fetch(self, url, timeout=20):
        req = urllib.request.Request(url, headers=self.config.HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read()
                try:
                    if raw[:2] == b'\x1f\x8b':
                        raw = gzip.decompress(raw)
                except Exception:
                    pass
                return raw.decode('utf-8', errors='ignore')
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                print(f"\n  ⚠ HTTP {e.code} — blocked. Stopping.")
                sys.exit(1)
            print(f"  HTTP {e.code}")
            return None
        except Exception as e:
            print(f"  ERROR: {e}")
            return None


    def extract_links(self, html, base_url):
        links = set()
        pattern = re.compile(r'<a[^>]+href="([^"]+)"', re.IGNORECASE)

        for match in pattern.finditer(html):
            href = match.group(1)
            if href.startswith(('#', 'javascript:', 'mailto:')):
                continue

            if href.startswith('/'):
                full_url = 'https://www.gutenberg.org' + href
            elif href.startswith('http'):
                full_url = href
            else:
                full_url = urllib.parse.urljoin(base_url, href)

            if self.config.ALLOWED_DOMAIN not in full_url:
                continue

            links.add(full_url)

        return list(links)

    def extract_book_links(self, html, base_url):
        links = self.extract_links(html, base_url)
        book_pattern = re.compile(self.config.BOOK_URL_PATTERN)
        return [link for link in links if book_pattern.match(link)]



    def crawl(self):

        print("=" * 70)
        print("CRAWLER START")
        print("=" * 70)
        print(f"Queue size:    {len(self.queue)}")
        print(f"Visited:       {len(self.visited)}")
        print()

        books_saved = 0

        while self.queue:
            url = self.queue.pop(0)

            if url in self.visited:
                continue
            self.visited.add(url)

            print(f"[{len(self.visited)}] {url}")

            html = self.fetch(url, timeout=self.config.TIMEOUT_HTML)
            if not html:
                self._save_state()
                continue

            # Is this a book page?
            if re.match(self.config.BOOK_URL_PATTERN, url):
                # Save HTML to disk
                book_id = self._extract_book_id(url)
                if book_id:
                    path = os.path.join(self.config.OUTPUT_DIR, f"{book_id}.html")
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(html)
                    books_saved += 1
                    print(f"  ✓ Saved HTML for book {book_id}")
            else:
                # Not a book page — extract links
                book_links = self.extract_book_links(html, url)
                all_links = self.extract_links(html, url)

                # Book links first (priority)
                for link in book_links:
                    if link not in self.visited:
                        self.queue.insert(0, link)

                for link in all_links:
                    if link not in self.visited and link not in self.queue:
                        self.queue.append(link)

                print(f"  → {len(book_links)} book links, {len(all_links)} total links")

            # Save state periodically
            self._save_state()
            time.sleep(self.config.DELAY)

        print()
        print("=" * 70)
        print(f"CRAWLER FINISHED: {books_saved} new book pages saved")
        print("=" * 70)

    def _extract_book_id(self, url):
        match = re.search(r'/ebooks/(\d+)', url)
        return match.group(1) if match else None