"""

fetcher.py

"""

import urllib.request
import urllib.error
import re
import time
import os

from config import Config


class Fetcher:

    def __init__(self, config):
        self.config = config

    def fetch_binary(self, url, timeout=30, max_retries=3):
        """Download binary content (text files) with retry."""
        for attempt in range(max_retries):
            req = urllib.request.Request(url, headers=self.config.HEADERS)
            chunks = []
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    return b''.join(chunks)
            except Exception as e:
                if chunks:
                    return b''.join(chunks)
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt+1}/{max_retries}: {e}")
                    time.sleep(2)
                else:
                    print(f"  ERROR (after {max_retries} attempts): {e}")
                    return None
        return None

    def fetch_all_texts(self):

        print("=" * 70)
        print("FETCH TEXT START")
        print("=" * 70)

        # Find all HTML files
        html_files = []
        for filename in os.listdir(self.config.OUTPUT_DIR):
            if filename.endswith('.html'):
                book_id = filename.replace('.html', '')
                if book_id.isdigit():
                    html_files.append(book_id)

        html_files.sort(key=int)
        print(f"HTML files:    {len(html_files)}")

        # How many already have text?
        already_have_txt = 0
        for book_id in html_files:
            if os.path.exists(os.path.join(self.config.OUTPUT_DIR, f"{book_id}.txt")):
                already_have_txt += 1

        print(f"Already have TXT: {already_have_txt}")
        print(f"To fetch:         {len(html_files) - already_have_txt}")
        print()

        fetched = 0
        skipped = 0
        failed = 0

        for i, book_id in enumerate(html_files, 1):
            txt_path = os.path.join(self.config.OUTPUT_DIR, f"{book_id}.txt")
            if os.path.exists(txt_path):
                skipped += 1
                continue

            # Read HTML to find text URL
            html_path = os.path.join(self.config.OUTPUT_DIR, f"{book_id}.html")
            with open(html_path, 'r', encoding='utf-8') as f:
                html = f.read()

            match = re.search(r'href="(/ebooks/\d+\.txt\.utf-8)"', html)
            if not match:
                print(f"[{i}/{len(html_files)}] {book_id}: no text URL")
                failed += 1
                continue

            text_url = 'https://www.gutenberg.org' + match.group(1)
            print(f"[{i}/{len(html_files)}] {book_id}: {text_url}")

            text_bytes = self.fetch_binary(text_url)
            if text_bytes:
                with open(txt_path, 'wb') as f:
                    f.write(text_bytes)
                fetched += 1
                print(f"  ✓ {len(text_bytes):,} B")
            else:
                failed += 1
                print(f"  ✗ FAILED")

            time.sleep(self.config.DELAY)

        print()
        print("=" * 70)
        print(f"FETCH TEXT FINISHED")
        print(f"  Fetched: {fetched}")
        print(f"  Skipped: {skipped} (already had text)")
        print(f"  Failed:  {failed}")
        print("=" * 70)