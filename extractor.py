"""

extractor.py

"""
import os
import re
import csv
import html

from config import Config


class Extractor:

    def __init__(self, config):
        self.config = config



    def clean(self, text):

        if not text:
            return ''
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()



    def extract(self, html_text, book_id):
        """Extract all metadata from a Gutenberg book page."""
        data = {'gutenberg_id': str(book_id)}

        match = re.search(r'<th>Title</th>\s*<td[^>]*>\s*(.*?)\s*</td>', html_text, re.DOTALL)
        data['title'] = self.clean(match.group(1)) if match else ''

        match = re.search(r'<a href="/ebooks/author/\d+"[^>]*>(.*?)</a>', html_text, re.DOTALL)
        data['author'] = self.clean(match.group(1)) if match else ''

        match = re.search(r'<th>Language</th>\s*<td>(.*?)</td>', html_text, re.DOTALL)
        data['language'] = self.clean(match.group(1)) if match else ''

        match = re.search(r'<th>Release Date</th>\s*<td[^>]*>(.*?)</td>', html_text, re.DOTALL)
        data['release_date'] = self.clean(match.group(1)) if match else ''

        match = re.search(r'<th>Last Update</th>\s*<td[^>]*>(.*?)</td>', html_text, re.DOTALL)
        data['last_update'] = self.clean(match.group(1)) if match else ''

        subjects = []
        for m in re.finditer(r'<th>Subject</th>\s*<td[^>]*>.*?<a[^>]*>(.*?)</a>', html_text, re.DOTALL):
            subjects.append(self.clean(m.group(1)))
        data['subjects'] = '|'.join(subjects)

        match = re.search(r'<th>LoC Class</th>\s*<td>\s*<a[^>]*>(.*?)</a>', html_text, re.DOTALL)
        data['loc_class'] = self.clean(match.group(1)) if match else ''

        match = re.search(r'<th>Downloads</th>\s*<td[^>]*>(.*?)</td>', html_text, re.DOTALL)
        data['downloads'] = self.clean(match.group(1)) if match else ''

        match = re.search(r'<th>Reading Level</th>\s*<td>\s*(.*?)\s*</td>', html_text, re.DOTALL)
        data['reading_ease'] = self.clean(match.group(1)) if match else ''

        match = re.search(r'<th>Credits</th>\s*<td>\s*(.*?)\s*</td>', html_text, re.DOTALL)
        data['credits'] = self.clean(match.group(1)) if match else ''

        match = re.search(r'Wikipedia page about this book:\s*<a href="([^"]+)"', html_text)
        data['wikipedia_url'] = match.group(1) if match else ''

        match = re.search(r'href="(/ebooks/\d+\.txt\.utf-8)"', html_text)
        data['text_url'] = 'https://www.gutenberg.org' + match.group(1) if match else ''

        return data


    def extract_all(self):

        print("=" * 70)
        print("EXTRACT START")
        print("=" * 70)

        # Find all HTML files
        html_files = []
        for filename in os.listdir(self.config.OUTPUT_DIR):
            if filename.endswith('.html'):
                book_id = filename.replace('.html', '')
                if book_id.isdigit():
                    html_files.append(book_id)

        html_files.sort(key=int)
        print(f"HTML files: {len(html_files)}")
        print()

        # Load existing TSV records
        existing = self._load_existing_tsv()
        print(f"Existing TSV records: {len(existing)}")
        print()

        # Open TSV for appending (write header if file is new/empty)
        write_header = not os.path.exists(self.config.TSV_FILE) or os.path.getsize(self.config.TSV_FILE) == 0

        with open(self.config.TSV_FILE, 'a', encoding='utf-8', newline='') as tsv_f:
            writer = csv.DictWriter(
                tsv_f, fieldnames=self.config.TSV_FIELDS,
                delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\'
            )
            if write_header:
                writer.writeheader()
                tsv_f.flush()

            extracted = 0
            skipped = 0
            failed = 0
            failed_ids = []

            for i, book_id in enumerate(html_files, 1):
                if book_id in existing:
                    skipped += 1
                    continue

                try:
                    html_path = os.path.join(self.config.OUTPUT_DIR, f"{book_id}.html")
                    with open(html_path, 'r', encoding='utf-8') as html_f:
                        html_text = html_f.read()

                    data = self.extract(html_text, book_id)
                    if not data['title']:
                        failed += 1
                        failed_ids.append(book_id)
                        print(f"[{i}/{len(html_files)}] {book_id}: SKIP (no title found)")
                        continue

                    data['html_size'] = os.path.getsize(html_path)

                    txt_path = os.path.join(self.config.OUTPUT_DIR, f"{book_id}.txt")
                    data['text_size'] = os.path.getsize(txt_path) if os.path.exists(txt_path) else 0

                    writer.writerow(data)
                    tsv_f.flush()
                    extracted += 1

                    print(f"[{i}/{len(html_files)}] {book_id}: {data['title'][:60]}")
                    print(f"    HTML: {data['html_size']:,} B | Text: {data['text_size']:,} B")

                except Exception as e:
                    failed += 1
                    failed_ids.append(book_id)
                    print(f"[{i}/{len(html_files)}] {book_id}: ERROR — {e}")
                    continue

        print()
        print("=" * 70)
        print(f"EXTRACT FINISHED")
        print(f"  Extracted: {extracted}")
        print(f"  Skipped:   {skipped} (already in TSV)")
        print(f"  Failed:    {failed}")
        print("=" * 70)

        if failed_ids:
            print(f"  Failed IDs (first 20 shown): {failed_ids[:20]}")

    def _load_existing_tsv(self):
        records = set()
        if os.path.exists(self.config.TSV_FILE):
            with open(self.config.TSV_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    records.add(row['gutenberg_id'])
        return records


    def delete_tsv(self):

        if os.path.exists(self.config.TSV_FILE):
            os.remove(self.config.TSV_FILE)
            print(f"Deleted {self.config.TSV_FILE}.")
            return True
        else:
            print(f"{self.config.TSV_FILE} does not exist, nothing to delete.")
            return False