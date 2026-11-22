"""

test_extractor.py

"""


import os
import re
import unittest

from config import Config
from extractor import Extractor

MIN_SAMPLE_SIZE = 20

# Known-good expected values for a few specific Gutenberg IDs
EXPECTED = {
    '1': {
        'title': 'The Declaration of Independence of the United States of America',
        'author': 'Jefferson, Thomas, 1743-1826',
    },
    '11': {
        'title': "Alice's Adventures in Wonderland",
        'author': 'Carroll, Lewis, 1832-1898',
    },
    '15': {
        'title': 'Moby-Dick; or, The Whale',
        'author': 'Melville, Herman, 1819-1891',
    },
    '46': {
        'author': 'Dickens, Charles, 1812-1870',
    },
    '28062': {
        'title': 'The Man Who Saw the Future',
        'author': 'Hamilton, Edmond, 1904-1977',
    },
}


def _sample_book_ids(config, limit=MIN_SAMPLE_SIZE):
    ids = sorted(EXPECTED.keys(), key=int)  # known ones first, if present
    for filename in sorted(os.listdir(config.OUTPUT_DIR)):
        if len(ids) >= limit:
            break
        if filename.endswith('.html'):
            book_id = filename[:-len('.html')]
            if book_id.isdigit() and book_id not in ids:
                ids.append(book_id)
    return ids[:max(limit, len(EXPECTED))]


class TestExtractorStructural(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.extractor = Extractor(cls.config)
        if not os.path.exists(cls.config.OUTPUT_DIR):
            raise unittest.SkipTest(f"{cls.config.OUTPUT_DIR} not found — run Crawl first.")
        cls.book_ids = _sample_book_ids(cls.config)
        if len(cls.book_ids) < MIN_SAMPLE_SIZE:
            raise unittest.SkipTest(
                f"Only {len(cls.book_ids)} HTML files available, "
                f"need at least {MIN_SAMPLE_SIZE}."
            )

    def _load(self, book_id):
        path = os.path.join(self.config.OUTPUT_DIR, f"{book_id}.html")
        with open(path, 'r', encoding='utf-8') as f:
            html_text = f.read()
        return self.extractor.extract(html_text, book_id)

    def test_sample_size_at_least_20(self):
        self.assertGreaterEqual(len(self.book_ids), MIN_SAMPLE_SIZE)

    def test_gutenberg_id_matches_filename(self):
        for book_id in self.book_ids:
            with self.subTest(book_id=book_id):
                data = self._load(book_id)
                self.assertEqual(data['gutenberg_id'], book_id)

    def test_no_leftover_html_tags_in_text_fields(self):
        text_fields = ['title', 'author', 'subjects', 'loc_class',
                       'downloads', 'reading_ease', 'credits']
        for book_id in self.book_ids:
            with self.subTest(book_id=book_id):
                data = self._load(book_id)
                for field in text_fields:
                    value = data.get(field, '')
                    self.assertNotIn('<', value, f"{field} of {book_id} still has a raw '<'")
                    self.assertNotIn('>', value, f"{field} of {book_id} still has a raw '>'")

    def test_no_raw_html_entities_left(self):
        text_fields = ['title', 'author', 'subjects', 'reading_ease']
        entity_re = re.compile(r'&[a-zA-Z#0-9]+;')
        for book_id in self.book_ids:
            with self.subTest(book_id=book_id):
                data = self._load(book_id)
                for field in text_fields:
                    value = data.get(field, '')
                    self.assertIsNone(
                        entity_re.search(value),
                        f"{field} of {book_id} still has a raw HTML entity: {value!r}"
                    )

    def test_no_glued_words_from_removed_tags(self):

        glued_re = re.compile(r'[a-z][A-Z]')
        for book_id in self.book_ids:
            with self.subTest(book_id=book_id):
                data = self._load(book_id)
                title = data.get('title', '')
                self.assertIsNone(
                    glued_re.search(title),
                    f"title of {book_id} looks glued (missing space): {title!r}"
                )

    def test_text_url_well_formed_when_present(self):
        pattern = re.compile(r'^https://www\.gutenberg\.org/ebooks/\d+\.txt\.utf-8$')
        for book_id in self.book_ids:
            with self.subTest(book_id=book_id):
                data = self._load(book_id)
                if data.get('text_url'):
                    self.assertRegex(data['text_url'], pattern)

    def test_wikipedia_url_well_formed_when_present(self):
        for book_id in self.book_ids:
            with self.subTest(book_id=book_id):
                data = self._load(book_id)
                url = data.get('wikipedia_url', '')
                if url:
                    self.assertTrue(url.startswith('https://en.wikipedia.org/wiki/'))

    def test_subjects_use_pipe_separator_not_raw_html(self):
        for book_id in self.book_ids:
            with self.subTest(book_id=book_id):
                data = self._load(book_id)
                subjects = data.get('subjects', '')
                self.assertNotIn('<a', subjects)
                self.assertNotIn('</a>', subjects)


class TestExtractorKnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.extractor = Extractor(cls.config)

    def _load(self, book_id):
        path = os.path.join(self.config.OUTPUT_DIR, f"{book_id}.html")
        if not os.path.exists(path):
            self.skipTest(f"{path} not present on disk (book not crawled yet).")
        with open(path, 'r', encoding='utf-8') as f:
            html_text = f.read()
        return self.extractor.extract(html_text, book_id)

    def test_known_titles_and_authors(self):
        for book_id, expected in EXPECTED.items():
            with self.subTest(book_id=book_id):
                data = self._load(book_id)
                for field, expected_value in expected.items():
                    self.assertEqual(data.get(field, ''), expected_value)


if __name__ == '__main__':
    unittest.main()