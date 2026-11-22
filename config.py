"""

config.py

"""

class Config:
    # HTTP
    HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36 Edg/155.0.0.0 '
        'VINF-SchoolProject/1.0 (ondrej.drengubiak1@gmail.com; student project at FIIT STU)'
    )
}
    TIMEOUT_HTML = 20
    TIMEOUT_TEXT = 30
    MAX_RETRIES = 3
    DELAY = 2

    # Paths
    OUTPUT_DIR = 'gutenberg_data'
    TSV_FILE = 'gutenberg.tsv'
    PROGRESS_FILE = 'progress.json'
    QUEUE_FILE = 'queue.json'       # fronta URL pre crawler
    VISITED_FILE = 'visited.json'   # navštívené URL

    # Seed URLs
    SEED_URLS = [
        'https://www.gutenberg.org/ebooks/bookshelf/13',
        'https://www.gutenberg.org/ebooks/bookshelf/40',
        'https://www.gutenberg.org/ebooks/bookshelf/639',
        'https://www.gutenberg.org/ebooks/bookshelf/645',
        'https://www.gutenberg.org/ebooks/bookshelf/649',
    ]

    # Domain restriction
    ALLOWED_DOMAIN = 'www.gutenberg.org'

    # Book URL pattern
    BOOK_URL_PATTERN = r'^https?://www\.gutenberg\.org/ebooks/\d+$'

    # TSV fields
    TSV_FIELDS = [
        'gutenberg_id', 'title', 'author', 'language', 'release_date',
        'last_update', 'subjects', 'loc_class', 'downloads',
        'reading_ease', 'credits', 'wikipedia_url', 'text_url',
        'html_size', 'text_size'
    ]

    # Index files , separate for metadata and text
    INDEX_METADATA_FILE = 'index_metadata.pkl'
    INDEX_TEXT_FILE = 'index_text.pkl'

    # Metadata fields (from TSV) that get tokenized and indexed.
    INDEX_METADATA_FIELDS = ['title', 'author', 'subjects', 'loc_class']

    # Whether to also index the full plain-text body of each book
    INDEX_FULL_TEXT = True

    # Safety cap on how many characters of the body text are read
    # per book (0 = no limit, read the whole file).
    INDEX_TEXT_MAX_CHARS = 0

    # Minimum token length kept in the index
    MIN_TOKEN_LEN = 2

    # How often (in number of documents) the index is flushed to disk
    INDEX_SAVE_EVERY = 500