"""

spark/spark_extractor.py
spark_extractor.py — Spark version of extractor.py.

Reads every .html file from gutenberg_data/, applies the SAME regex
patterns as the original extractor.py, and writes a Parquet file with
one row per book.

Same regex, same output schema, just parallel across Spark executors.

Usage (from project root):
    python spark/spark_extractor.py
"""

import os
import re
import sys

# Make sure `from spark_config import ...` works when launched from the
# project root (python spark/spark_extractor.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from spark_config import SparkConfig


# ---------------------------------------------------------------------------
# Regex patterns — identical to extractor.py (1st submission).
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(r'&(#x[0-9A-Fa-f]+|#[0-9]+|[A-Za-z]+);')
_NAMED_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&apos;': "'",
    '&nbsp;': ' ', '&mdash;': '—', '&ndash;': '–', '&hellip;': '…',
    '&lsquo;': '‘', '&rsquo;': '’', '&ldquo;': '“', '&rdquo;': '”',
    '&copy;': '©', '&deg;': '°', '&pound;': '£', '&euro;': '€',
}


def _decode_entity(m):
    body = m.group(1)
    if not body.startswith('#'):
        return _NAMED_ENTITIES.get('&' + body + ';', m.group(0))
    try:
        code = int(body[2:], 16) if body[1] in 'xX' else int(body[1:])
        return chr(code)
    except (ValueError, OverflowError):
        return m.group(0)


def clean(text):
    """Same as extractor.clean() — drop tags, decode entities, normalise ws."""
    if not text:
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = _ENTITY_RE.sub(_decode_entity, text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


_PATTERNS = {
    'title':        re.compile(r'<th>Title</th>\s*<td[^>]*>\s*(.*?)\s*</td>', re.DOTALL),
    'author':       re.compile(r'<a href="/ebooks/author/\d+"[^>]*>(.*?)</a>', re.DOTALL),
    'language':     re.compile(r'<th>Language</th>\s*<td>(.*?)</td>', re.DOTALL),
    'release_date': re.compile(r'<th>Release Date</th>\s*<td[^>]*>(.*?)</td>', re.DOTALL),
    'last_update':  re.compile(r'<th>Last Update</th>\s*<td[^>]*>(.*?)</td>', re.DOTALL),
    'loc_class':    re.compile(r'<th>LoC Class</th>\s*<td>\s*<a[^>]*>(.*?)</a>', re.DOTALL),
    'downloads':    re.compile(r'<th>Downloads</th>\s*<td[^>]*>(.*?)</td>', re.DOTALL),
    'reading_ease': re.compile(r'<th>Reading Level</th>\s*<td>\s*(.*?)\s*</td>', re.DOTALL),
    'credits':      re.compile(r'<th>Credits</th>\s*<td>\s*(.*?)\s*</td>', re.DOTALL),
    'wikipedia_url':re.compile(r'Wikipedia page about this book:\s*<a href="([^"]+)"'),
    'text_url':     re.compile(r'href="(/ebooks/\d+\.txt\.utf-8)"'),
}
_SUBJECT_RE = re.compile(r'<th>Subject</th>\s*<td[^>]*>.*?<a[^>]*>(.*?)</a>', re.DOTALL)


def extract_one(pair):
    """
    Extract metadata from one (path, html_text) pair.

    Spark's wholeTextFiles gives (path, content). Derive book_id from
    the path, run the same regexes as the original extractor.
    """
    path, html_text = pair

    filename = os.path.basename(path)
    book_id = filename[:-len('.html')] if filename.endswith('.html') else filename

    def _get(field):
        m = _PATTERNS[field].search(html_text)
        return clean(m.group(1)) if m else ''

    title = _get('title')
    author = _get('author')
    language = _get('language')
    release_date = _get('release_date')
    last_update = _get('last_update')

    subjects = [clean(m.group(1)) for m in _SUBJECT_RE.finditer(html_text)]
    subjects_str = '|'.join(subjects)

    loc_class = _get('loc_class')
    downloads = _get('downloads')
    reading_ease = _get('reading_ease')
    credits = _get('credits')
    wikipedia_url = _get('wikipedia_url')

    text_url = ''
    m = _PATTERNS['text_url'].search(html_text)
    if m:
        text_url = 'https://www.gutenberg.org' + m.group(1)

    return (
        book_id, title, author, language, release_date, last_update,
        subjects_str, loc_class, downloads, reading_ease, credits,
        wikipedia_url, text_url,
    )


OUTPUT_SCHEMA = StructType([
    StructField('gutenberg_id', StringType(), False),
    StructField('title',        StringType(), True),
    StructField('author',       StringType(), True),
    StructField('language',     StringType(), True),
    StructField('release_date', StringType(), True),
    StructField('last_update',  StringType(), True),
    StructField('subjects',     StringType(), True),
    StructField('loc_class',    StringType(), True),
    StructField('downloads',    StringType(), True),
    StructField('reading_ease', StringType(), True),
    StructField('credits',      StringType(), True),
    StructField('wikipedia_url',StringType(), True),
    StructField('text_url',     StringType(), True),
])


def main():
    spark = SparkSession.builder \
        .appName(SparkConfig.APP_NAME + " - extractor") \
        .master(SparkConfig.MASTER) \
        .config("spark.sql.shuffle.partitions", SparkConfig.SHUFFLE_PARTITIONS) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import inspect
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    spark.sparkContext.addPyFile(os.path.join(current_dir, "spark_extractor.py"))
    spark.sparkContext.addPyFile(os.path.join(current_dir, "spark_config.py"))

    html_glob = os.path.join(SparkConfig.GUTENBERG_DIR, "*.html")
    print(f"Reading: {html_glob}")

    rdd = spark.sparkContext.wholeTextFiles(html_glob)
    total = rdd.count()
    print(f"Found {total} HTML files")

    extracted = rdd.map(extract_one)
    df = spark.createDataFrame(extracted, schema=OUTPUT_SCHEMA)

    print(f"Extracted rows: {df.count()}")
    print("Sample:")
    df.show(5, truncate=60)

    df.write.mode("overwrite").parquet(SparkConfig.GUTENBERG_PARQUET)
    print(f"Wrote: {SparkConfig.GUTENBERG_PARQUET}")

    spark.stop()


if __name__ == "__main__":
    main()