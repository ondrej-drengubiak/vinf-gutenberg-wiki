"""
spark/spark_wiki_parser.py — parse Wikipedia XML dump with REGEX ONLY,
keeping ONLY pages that match a Gutenberg book (broadcast filter).

All parsing (page / title / text boundaries, entities, percent-decoding,
infobox extraction, categories, links, cleaning) is done with the `re`
module. No XML parser, no str.find / slicing for structure, no html.unescape,
no urllib.unquote. Output is the same as the previous str.find-based version.

Pipeline:
    1. Read gutenberg_books.parquet, build the set of normalized join keys
       (same normalization as spark_join.py) and broadcast it.
    2. Read the dump with a Hadoop record delimiter "</page>" (one record = one page).
    3. Check the title (regex) against the broadcast set.
    4. Only for matches run the heavy regexes (categories, links, infobox).

Output: spark_output/wiki_pages.parquet

Usage (from project root):
    python spark/spark_wiki_parser.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Workers must use the same interpreter as the driver (important on Windows).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from spark_config import SparkConfig


# ---------------------------------------------------------------------------
# Regex patterns — XML structure of the dump.
# ---------------------------------------------------------------------------

_PAGE_START_RE = re.compile(r'<page>')
_PAGE_ID_RE = re.compile(r'<id>(\d+)</id>')
_TITLE_RE = re.compile(r'<title>(.*?)</title>', re.DOTALL)
# <text ...attrs...> body </text>   (body runs to the end if </text> is missing)
_TEXT_RE = re.compile(r'<text[^>]*>(.*?)(?:</text>|\Z)', re.DOTALL)

# XML entities used by the dump (&amp; &lt; &gt; &quot; &apos; + numeric).
_XML_ENTITY_RE = re.compile(r'&(#[xX][0-9A-Fa-f]+|#[0-9]+|amp|lt|gt|quot|apos);')
_XML_ENTITIES = {'amp': '&', 'lt': '<', 'gt': '>', 'quot': '"', 'apos': "'"}

# Percent-encoded runs in URLs, e.g. Les_Mis%C3%A9rables
_PCT_RUN_RE = re.compile(r'(?:%[0-9A-Fa-f]{2})+')
_URL_TITLE_RE = re.compile(r'/wiki/([^?#]+)')
_UNDERSCORE_RE = re.compile(r'_')

# ---------------------------------------------------------------------------
# Regex patterns — MediaWiki markup.
# ---------------------------------------------------------------------------

_CATEGORY_RE = re.compile(r'\[\[Category:([^\]\|]+)(?:\|[^\]]*)?\]\]')
_LINK_RE = re.compile(r'\[\[([^\]\|#:]+)(?:[#\|][^\]]*)?\]\]')
_INFOBOX_START_RE = re.compile(r'\{\{\s*Infobox', re.IGNORECASE)
_BRACE_RE = re.compile(r'\{\{|\}\}')                 # tokenizer for brace matching
_SECTION_RE = re.compile(r'\n\s*==+[^=]+==+\s*\n')

_REF_RE = re.compile(r'<ref[^>]*?/>|<ref[^>]*>.*?</ref>', re.DOTALL | re.IGNORECASE)
_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)
_WIKI_LINK_RE = re.compile(r'\[\[(?:[^\]\|]+\|)?([^\]]+)\]\]')
_WIKI_BOLD_RE = re.compile(r"'{2,5}")
_WIKI_TEMPLATE_RE = re.compile(r'\{\{[^{}]*\}\}')   # innermost templates only
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_WHITESPACE_RE = re.compile(r'\s+')
_TRAILING_PAREN_RE = re.compile(r'\s*\([^)]*\)\s*$')


# ---------------------------------------------------------------------------
# Regex-based decoders (replace html.unescape and urllib.parse.unquote).
# ---------------------------------------------------------------------------

def _entity_repl(m):
    body = m.group(1)
    if body[0] != '#':
        return _XML_ENTITIES[body]
    try:
        code = int(body[2:], 16) if body[1] in 'xX' else int(body[1:])
    except ValueError:
        return m.group(0)
    if code == 0 or 0xD800 <= code <= 0xDFFF or code > 0x10FFFF:
        return '\uFFFD'
    return chr(code)


def decode_entities(s):
    """Decode XML entities with a regex (one pass, like html.unescape on the dump)."""
    return _XML_ENTITY_RE.sub(_entity_repl, s)


def _pct_repl(m):
    return bytes.fromhex(m.group(0).replace('%', '')).decode('utf-8', 'replace')


def percent_decode(s):
    """Decode %XX runs with a regex (UTF-8, invalid bytes replaced)."""
    return _PCT_RUN_RE.sub(_pct_repl, s)


# ---------------------------------------------------------------------------
# Join key normalization — MUST stay identical to spark_join.
# ---------------------------------------------------------------------------

def normalize_title(s):
    if s is None:
        return None
    s = _UNDERSCORE_RE.sub(' ', percent_decode(s)).lower()
    s = _WHITESPACE_RE.sub(' ', s).strip()
    s = _TRAILING_PAREN_RE.sub('', s)
    return s


def wiki_title_from_url(url):
    if not url:
        return None
    m = _URL_TITLE_RE.search(url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Wikitext processing.
# ---------------------------------------------------------------------------

def _strip_templates(text):
    """Remove nested {{...}} by repeatedly deleting the innermost ones."""
    for _ in range(12):
        new = _WIKI_TEMPLATE_RE.sub('', text)
        if new == text:
            break
        text = new
    return text


def _clean_wikitext(text, max_len=None):
    if not text:
        return ''
    text = _COMMENT_RE.sub('', text)
    text = _REF_RE.sub('', text)
    text = _strip_templates(text)
    text = _WIKI_LINK_RE.sub(r'\1', text)
    text = _WIKI_BOLD_RE.sub('', text)
    text = _HTML_TAG_RE.sub(' ', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    if max_len is not None and len(text) > max_len:
        text = text[:max_len].rsplit(' ', 1)[0] + '...'
    return text


def _extract_abstract(text):
    if not text:
        return ''
    m = _SECTION_RE.search(text)
    if m:
        text = text[:m.start()]
    return _clean_wikitext(text, max_len=1000)


def _extract_infobox(text, max_len=2000):
    """
    Brace matching with a regex tokenizer: _BRACE_RE yields every '{{' and '}}'
    from the infobox start, depth is tracked until it returns to 0
    (infoboxes contain nested {{...}}, which a single flat regex cannot match).
    """
    m = _INFOBOX_START_RE.search(text)
    if not m:
        return ''
    start = m.start()
    depth = 0
    for tok in _BRACE_RE.finditer(text, start):
        depth += 1 if tok.group(0) == '{{' else -1
        if depth == 0:
            return text[start:tok.end()][:max_len]
    return text[start:start + max_len]


def parse_page_block(body):
    """Parse the inside of one <page> (string after '<page>'). Heavy regexes."""
    id_m = _PAGE_ID_RE.search(body)
    title_m = _TITLE_RE.search(body)
    text_m = _TEXT_RE.search(body)
    if not id_m or not title_m or not text_m:
        return None

    wikitext = decode_entities(text_m.group(1))
    title = decode_entities(title_m.group(1))
    categories = '|'.join(sorted({c.strip() for c in _CATEGORY_RE.findall(wikitext)}))
    links = '|'.join(sorted({l.strip() for l in _LINK_RE.findall(wikitext)}))

    return (id_m.group(1), title[:300], _extract_abstract(wikitext),
            categories, links, _extract_infobox(wikitext))


OUTPUT_SCHEMA = StructType([
    StructField('page_id',    StringType(), False),
    StructField('title',      StringType(), True),
    StructField('abstract',   StringType(), True),
    StructField('categories', StringType(), True),
    StructField('links',      StringType(), True),
    StructField('infobox',    StringType(), True),
])


# ---------------------------------------------------------------------------
# Spark helpers
# ---------------------------------------------------------------------------

def load_join_keys(spark):
    """Collect normalized join keys of all Gutenberg books that have a wiki URL."""
    df = spark.read.parquet(SparkConfig.GUTENBERG_PARQUET) \
        .select("wikipedia_url") \
        .where("wikipedia_url IS NOT NULL AND wikipedia_url != ''")
    keys = set()
    for row in df.toLocalIterator():
        t = wiki_title_from_url(row[0])
        if t:
            keys.add(normalize_title(t))
    return keys


def make_record_parser(bc_keys):
    def parse_record(rec):
        page = _PAGE_START_RE.search(rec)
        if page is None:                  # file header or trailing </mediawiki>
            return None
        title_m = _TITLE_RE.search(rec, page.end())
        if title_m is None:
            return None
        title = decode_entities(title_m.group(1))
        if normalize_title(title) not in bc_keys.value:
            return None                   # ~99.9 % of pages stop here, cheaply
        return parse_page_block(rec[page.end():])
    return parse_record


def main(xml_path=None):
    spark = SparkSession.builder \
        .appName(SparkConfig.APP_NAME + " - wiki parser") \
        .master("local[4]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "1g") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .config("spark.sql.shuffle.partitions", 24) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Ship this module (normalize_title, parse_page_block, ...) to the workers.
    spark.sparkContext.addPyFile(os.path.abspath(__file__))
    spark.sparkContext.addPyFile(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "spark_config.py"))

    path = xml_path or SparkConfig.WIKI_FULL_XML
    if not os.path.exists(path):
        print(f"ERROR: wiki dump not found at {path}")
        spark.stop()
        return
    if not os.path.exists(SparkConfig.GUTENBERG_PARQUET):
        print(f"ERROR: {SparkConfig.GUTENBERG_PARQUET} not found. Run Spark Extractor first.")
        spark.stop()
        return

    print(f"Reading dump: {path}  ({os.path.getsize(path) / 1024**3:.2f} GB)")

    keys = load_join_keys(spark)
    print(f"Join keys from Gutenberg: {len(keys)}")
    bc_keys = spark.sparkContext.broadcast(keys)

    # One record = everything up to the next "</page>". Split size 64 MB.
    uri = "file:///" + os.path.abspath(path).replace("\\", "/")
    rdd = spark.sparkContext.newAPIHadoopFile(
        uri,
        "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
        "org.apache.hadoop.io.LongWritable",
        "org.apache.hadoop.io.Text",
        conf={
            "textinputformat.record.delimiter": "</page>",
            "mapreduce.input.fileinputformat.split.maxsize": str(64 * 1024 * 1024),
        },
    ).map(lambda kv: kv[1])

    parsed = rdd.map(make_record_parser(bc_keys)).filter(lambda x: x is not None)
    df = spark.createDataFrame(parsed, schema=OUTPUT_SCHEMA)

    # Single pass over the dump: write first, count afterwards from the small parquet.
    df.write.mode("overwrite").parquet(SparkConfig.WIKI_PARQUET)
    print(f"Wrote: {SparkConfig.WIKI_PARQUET}")

    out = spark.read.parquet(SparkConfig.WIKI_PARQUET)
    print(f"Matched wiki pages: {out.count()}")
    out.show(5, truncate=60)

    spark.stop()


if __name__ == "__main__":
    main()