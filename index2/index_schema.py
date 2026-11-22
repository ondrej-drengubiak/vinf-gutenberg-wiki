"""
index2/index_schema.py — field definitions of the fielded (Lucene-style) index.

One place that says WHICH fields exist, of WHICH type and with WHICH boost.
Everything else (builder, index, query parser) reads this file.

Field types (same idea as Lucene):
    TEXT     tokenized, positions stored  -> term, phrase, fuzzy, prefix queries
    KEYWORD  whole value, lower-cased     -> exact match (language:english)
    NUMERIC  integer, kept sorted         -> range queries (pub_year:[1850 TO 1900])
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Spark export (joined.parquet -> JSON lines) and the built index.
JSONL_DIR = os.path.join(_ROOT, "spark_output", "joined_jsonl")
INDEX_DIR = os.path.join(_ROOT, "index2_output")
INDEX_FILE = os.path.join(INDEX_DIR, "fielded_index.pkl")

TEXT, KEYWORD, NUMERIC = "text", "keyword", "numeric"

# Gap between values of a multi-valued field (subjects, categories), so a
# phrase query can never match across two different values.
POSITION_GAP = 100

# name -> (type, boost)
FIELDS = {
    "title":        (TEXT,    3.0),   # what the user usually types
    "author":       (TEXT,    2.0),
    "subjects":     (TEXT,    1.5),   # Gutenberg subject headings
    "loc_class":    (TEXT,    1.0),   # Library of Congress class
    "abstract":     (TEXT,    1.0),   # Wikipedia abstract (description search)
    "categories":   (TEXT,    1.5),   # Wikipedia categories
    "publisher":    (TEXT,    1.0),   # from Wikipedia infobox
    "genre":        (TEXT,    1.5),   # from Wikipedia infobox
    "language":     (KEYWORD, 1.0),
    "has_wiki":     (KEYWORD, 1.0),   # yes / no
    "pub_year":     (NUMERIC, 1.0),   # first publication year (infobox)
    "release_year": (NUMERIC, 1.0),   # year of the Gutenberg e-book release
    "downloads":    (NUMERIC, 1.0),
}

# Searched when the query has no "field:" prefix.
DEFAULT_FIELDS = ["title", "author", "subjects", "categories",
                  "abstract", "genre", "publisher"]

# Kept per document only for printing results (not searchable by themselves).
STORED = ["gutenberg_id", "title", "author", "language", "pub_year",
          "downloads", "wiki_title"]
