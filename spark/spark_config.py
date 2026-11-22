"""

spark/spark_config.py

Configuration for Spark jobs.

Separate from the main Config because Spark paths are different:
input is the same Gutenberg data, but output goes to Parquet.
"""

import os

# Anchor paths to the project root (parent of spark/), so the script
# works no matter where it's launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SparkConfig:
    # Spark
    APP_NAME = "VINF-Gutenberg-Wiki"
    MASTER = "local[*]"
    SHUFFLE_PARTITIONS = 12

    # Input (from 1st submission)
    GUTENBERG_DIR = os.path.join(_PROJECT_ROOT, "gutenberg_data")
    GUTENBERG_TSV = os.path.join(_PROJECT_ROOT, "gutenberg.tsv")

    # Output (Spark jobs write Parquet)
    OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "spark_output")
    GUTENBERG_PARQUET = os.path.join(OUTPUT_DIR, "gutenberg_books.parquet")
    WIKI_PARQUET = os.path.join(OUTPUT_DIR, "wiki_pages.parquet")
    JOINED_PARQUET = os.path.join(OUTPUT_DIR, "joined.parquet")

    # Wikipedia dump (FULL dump only — no sample)
    WIKI_DIR = os.path.join(_PROJECT_ROOT, "wiki_data")
    WIKI_FULL_XML = os.path.join(WIKI_DIR, "enwiki_full.xml")