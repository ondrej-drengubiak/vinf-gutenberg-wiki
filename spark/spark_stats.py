"""
spark/spark_stats.py — statistics and previews for Part 2 (Spark outputs).

Reads the Parquet files written by spark_extractor / spark_wiki_parser /
spark_join and prints report-friendly numbers.

Used from main.py:  show_stats()  and  preview(name, n)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, functions as F

from spark_config import SparkConfig

WIDTH = 70

PATHS = {
    'gutenberg': SparkConfig.GUTENBERG_PARQUET,
    'wiki': SparkConfig.WIKI_PARQUET,
    'joined': SparkConfig.JOINED_PARQUET,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title):
    print()
    print('-' * WIDTH)
    print(title)
    print('-' * WIDTH)


def _kv(label, value, pct_of=None):
    value = value or 0
    s = f"  {label:<38}{value:>12,}"
    if pct_of:
        s += f"   ({100 * value / pct_of:5.1f} %)"
    print(s)


def _kv_f(label, value, unit=''):
    value = value or 0.0
    print(f"  {label:<38}{value:>12,.1f} {unit}")


def _nonempty(col):
    return F.col(col).isNotNull() & (F.col(col) != '')


def _count_if(cond):
    return F.sum(F.when(cond, 1).otherwise(0))


def _dir_info(path):
    """(number of .parquet files, total size in MB) of a Parquet directory."""
    n_files, total = 0, 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
            if f.endswith('.parquet'):
                n_files += 1
    return n_files, total / 1024 / 1024


def _get_spark():
    spark = SparkSession.builder \
        .appName(SparkConfig.APP_NAME + " - stats") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.shuffle.partitions", 8) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _missing(path, hint):
    print(f"  Not found: {path}")
    print(f"  {hint}")


# ---------------------------------------------------------------------------
# Individual statistics blocks
# ---------------------------------------------------------------------------

def _gutenberg_stats(spark):
    _section("1) GUTENBERG BOOKS  (Spark Extractor output)")
    path = PATHS['gutenberg']
    if not os.path.exists(path):
        _missing(path, "Run 'Spark Extractor' (10) first.")
        return None

    n_files, mb = _dir_info(path)
    df = spark.read.parquet(path)
    r = df.agg(
        F.count('*').alias('rows'),
        F.countDistinct('author').alias('authors'),
        _count_if(_nonempty('title')).alias('title'),
        _count_if(_nonempty('author')).alias('author'),
        _count_if(_nonempty('subjects')).alias('subjects'),
        _count_if(_nonempty('wikipedia_url')).alias('wiki_url'),
        _count_if(_nonempty('text_url')).alias('text_url'),
    ).first()

    total = r['rows']
    print(f"  Parquet: {n_files} files, {mb:.2f} MB")
    _kv("Books (rows)", total)
    _kv("Distinct authors", r['authors'])
    _kv("With title", r['title'], total)
    _kv("With author", r['author'], total)
    _kv("With subjects", r['subjects'], total)
    _kv("With text URL", r['text_url'], total)
    _kv("With Wikipedia URL (joinable)", r['wiki_url'], total)

    print("\n  Top languages:")
    for row in df.groupBy('language').count().orderBy(F.desc('count')).limit(5).collect():
        print(f"    {(row['language'] or '(empty)'):<30}{row['count']:>10,}")

    return total, r['wiki_url']


def _wiki_stats(spark):
    _section("2) WIKIPEDIA PAGES  (Wiki Parser output, filtered by Gutenberg titles)")
    path = PATHS['wiki']
    if not os.path.exists(path):
        _missing(path, "Run 'Wiki Parser' (11) first.")
        return

    n_files, mb = _dir_info(path)
    df = spark.read.parquet(path)
    cat_count = F.when(F.col('categories') != '',
                       F.size(F.split('categories', r'\|'))).otherwise(0)
    link_count = F.when(F.col('links') != '',
                        F.size(F.split('links', r'\|'))).otherwise(0)
    r = df.agg(
        F.count('*').alias('rows'),
        F.countDistinct('page_id').alias('unique'),
        _count_if(_nonempty('abstract')).alias('abstract'),
        _count_if(_nonempty('categories')).alias('categories'),
        _count_if(_nonempty('links')).alias('links'),
        _count_if(_nonempty('infobox')).alias('infobox'),
        F.avg(F.length('abstract')).alias('abstract_len'),
        F.avg(cat_count).alias('avg_cats'),
        F.avg(link_count).alias('avg_links'),
    ).first()

    total = r['rows']
    print(f"  Parquet: {n_files} files, {mb:.2f} MB")
    _kv("Matched wiki pages (rows)", total)
    _kv("Unique page_id", r['unique'])
    _kv("With abstract", r['abstract'], total)
    _kv("With categories", r['categories'], total)
    _kv("With links", r['links'], total)
    _kv("With infobox", r['infobox'], total)
    _kv_f("Avg abstract length", r['abstract_len'], "chars")
    _kv_f("Avg categories per page", r['avg_cats'])
    _kv_f("Avg links per page", r['avg_links'])


def _join_stats(spark, g_total, g_with_url):
    _section("3) JOIN  (Gutenberg + Wikipedia)")
    path = PATHS['joined']
    if not os.path.exists(path):
        _missing(path, "Run 'Join' (12) first.")
        return

    n_files, mb = _dir_info(path)
    j = spark.read.parquet(path)

    matched = j.filter(F.col('wiki_page_id').isNotNull())
    rows = j.count()  # 78,809 
    books_in_output = j.select('gutenberg_id').distinct().count()  # 78,809 
    m_rows = matched.count()  # 8,678 (po dedupe)

    m_books = matched.select('gutenberg_id').distinct().count()
    m_pages = matched.select('wiki_page_id').distinct().count()
    multi = matched.groupBy('gutenberg_id').count().filter('count > 1').count()

    print(f"  Parquet: {n_files} files, {mb:.2f} MB")
    _kv("Rows in joined output", rows)
    _kv("Books in joined output", books_in_output)
    print()
    _kv("Matched rows", m_rows)
    _kv("Unique Gutenberg books joined", m_books)
    _kv("Unique Wikipedia pages joined", m_pages)
    _kv("Books with >1 wiki candidate", multi, m_books)
    if g_total:
        _kv("Coverage of all books", m_books, g_total)
    if g_with_url:
        _kv("Coverage of books with wiki URL", m_books, g_with_url)

    r = matched.agg(
        _count_if(_nonempty('wiki_abstract')).alias('abstract'),
        _count_if(_nonempty('wiki_categories')).alias('categories'),
        _count_if(_nonempty('wiki_infobox')).alias('infobox'),
    ).first()
    print()
    _kv("Matched rows with abstract", r['abstract'], m_rows)
    _kv("Matched rows with categories", r['categories'], m_rows)
    _kv("Matched rows with infobox", r['infobox'], m_rows)

    print("\n  Top 10 Wikipedia categories among joined books:")
    top = matched \
        .select(F.explode(F.split('wiki_categories', r'\|')).alias('cat')) \
        .filter(F.col('cat') != '') \
        .groupBy('cat').count().orderBy(F.desc('count')).limit(10).collect()
    for row in top:
        print(f"    {row['cat'][:50]:<52}{row['count']:>8,}")


# ---------------------------------------------------------------------------
# Public API (used by main.py)
# ---------------------------------------------------------------------------

def show_stats():
    if not any(os.path.exists(p) for p in PATHS.values()):
        print("  No Parquet output yet. Run the Part 2 jobs (10, 11, 12) first.")
        return

    spark = _get_spark()
    try:
        g = _gutenberg_stats(spark)
        _wiki_stats(spark)
        _join_stats(spark, g[0] if g else None, g[1] if g else None)
    finally:
        spark.stop()
    print()
    print('=' * WIDTH)


def preview(name, n=5):
    path = PATHS[name]
    if not os.path.exists(path):
        print(f"  Not found: {path}")
        print("  Run the corresponding Spark job first.")
        return

    spark = _get_spark()
    try:
        df = spark.read.parquet(path)
        print(f"\n  {name}: {len(df.columns)} columns")
        print(f"  Columns: {', '.join(df.columns)}\n")
        df.show(n, truncate=60, vertical=True)
    finally:
        spark.stop()