"""
spark/spark_join.py — join Gutenberg books with Wikipedia pages.

Output: joined.parquet with EXACTLY ONE ROW PER BOOK (78,809 rows).
    - For books with a Wikipedia URL, the best matching wiki page is
      attached (exact title match preferred over "(novel)" / "(film)").
    - For books without a Wikipedia URL (or no match), wiki columns
      are NULL.

Uses NATIVE Spark functions (no Python UDF) to avoid Python worker crashes.

Stats printed:
    - total rows (should equal 78,809)
    - matched rows (books that got a wiki page) — same as before
    - unique books matched
    - unique wiki pages used
    - coverage (with URL) — same as before
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

from spark_config import SparkConfig


def main():
    spark = SparkSession.builder \
        .appName(SparkConfig.APP_NAME + " - join") \
        .master("local[4]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", 24) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if not os.path.exists(SparkConfig.GUTENBERG_PARQUET):
        print(f"ERROR: {SparkConfig.GUTENBERG_PARQUET} not found.")
        spark.stop()
        return
    if not os.path.exists(SparkConfig.WIKI_PARQUET):
        print(f"ERROR: {SparkConfig.WIKI_PARQUET} not found.")
        spark.stop()
        return

    print(f"Reading: {SparkConfig.GUTENBERG_PARQUET}")
    gutenberg = spark.read.parquet(SparkConfig.GUTENBERG_PARQUET)

    print(f"Reading: {SparkConfig.WIKI_PARQUET}")
    wiki = spark.read.parquet(SparkConfig.WIKI_PARQUET)

    gutenberg_count = gutenberg.count()
    wiki_count = wiki.count()
    print()
    print("=" * 70)
    print("INPUT")
    print("=" * 70)
    print(f"  Gutenberg books:           {gutenberg_count:,}")
    print(f"  Wikipedia pages:           {wiki_count:,}")

    # -------------------------------------------------------------------
    # Gutenberg side: extract wiki title from URL, normalize, rename.
    # Keep ALL books (not just those with a URL) — we want one row
    # per book in the final output.
    # -------------------------------------------------------------------
    g = gutenberg \
        .withColumn(
            "wiki_title_raw",
            F.when(
                F.col("wikipedia_url").isNotNull() & (F.col("wikipedia_url") != ""),
                F.regexp_extract(F.col("wikipedia_url"), r'/wiki/([^?#]+)', 1)
            ).otherwise(F.lit(None))
        ) \
        .withColumn(
            "join_key",
            F.when(
                F.col("wiki_title_raw").isNotNull() & (F.col("wiki_title_raw") != ""),
                F.trim(
                    F.regexp_replace(
                        F.regexp_replace(
                            F.regexp_replace(
                                F.lower(F.col("wiki_title_raw")),
                                "_", " "
                            ),
                            r"\s+", " "
                        ),
                        r"\s*\([^)]*\)\s*$", ""
                    )
                )
            ).otherwise(F.lit(None))
        ) \
        .select(
            "join_key",
            F.col("gutenberg_id"),
            F.col("title").alias("gutenberg_title"),
            F.col("author").alias("gutenberg_author"),
            F.col("language").alias("gutenberg_language"),
            F.col("release_date").alias("gutenberg_release_date"),
            F.col("subjects").alias("gutenberg_subjects"),
            F.col("loc_class").alias("gutenberg_loc_class"),
            F.col("wikipedia_url").alias("gutenberg_wikipedia_url"),
        )

    g_with_url = g.filter(F.col("join_key").isNotNull()).count()
    print(f"  Gutenberg with wiki URL:   {g_with_url:,}")

    # -------------------------------------------------------------------
    # Wiki side: normalize title, rename, and add priority.
    # priority = 1 for exact title (no parentheses),
    #          = 2 for "(novel)" / "(film)" / etc.
    # -------------------------------------------------------------------
    w = wiki \
        .withColumn(
            "join_key",
            F.trim(
                F.regexp_replace(
                    F.regexp_replace(
                        F.regexp_replace(
                            F.lower(F.col("title")),
                            "_", " "
                        ),
                        r"\s+", " "
                    ),
                    r"\s*\([^)]*\)\s*$", ""
                )
            )
        ) \
        .withColumn(
            "priority",
            F.when(F.col("title").rlike(r"^[^(]+$"), 1).otherwise(2)
        ) \
        .select(
            "join_key",
            "priority",
            F.col("page_id").alias("wiki_page_id"),
            F.col("title").alias("wiki_title"),
            F.col("abstract").alias("wiki_abstract"),
            F.col("categories").alias("wiki_categories"),
            F.col("links").alias("wiki_links"),
            F.col("infobox").alias("wiki_infobox"),
        )

    # -------------------------------------------------------------------
    # Step 1: For each book with a join_key, pick the BEST wiki page
    #         (lowest priority number; tie-break by page_id for determinism).
    # -------------------------------------------------------------------
    window = Window.partitionBy("gutenberg_id").orderBy(
        F.col("priority").asc(),
        F.col("wiki_page_id").asc(),
    )

    best_wiki_per_book = g \
    .filter(F.col("join_key").isNotNull()) \
    .join(w, on="join_key", how="inner") \
    .withColumn("rn", F.row_number().over(window)) \
    .filter(F.col("rn") == 1) \
    .drop("rn", "priority", "join_key") \
    .select(
        "gutenberg_id",
        "wiki_page_id",
        "wiki_title",
        "wiki_abstract",
        "wiki_categories",
        "wiki_links",
        "wiki_infobox",
    )

    # -------------------------------------------------------------------
    # Step 2: LEFT JOIN back to ALL books so books without a wiki
    #         still appear (with NULL wiki columns).
    #         Result: exactly one row per book = 78,809 rows.
    # -------------------------------------------------------------------
    g_all = gutenberg.select(
        F.col("gutenberg_id"),
        F.col("title").alias("gutenberg_title"),
        F.col("author").alias("gutenberg_author"),
        F.col("language").alias("gutenberg_language"),
        F.col("release_date").alias("gutenberg_release_date"),
        F.col("subjects").alias("gutenberg_subjects"),
        F.col("loc_class").alias("gutenberg_loc_class"),
        F.col("wikipedia_url").alias("gutenberg_wikipedia_url"),
    )

    joined = g_all.join(best_wiki_per_book, on="gutenberg_id", how="left")

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------
    total_rows = joined.count()
    matched_rows = joined.filter(F.col("wiki_page_id").isNotNull()).count()
    unique_books_total = joined.select("gutenberg_id").distinct().count()
    unique_wiki_matched = joined \
        .filter(F.col("wiki_page_id").isNotNull()) \
        .select("wiki_page_id").distinct().count()

    print()
    print("=" * 70)
    print("JOIN RESULT")
    print("=" * 70)
    print(f"  Total rows (one per book):       {total_rows:,}")
    print(f"  Unique Gutenberg books:          {unique_books_total:,}")
    print()
    print("  --- MATCHED ---")
    print(f"  Books with a wiki page:          {matched_rows:,}")
    print(f"  Unique Wikipedia pages used:     {unique_wiki_matched:,}")
    if g_with_url > 0:
        print(f"  Coverage (with URL):             "
              f"{100*matched_rows/g_with_url:.2f}%")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------------
    print()
    print("Sample rows (with wiki):")
    joined.filter(F.col("wiki_page_id").isNotNull()).show(5, truncate=40)

    print()
    print("Sample rows (without wiki):")
    joined.filter(F.col("wiki_page_id").isNull()).show(5, truncate=40)

    # -------------------------------------------------------------------
    # Write — one row per book
    # -------------------------------------------------------------------
    joined.write.mode("overwrite").parquet(SparkConfig.JOINED_PARQUET)
    print(f"\nWrote: {SparkConfig.JOINED_PARQUET}")

    with_abstract = joined.filter(
        F.col("wiki_abstract").isNotNull() & (F.col("wiki_abstract") != "")
    ).count()
    print(f"Rows with wiki abstract: {with_abstract:,} "
          f"({100*with_abstract/total_rows:.1f}% of all rows)")

    spark.stop()


if __name__ == "__main__":
    main()