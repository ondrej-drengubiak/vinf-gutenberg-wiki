"""
spark/spark_export.py — export joined.parquet (+ downloads) to JSON lines.

Why: the own fielded index (index2/) is pure Python and cannot read Parquet
without pyarrow/pandas, so Spark writes the joined data as JSON lines once.

Output: spark_output/joined_jsonl/part-*.json   (one book per line)

Usage (from project root):
    python spark/spark_export.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession

from spark_config import SparkConfig

JSONL_DIR = os.path.join(SparkConfig.OUTPUT_DIR, "joined_jsonl")


def main():
    spark = SparkSession.builder \
        .appName(SparkConfig.APP_NAME + " - export") \
        .master("local[4]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    for p in (SparkConfig.JOINED_PARQUET, SparkConfig.GUTENBERG_PARQUET):
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run the earlier Spark jobs first.")
            spark.stop()
            return

    joined = spark.read.parquet(SparkConfig.JOINED_PARQUET)
    # 'downloads' is not part of the join output -> take it from the books table.
    downloads = spark.read.parquet(SparkConfig.GUTENBERG_PARQUET) \
        .select("gutenberg_id", "downloads")

    out = joined.join(downloads, on="gutenberg_id", how="left")
    print(f"Rows to export: {out.count():,}")

    out.coalesce(1).write.mode("overwrite").json(JSONL_DIR)
    print(f"Wrote: {JSONL_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()