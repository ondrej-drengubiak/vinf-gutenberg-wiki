"""
index2/index_search.py — interactive search + statistics for the fielded index.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fielded_index import FieldedIndex
from index_schema import INDEX_FILE, FIELDS
from query_parser import QueryParseError

HELP = """\
Query syntax (Lucene-like):
  whale                      term in all default fields
  title:whale  author:poe    term in one field
  "great expectations"       phrase
  title:whal*                prefix
  title:frankenstien~1       fuzzy (max 1 edit)
  pub_year:[1850 TO 1900]    numeric range ('*' = open end)
  a AND b | a OR b | NOT a   boolean, also +a -a and ( )
  title:whale^3              boost
Fields: """ + ", ".join(FIELDS) + """
Commands: :or / :and = default operator between clauses, :k N = top-K, :help"""


def load_index():
    if not os.path.exists(INDEX_FILE):
        print(f"ERROR: {INDEX_FILE} not found. Run 'Build Fielded Index' (16) first.")
        return None
    return FieldedIndex.load(INDEX_FILE)


def format_result(rank, doc, score):
    year = doc['pub_year'] if doc['pub_year'] else '-'
    return (f"{rank:>3}. {score:8.3f}  {doc['title'][:58]:<58}  "
            f"{doc['author'][:28]:<28}  [#{doc['gutenberg_id']}, "
            f"{doc['language'] or '?'}, {year}]")


def print_results(idx, query, hits, total, ms):
    print(f"\n  '{query}' -> {total:,} hits, {ms:.1f} ms")
    for rank, (d, s) in enumerate(hits, 1):
        print("  " + format_result(rank, idx.docs[d], s))


def interactive():
    idx = load_index()
    if idx is None:
        return
    print(f"Loaded {idx.N:,} documents.")
    print(HELP)
    op, top_k = 'AND', 10
    while True:
        q = input(f"\nQuery [{op}, top {top_k}] (empty = back): ").strip()
        if not q:
            print("Back to menu.")
            return
        if q == ':help':
            print(HELP)
            continue
        if q in (':or', ':and'):
            op = q[1:].upper()
            continue
        if q.startswith(':k '):
            try:
                top_k = int(q[3:])
            except ValueError:
                print("Usage: :k 5")
            continue
        t0 = time.perf_counter()
        try:
            hits, total = idx.search(q, top_k=top_k, default_op=op)
        except (QueryParseError, ValueError) as e:
            print(f"  Query error: {e}")
            continue
        print_results(idx, q, hits, total, (time.perf_counter() - t0) * 1000)


def show_stats():
    idx = load_index()
    if idx is None:
        return
    s = idx.stats()
    print("=" * 70)
    print("FIELDED INDEX STATISTICS")
    print("=" * 70)
    print(f"  Documents: {s['documents']:,}   "
          f"File: {os.path.getsize(INDEX_FILE) / 1024 / 1024:.1f} MB")
    print("\n  TEXT fields            docs      vocabulary    postings")
    for f, v in s['text'].items():
        print(f"    {f:<18}{v['docs_with_field']:>9,}{v['vocabulary']:>14,}{v['postings']:>12,}")
    print("\n  KEYWORD fields         docs      distinct values")
    for f, v in s['keyword'].items():
        print(f"    {f:<18}{v['docs']:>9,}{v['distinct_values']:>14,}")
    print("\n  NUMERIC fields         docs      min .. max")
    for f, v in s['numeric'].items():
        print(f"    {f:<18}{v['docs']:>9,}      {v['min']} .. {v['max']}")
    print("=" * 70)


if __name__ == '__main__':
    interactive()
