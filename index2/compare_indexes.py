"""
index2/compare_indexes.py — OLD index (1st submission) vs NEW fielded index.

For every test case we run
    - the plain free-text query on the old searcher (metadata index, 'sublinear')
    - the structured query on the new fielded index
and print top-5 titles, number of hits, time and overlap@5 as TEXT
(the assignment wants text tables, not screenshots).
The same report is written to index2_output/comparison.txt.

Edit TEST_CASES to your own queries; the old query is what a user without
fields / phrases / ranges would type.
"""

import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # project root: config, searcher

from fielded_index import FieldedIndex
from index_schema import INDEX_FILE, INDEX_DIR

TOP = 5

# (what it tests, old free-text query, new structured query)
TEST_CASES = [
    ("author + topic",
     "dickens london",
     'author:dickens AND (subjects:london OR abstract:london)'),
    ("exact phrase in title",
     "great expectations",
     'title:"great expectations"'),
    ("typo (fuzzy)",
     "frankenstien",
     'title:frankenstien~2'),
    ("year range + genre",
     "novel 1850 1870",
     'genre:novel AND pub_year:[1850 TO 1870]'),
    ("Wikipedia category + language",
     "novels adapted into films english",
     'categories:"adapted into films" AND language:english'),
    ("boolean NOT",
     "adventure sea french",
     'subjects:adventure AND subjects:sea AND NOT language:french'),
    ("description in abstract",
     "novel about a white whale revenge",
     'abstract:whale AND abstract:revenge'),
]


def _old_id(r):
    """Old Searcher result -> gutenberg id as string (tolerant to the format)."""
    if isinstance(r, (tuple, list)):
        return str(r[0])
    if isinstance(r, dict):
        for key in ('gutenberg_id', 'doc_id', 'book_id', 'id'):
            if key in r:
                return str(r[key])
    return str(getattr(r, 'doc_id', r))


def _load_titles(config):
    titles = {}
    if os.path.exists(config.TSV_FILE):
        with open(config.TSV_FILE, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                titles[row['gutenberg_id']] = row.get('title', '')
    return titles


def main():
    from config import Config
    from searcher import Searcher

    if not os.path.exists(INDEX_FILE):
        print(f"ERROR: {INDEX_FILE} not found. Run 'Build Fielded Index' (16) first.")
        return
    config = Config()
    try:
        old = Searcher(config, index_type='metadata')
    except RuntimeError as e:
        print(f"ERROR (old index): {e}")
        return
    new = FieldedIndex.load(INDEX_FILE)
    titles = _load_titles(config)

    lines = []
    out = lines.append
    out("=" * 78)
    out("OLD INDEX (metadata, own TF-IDF) vs NEW FIELDED INDEX")
    out("=" * 78)

    sum_old_ms = sum_new_ms = 0.0
    for n, (label, q_old, q_new) in enumerate(TEST_CASES, 1):
        t0 = time.perf_counter()
        old_res = old.search(q_old, method='sublinear', top_k=TOP)
        old_ms = (time.perf_counter() - t0) * 1000
        old_ids = [_old_id(r) for r in old_res]

        t0 = time.perf_counter()
        try:
            hits, total = new.search(q_new, top_k=TOP)
            err = None
        except ValueError as e:
            hits, total, err = [], 0, str(e)
        new_ms = (time.perf_counter() - t0) * 1000
        new_ids = [new.docs[d]['gutenberg_id'] for d, _ in hits]

        sum_old_ms += old_ms
        sum_new_ms += new_ms
        overlap = len(set(old_ids) & set(new_ids))

        out("")
        out(f"[{n}] {label}")
        out(f"    OLD  query: {q_old}")
        out(f"    NEW  query: {q_new}")
        out(f"    time: old {old_ms:.1f} ms | new {new_ms:.1f} ms   "
            f"new hits: {total:,}   overlap@{TOP}: {overlap}")
        if err:
            out(f"    NEW ERROR: {err}")
        out("    rank | OLD top results                         | NEW top results")
        for i in range(TOP):
            o = titles.get(old_ids[i], old_ids[i])[:38] if i < len(old_ids) else ''
            w = new.docs[hits[i][0]]['title'][:38] if i < len(hits) else ''
            out(f"    {i + 1:>4} | {o:<40} | {w}")

    out("")
    out("-" * 78)
    out(f"Total time: old {sum_old_ms:.1f} ms | new {sum_new_ms:.1f} ms "
        f"over {len(TEST_CASES)} queries")
    out("=" * 78)

    text = "\n".join(lines)
    print(text)
    os.makedirs(INDEX_DIR, exist_ok=True)
    path = os.path.join(INDEX_DIR, "comparison.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
