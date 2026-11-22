"""
index2/index_builder.py — build the fielded index from spark_output/joined_jsonl.

Reads one JSON object per line (Spark export), derives the index fields
(including a few attributes pulled out of the Wikipedia infobox with REGEX)
and writes index2_output/fielded_index.pkl.

Usage:  python index2/index_builder.py     (or menu item 16 in main.py)
"""

import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fielded_index import FieldedIndex
from index_schema import JSONL_DIR, INDEX_FILE

# ---------------------------------------------------------------------------
# Infobox attributes (REGEX only). The infobox is raw wikitext:
#   {{Infobox book
#   | publisher = [[Chapman & Hall]]
#   | pub_date  = 1851
#   }}
# ---------------------------------------------------------------------------

_IB_PARAM_RE = re.compile(
    r'^\s*\|\s*(publisher|genre|pub_date|published|country)\s*=(.*?)(?=^\s*\||\Z)',
    re.M | re.S | re.I)
_IB_REF_RE = re.compile(r'<ref[^>]*?/>|<ref[^>]*>.*?</ref>|<!--.*?-->', re.S)
_IB_LINK_RE = re.compile(r'\[\[(?:[^\]|]+\|)?([^\]]+)\]\]')
_IB_TEMPLATE_HEAD_RE = re.compile(r'\{\{\s*[^{}|]*\|?')   # {{hlist|  {{plainlist|
_IB_JUNK_RE = re.compile(r'[{}<>\[\]|]|&nbsp;')
_WS_RE = re.compile(r'\s+')
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-2][0-9])\b')
_DIGITS_RE = re.compile(r'[\d,]+')


def _clean_ib(value):
    value = _IB_REF_RE.sub(' ', value)
    value = _IB_LINK_RE.sub(r'\1', value)
    value = _IB_TEMPLATE_HEAD_RE.sub(' ', value)
    value = _IB_JUNK_RE.sub(' ', value)
    return _WS_RE.sub(' ', value).strip()


def parse_infobox(infobox):
    """-> dict with keys publisher / genre / pub_date / country (cleaned text)."""
    out = {}
    for m in _IB_PARAM_RE.finditer(infobox or ''):
        key = m.group(1).lower()
        key = 'pub_date' if key == 'published' else key
        out.setdefault(key, _clean_ib(m.group(2)))
    return out


def _first_year(text):
    m = _YEAR_RE.search(text or '')
    return int(m.group(1)) if m else None


def _first_int(text):
    m = _DIGITS_RE.search(text or '')
    if not m:
        return None
    digits = m.group(0).replace(',', '')
    return int(digits) if digits else None


def derive(rec):
    """JSON record -> (index fields, stored fields)."""
    ib = parse_infobox(rec.get('wiki_infobox'))
    subjects = [s for s in (rec.get('gutenberg_subjects') or '').split('|') if s]
    cats = [c for c in (rec.get('wiki_categories') or '').split('|') if c]
    pub_year = _first_year(ib.get('pub_date'))

    fields = {
        'title':        [rec.get('gutenberg_title') or ''],
        'author':       [rec.get('gutenberg_author') or ''],
        'subjects':     subjects,
        'loc_class':    [rec.get('gutenberg_loc_class') or ''],
        'abstract':     [rec.get('wiki_abstract') or ''],
        'categories':   cats,
        'publisher':    [ib.get('publisher', '')],
        'genre':        [ib.get('genre', '')],
        'language':     rec.get('gutenberg_language') or '',
        'has_wiki':     'yes' if rec.get('wiki_title') else 'no',
        'pub_year':     pub_year,
        'release_year': _first_year(rec.get('gutenberg_release_date')),
        'downloads':    _first_int(rec.get('downloads')),
    }
    stored = {
        'gutenberg_id': rec.get('gutenberg_id'),
        'title':        rec.get('gutenberg_title') or '',
        'author':       rec.get('gutenberg_author') or '',
        'language':     rec.get('gutenberg_language') or '',
        'pub_year':     pub_year,
        'downloads':    fields['downloads'],
        'wiki_title':   rec.get('wiki_title') or '',
    }
    return fields, stored


def build(jsonl_dir=JSONL_DIR, index_file=INDEX_FILE):
    files = sorted(glob.glob(os.path.join(jsonl_dir, '*.json')))
    if not files:
        print(f"ERROR: no JSON files in {jsonl_dir}. Run 'Spark Export' (15) first.")
        return None

    idx = FieldedIndex()
    bad = 0
    t0 = time.time()
    for path in files:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                idx.add_document(*derive(rec))
                if idx.N % 10000 == 0:
                    print(f"  ... {idx.N:,} documents")
    idx.finalize()
    idx.save(index_file)
    elapsed = time.time() - t0

    print()
    print(f"Indexed documents: {idx.N:,}   (bad JSON lines: {bad})")
    print(f"Build time:        {elapsed:.1f} s")
    print(f"Index file:        {index_file}  ({os.path.getsize(index_file) / 1024 / 1024:.1f} MB)")
    return idx


def main():
    print("=" * 70)
    print("BUILD FIELDED INDEX")
    print("=" * 70)
    build()


if __name__ == '__main__':
    main()
