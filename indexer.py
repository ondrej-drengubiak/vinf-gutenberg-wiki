"""

indexer.py

Indexer: builds an inverted index over Gutenberg books.

Input:
    - gutenberg.tsv       (metadata: title, author, subjects, loc_class, ...)
    - gutenberg_data/<id>.txt  (full plain text of the book, optional)

Output:
    - index.pkl           (the inverted index + supporting structures)

Design notes
------------
This is a from-scratch inverted index, no Lucene/Whoosh/pandas — just
the standard library (re, csv, pickle, math, collections), as required
for this project.

The index stores, for every term, a mapping doc_id -> term frequency
(raw count of how many times the term occurs in that document):

    inverted_index = {
        "whale":  {"15": 512, "2701": 87, ...},
        "love":   {"1342": 203, ...},
        ...
    }

Document frequency df(term) is NOT stored separately — it is always
derived as len(inverted_index[term]). This guarantees df can never
drift out of sync with the postings list, at the cost of a tiny bit
of extra computation when a weight is needed. N (total number of
indexed documents) is stored once in indexed_docs.

Two ranking/weighting methods
------------------------------
The assignment explicitly forbids "probability idf" (the classic
probabilistic BM25-style idf, log((N - df + 0.5) / (df + 0.5))),
because that is the one every other group used last year. Both
methods below are therefore deliberately NOT probabilistic:

  Method 1 — "tfidf" (classic, linear term frequency)
      weight(t, d) = tf(t, d) * log10(N / df(t))

      This is the textbook vector-space-model weight (Salton).
      It scales linearly with raw term frequency, so a term that
      occurs 10x more often in a document gets 10x the weight.

  Method 2 — "sublinear" (dampened term frequency)
      weight(t, d) = (1 + log10(tf(t, d))) * log10(1 + N / df(t))

      Term frequency is dampened logarithmically (a term appearing
      100 times is not treated as "10x more relevant" than one
      appearing 10 times — diminishing returns), and the idf part
      uses a smoothed log(1 + N/df) instead of the plain log(N/df).
      This tends to be more robust for long documents (like entire
      books!) where a handful of words can otherwise dominate purely
      by raw repetition count.

Both weights are combined with cosine similarity at query time
(implemented in searcher.py); to make that fast, this module also
precomputes the L2 norm of every document's weight vector, for both
methods, once indexing is complete (see finalize()).

Restartability
--------------
Indexing tens of thousands of books is slow, so this must be able
to stop (Ctrl+C) and resume, like every other step of this project.
indexed_docs is the source of truth for "already done"; the index
is flushed to disk every INDEX_SAVE_EVERY documents and again on
KeyboardInterrupt, and build_index() skips any doc_id already in
indexed_docs on the next run.
"""

import os
import re
import csv
import math
import pickle
import collections

from config import Config


_GUTENBERG_START_RE = re.compile(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.IGNORECASE)
_GUTENBERG_END_RE = re.compile(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.IGNORECASE)

_TOKEN_RE = re.compile(r"[A-Za-z']+")


class Indexer:

    def __init__(self, config, index_type='metadata'):
        """
        Args:
            config: Config object
            index_type: 'metadata' or 'text'
        """
        self.config = config
        self.index_type = index_type

        # Choose index file based on type
        if index_type == 'metadata':
            self.index_file = config.INDEX_METADATA_FILE
        elif index_type == 'text':
            self.index_file = config.INDEX_TEXT_FILE
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        self.inverted_index = collections.defaultdict(dict)
        self.doc_lengths = {}
        self.indexed_docs = set()
        self.doc_norms = {}
        self._load_index()


    def _load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'rb') as f:
                state = pickle.load(f)
            self.inverted_index = collections.defaultdict(dict, state['inverted_index'])
            self.doc_lengths = state['doc_lengths']
            self.indexed_docs = state['indexed_docs']
            self.doc_norms = state.get('doc_norms', {})

    def _save_index(self):
        state = {
            'inverted_index': dict(self.inverted_index),
            'doc_lengths': self.doc_lengths,
            'indexed_docs': self.indexed_docs,
            'doc_norms': self.doc_norms,
        }
        tmp_path = self.index_file + '.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.index_file)


    def tokenize(self, text):

        if not text:
            return []
        tokens = _TOKEN_RE.findall(text.lower())
        return [t for t in tokens if len(t) >= self.config.MIN_TOKEN_LEN]

    def _strip_gutenberg_boilerplate(self, text):
        """Cut away the license header/footer, keep only the book body."""
        start_match = _GUTENBERG_START_RE.search(text)
        end_match = _GUTENBERG_END_RE.search(text)
        start = start_match.end() if start_match else 0
        end = end_match.start() if end_match else len(text)
        if start >= end:
            # Markers not found / in a weird order — fall back to
            # the whole text rather than silently indexing nothing.
            return text
        return text[start:end]

    def _document_text(self, row, book_id):
        parts = []

        # Metadata (only for 'metadata' index)
        if self.index_type == 'metadata':
            for field in self.config.INDEX_METADATA_FIELDS:
                value = row.get(field, '')
                if value:
                    parts.append(value.replace('|', ' ').replace('--', ' '))

        # Full text (only for 'text' index)
        if self.index_type == 'text':
            txt_path = os.path.join(self.config.OUTPUT_DIR, f"{book_id}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    if self.config.INDEX_TEXT_MAX_CHARS > 0:
                        body = f.read(self.config.INDEX_TEXT_MAX_CHARS)
                    else:
                        body = f.read()
                parts.append(self._strip_gutenberg_boilerplate(body))

        return ' '.join(parts)

    def build_index(self):
        print("=" * 70)
        print(f"INDEX BUILD START — {self.index_type.upper()}")
        print("=" * 70)

        if not os.path.exists(self.config.TSV_FILE):
            print(f"ERROR: {self.config.TSV_FILE} not found. Run Extract first.")
            return

        with open(self.config.TSV_FILE, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f, delimiter='\t'))

        total = len(rows)
        print(f"Records in TSV:   {total}")
        print(f"Already indexed:  {len(self.indexed_docs)}")
        print(f"Full text:        {'yes' if self.config.INDEX_FULL_TEXT else 'no (metadata only)'}")
        print()

        processed_since_save = 0
        newly_indexed = 0
        failed = 0

        try:
            for i, row in enumerate(rows, 1):
                book_id = row['gutenberg_id']
                if book_id in self.indexed_docs:
                    continue

                try:
                    text = self._document_text(row, book_id)
                    tokens = self.tokenize(text)

                    if not tokens:
                        # Nothing to index (no title, no body) — still
                        # mark as done so we don't retry forever, but
                        # count it so the operator notices if this
                        # number is surprisingly high.
                        failed += 1
                        self.indexed_docs.add(book_id)
                        continue

                    term_counts = collections.Counter(tokens)
                    for term, tf in term_counts.items():
                        self.inverted_index[term][book_id] = tf

                    self.doc_lengths[book_id] = len(tokens)
                    self.indexed_docs.add(book_id)
                    newly_indexed += 1
                    processed_since_save += 1

                    if newly_indexed % 100 == 0 or i == total:
                        print(f"[{i}/{total}] indexed {book_id} "
                              f"({len(tokens)} tokens, {len(term_counts)} unique terms)")

                except Exception as e:
                    failed += 1
                    self.indexed_docs.add(book_id)  # don't retry a permanently broken file forever
                    print(f"[{i}/{total}] {book_id}: ERROR — {e}")

                if processed_since_save >= self.config.INDEX_SAVE_EVERY:
                    self._save_index()
                    processed_since_save = 0
                    print(f"  ... progress saved ({len(self.indexed_docs)} docs indexed so far)")

        except KeyboardInterrupt:
            self._save_index()
            print("\n\nSTOPPED BY USER (Ctrl+C) — index saved, run Index again to continue.")
            return

        # Final save of raw postings before computing norms
        self._save_index()

        print()
        print("=" * 70)
        print(f"INDEX BUILD FINISHED")
        print(f"  Newly indexed: {newly_indexed}")
        print(f"  Empty/failed:  {failed}")
        print(f"  Total indexed: {len(self.indexed_docs)}")
        print(f"  Vocabulary size (unique terms): {len(self.inverted_index)}")
        print("=" * 70)

        self.finalize()

    def df(self, term):
        return len(self.inverted_index.get(term, {}))

    @property
    def N(self):
        return len(self.indexed_docs)

    def weight_tfidf(self, tf, df):
        """
        Method 1: classic linear TF-IDF.
        weight = tf * log10(N / df)
        """
        if tf <= 0 or df <= 0:
            return 0.0
        return tf * math.log10(self.N / df)

    def weight_sublinear(self, tf, df):
        """
        Method 2: sublinear (dampened) TF-IDF.
        weight = (1 + log10(tf)) * log10(1 + N / df)
        """
        if tf <= 0 or df <= 0:
            return 0.0
        return (1 + math.log10(tf)) * math.log10(1 + self.N / df)



    def finalize(self):

        print()
        print("Finalizing index (computing document norms for cosine similarity)...")

        sumsq_tfidf = collections.defaultdict(float)
        sumsq_sublinear = collections.defaultdict(float)

        for term, postings in self.inverted_index.items():
            df = len(postings)
            for doc_id, tf in postings.items():
                w1 = self.weight_tfidf(tf, df)
                w2 = self.weight_sublinear(tf, df)
                sumsq_tfidf[doc_id] += w1 * w1
                sumsq_sublinear[doc_id] += w2 * w2

        self.doc_norms = {
            doc_id: {
                'tfidf': math.sqrt(sumsq_tfidf.get(doc_id, 0.0)),
                'sublinear': math.sqrt(sumsq_sublinear.get(doc_id, 0.0)),
            }
            for doc_id in self.indexed_docs
        }

        self._save_index()
        print("Finalize done, norms saved.")


    def print_stats(self):
        print()
        print("=" * 70)
        print("INDEX STATISTICS")
        print("=" * 70)
        print(f"  Indexed documents:  {self.N}")
        print(f"  Vocabulary size:    {len(self.inverted_index)}")
        if self.doc_lengths:
            avg_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)
            print(f"  Avg. document length (tokens): {avg_len:.1f}")
        if self.doc_norms:
            print(f"  Document norms precomputed: yes ({len(self.doc_norms)} docs)")
        else:
            print(f"  Document norms precomputed: no — run finalize (happens automatically "
                  f"at the end of a completed, non-interrupted build)")
        if self.inverted_index:
            top_terms = sorted(self.inverted_index.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
            print("  Most common terms (by document frequency):")
            for term, postings in top_terms:
                print(f"    {term:20s} df={len(postings)}")
        print("=" * 70)

    def delete_index(self):
        """
        Delete index.pkl completely (derived data only — TSV, HTML
        and TXT crawl data are untouched). Rebuild with Index again.
        """
        if os.path.exists(self.index_file):    
            os.remove(self.index_file)
            self.inverted_index = collections.defaultdict(dict)
            self.doc_lengths = {}
            self.indexed_docs = set()
            self.doc_norms = {}
            print(f"Deleted {self.index_file}.")
            return True
        else:
            print(f"{self.index_file} does not exist, nothing to delete.")
            return False