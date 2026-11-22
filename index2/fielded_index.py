"""
index2/fielded_index.py — own fielded inverted index (Lucene-style), pure Python.

Structure (per TEXT field):
    post[term][doc_id] = [positions]      -> tf = len(positions), phrase queries
    dl[doc_id]         = number of tokens -> length normalisation
KEYWORD field:  value -> [doc_id]
NUMERIC field:  sorted list of (value, doc_id)  -> range queries via bisect

Scoring (NOT probabilistic idf, as required by the assignment):
    score(term, field, doc) =
        (1 + log10(tf)) * log10(1 + N_field / df) * boost(field) / sqrt(len(field, doc))
    - sublinear tf, smoothed idf, Lucene-like length norm and field boost
    - a document's score for a query is the sum over matching clauses.
"""

import bisect
import heapq
import math
import os
import pickle
import re

from index_schema import (FIELDS, DEFAULT_FIELDS, TEXT, KEYWORD, NUMERIC,
                          POSITION_GAP)
from query_parser import parse

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_INF = float("inf")


def tokenize(text):
    """Lower-case alphanumeric tokens; 1-letter tokens are dropped (digits stay)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 or t.isdigit()]


def edit_distance_le(a, b, k):
    """Levenshtein distance if it is <= k, otherwise k + 1 (early exit)."""
    if abs(len(a) - len(b)) > k:
        return k + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            if v < best:
                best = v
        if best > k:
            return k + 1
        prev = cur
    return prev[-1] if prev[-1] <= k else k + 1


class FieldedIndex:

    def __init__(self):
        self.text = {f: {"post": {}, "dl": {}}
                     for f, (t, _) in FIELDS.items() if t == TEXT}
        self.keyword = {f: {} for f, (t, _) in FIELDS.items() if t == KEYWORD}
        self.numeric = {f: [] for f, (t, _) in FIELDS.items() if t == NUMERIC}
        self.docs = []                       # stored fields, index = doc_id

    @property
    def N(self):
        return len(self.docs)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_document(self, fields, stored):
        doc = len(self.docs)
        self.docs.append(stored)
        for name, value in fields.items():
            kind = FIELDS[name][0]
            if kind == TEXT:
                self._add_text(name, doc, value)
            elif kind == KEYWORD:
                if value:
                    self.keyword[name].setdefault(value.lower(), []).append(doc)
            elif value is not None:
                self.numeric[name].append((value, doc))
        return doc

    def _add_text(self, field, doc, values):
        post = self.text[field]["post"]
        pos = length = 0
        for value in values:
            for tok in tokenize(value):
                post.setdefault(tok, {}).setdefault(doc, []).append(pos)
                pos += 1
                length += 1
            pos += POSITION_GAP
        if length:
            self.text[field]["dl"][doc] = length

    def finalize(self):
        for arr in self.numeric.values():
            arr.sort()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {"text": self.text, "keyword": self.keyword,
                 "numeric": self.numeric, "docs": self.docs}
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        idx = cls()
        idx.text, idx.keyword = state["text"], state["keyword"]
        idx.numeric, idx.docs = state["numeric"], state["docs"]
        return idx

    # ------------------------------------------------------------------
    # Scoring primitives (one TEXT field)
    # ------------------------------------------------------------------

    def _idf(self, field, df):
        return math.log10(1 + len(self.text[field]["dl"]) / df)

    def _term_scores(self, field, term):
        post = self.text[field]["post"].get(term)
        if not post:
            return {}
        dl = self.text[field]["dl"]
        w = FIELDS[field][1] * self._idf(field, len(post))
        return {d: (1 + math.log10(len(p))) * w / math.sqrt(dl[d])
                for d, p in post.items()}

    def _phrase_scores(self, field, terms):
        post = self.text[field]["post"]
        lists = [post.get(t) for t in terms]
        if any(l is None for l in lists):
            return {}
        docs = set(lists[0]).intersection(*lists[1:])
        if not docs:
            return {}
        dl = self.text[field]["dl"]
        w = FIELDS[field][1] * sum(self._idf(field, len(l)) for l in lists)
        out = {}
        for d in docs:
            starts = set(lists[0][d])
            for i, l in enumerate(lists[1:], 1):
                starts &= {p - i for p in l[d]}
                if not starts:
                    break
            if starts:
                out[d] = (1 + math.log10(len(starts))) * w / math.sqrt(dl[d])
        return out

    def _fuzzy_scores(self, field, term, k):
        out = {}
        for t in self.text[field]["post"]:
            if t[0] != term[0]:                 # prefix length 1, like Lucene tuning
                continue
            dist = edit_distance_le(term, t, k)
            if dist > k:
                continue
            sim = 1.0 - dist / max(len(term), len(t))
            for d, s in self._term_scores(field, t).items():
                s *= sim
                if s > out.get(d, 0.0):
                    out[d] = s
        return out

    def _prefix_scores(self, field, prefix):
        out = {}
        for t in self.text[field]["post"]:
            if t.startswith(prefix):
                for d, s in self._term_scores(field, t).items():
                    if s > out.get(d, 0.0):
                        out[d] = s
        return out

    def _range(self, field, lo, hi):
        if field not in self.numeric:
            raise ValueError(f"Range queries need a numeric field "
                             f"({', '.join(self.numeric)}), got '{field}'")
        arr = self.numeric[field]
        lo = -_INF if lo is None else lo
        hi = _INF if hi is None else hi
        i = bisect.bisect_left(arr, (lo, -1))
        j = bisect.bisect_right(arr, (hi, _INF))
        return {d: 1.0 for _, d in arr[i:j]}

    # ------------------------------------------------------------------
    # Query evaluation
    # ------------------------------------------------------------------

    def evaluate(self, node):
        """AST node -> {doc_id: score}."""
        if node[0] == "bool":
            res = self._eval_bool(node[1], node[2], node[3])
        else:
            res = self._eval_leaf(node)
        boost = node[-1]
        if boost != 1.0:
            res = {d: s * boost for d, s in res.items()}
        return res

    def _eval_bool(self, must, should, must_not):
        if must:
            res = self.evaluate(must[0])
            for n in must[1:]:
                r = self.evaluate(n)
                res = {d: s + r[d] for d, s in res.items() if d in r}
            for n in should:                       # optional clauses add score
                r = self.evaluate(n)
                for d in res:
                    if d in r:
                        res[d] += r[d]
        elif should:
            res = {}
            for n in should:
                for d, s in self.evaluate(n).items():
                    res[d] = res.get(d, 0.0) + s
        else:
            return {}                               # only NOT clauses -> nothing
        for n in must_not:
            for d in self.evaluate(n):
                res.pop(d, None)
        return res

    def _eval_leaf(self, node):
        kind, field = node[0], node[1]
        if kind == "range":
            return self._range(field, node[2], node[3])
        if field is not None and field not in FIELDS:
            raise ValueError(f"Unknown field '{field}'. Fields: {', '.join(FIELDS)}")
        out = {}
        for f in ([field] if field else DEFAULT_FIELDS):
            ftype = FIELDS[f][0]
            if ftype == TEXT:
                part = self._text_leaf(f, node)
            elif ftype == KEYWORD:
                part = self._keyword_leaf(f, node)
            else:
                part = self._numeric_leaf(f, node)
            for d, s in part.items():
                out[d] = out.get(d, 0.0) + s
        return out

    def _text_leaf(self, f, node):
        kind, text = node[0], node[2]
        if kind == "prefix":
            return self._prefix_scores(f, re.sub(r"[^a-z0-9]", "", text.lower()))
        toks = tokenize(text)
        if not toks:
            return {}
        if len(toks) > 1:
            return self._phrase_scores(f, toks)
        if kind == "fuzzy":
            return self._fuzzy_scores(f, toks[0], node[3])
        return self._term_scores(f, toks[0])

    def _keyword_leaf(self, f, node):
        kind, text = node[0], node[2].lower()
        if kind == "prefix":
            return {d: 1.0 for v, docs in self.keyword[f].items()
                    if v.startswith(text) for d in docs}
        return {d: 1.0 for d in self.keyword[f].get(text, [])}

    def _numeric_leaf(self, f, node):
        try:
            v = float(node[2])
        except ValueError:
            raise ValueError(f"'{f}' is numeric: use {f}:1851 or {f}:[1850 TO 1900]")
        return self._range(f, v, v)

    def search(self, query, top_k=10, default_op="AND"):
        """Returns ([(doc_id, score), ...], total_hits)."""
        scores = self.evaluate(parse(query, default_op))
        top = heapq.nsmallest(top_k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return top, len(scores)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self):
        s = {"documents": self.N, "text": {}, "keyword": {}, "numeric": {}}
        for f, data in self.text.items():
            s["text"][f] = {
                "docs_with_field": len(data["dl"]),
                "vocabulary": len(data["post"]),
                "postings": sum(len(p) for p in data["post"].values()),
            }
        for f, data in self.keyword.items():
            s["keyword"][f] = {"distinct_values": len(data),
                               "docs": sum(len(v) for v in data.values())}
        for f, arr in self.numeric.items():
            s["numeric"][f] = {"docs": len(arr),
                               "min": arr[0][0] if arr else None,
                               "max": arr[-1][0] if arr else None}
        return s
