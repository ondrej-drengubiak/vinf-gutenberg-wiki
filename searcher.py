"""

searcher.py


Two weighting methods are supported, matching Indexer:
    - 'tfidf'      : tf * log10(N / df)                 (classic VSM)
    - 'sublinear'  : (1 + log10(tf)) * log10(1 + N/df)  (dampened)

Cosine similarity:
    sim(q, d) = sum_t w_q(t) * w_d(t)  /  (||q|| * ||d||)

where w_q(t) uses the same weighting method as w_d(t).

Query language (boolean layer on top of cosine similarity)
----------------------------------------------------------
The base search is cosine similarity over the metadata index. On top
of that, this module adds a small boolean query language:

    pride AND prejudice
    pride OR prejudice
    pride AND NOT zombies
    love AND marriage AND NOT war
    (pride OR prejudice) AND marriage

Rules:
    * AND / OR / NOT must be UPPERCASE (so the word "and" inside
      a normal query is NOT treated as an operator — important for
      titles like "Pride and Prejudice" if a user types them raw).
    * AND binds tighter than OR (standard precedence, like most
      search engines and like Python's "and"/"or").
    * NOT applies to the single following term / parenthesised group.
    * Parentheses ( ) override precedence.
    * If the query contains NO operators at all, the behaviour is
      exactly the old one: pure cosine similarity, OR-like.

How boolean filtering interacts with cosine
--------------------------------------------
1. Compute cosine scores for every term in the query (same as before).
2. Evaluate the boolean expression per document, using the document's
   postings as truth values ("does this doc contain this term?").
3. Keep only documents for which the boolean expression is True.
4. Sort kept documents by their cosine score.

This means: boolean acts as a HARD FILTER, cosine acts as the RANKING.
That is exactly how Lucene / Whoosh do it, and how users expect it.
"""

import math

from config import Config
from indexer import Indexer


# ---------------------------------------------------------------------------
# Tokenizer for the query language
# ---------------------------------------------------------------------------

# Tokens we care about: ( ) AND OR NOT  and  "words"  (with apostrophes).
_QUERY_TOKEN_RE = None  # built lazily in _tokenize_query() to avoid importing re at module top if not needed
import re as _re

_QUERY_TOKEN_RE = _re.compile(
    r'\(|\)|\bAND\b|\bOR\b|\bNOT\b|[A-Za-z\']+',
    _re.IGNORECASE,
)


def _tokenize_query(query):
    """
    Split a raw query string into a flat list of tokens:
      '(' , ')' , 'AND' , 'OR' , 'NOT'  (uppercased)
      and plain words (lowercased, >= 2 chars).

    Words of length < 2 are dropped (same rule as the index tokenizer),
    so 'a' or a stray single letter never reaches the parser.
    """
    out = []
    for raw in _QUERY_TOKEN_RE.findall(query):
        if raw in ('(', ')'):
            out.append(raw)
            continue
        upper = raw.upper()
        if upper in ('AND', 'OR', 'NOT'):
            # Only treat as operator if the ORIGINAL text was uppercase.
            # This preserves titles like "Pride and Prejudice" typed
            # verbatim, where "and" must stay a normal word.
            if raw == upper:
                out.append(upper)
                continue
        word = raw.lower()
        if len(word) >= 2:
            out.append(word)
    return out


# ---------------------------------------------------------------------------
# Recursive-descent parser for the boolean query language
# ---------------------------------------------------------------------------
#
# Grammar (standard precedence):
#
#   expr    := or_expr
#   or_expr := and_expr ( 'OR' and_expr )*
#   and_expr:= unary ( 'AND' unary )*
#   unary   := 'NOT' unary | primary
#   primary := '(' expr ')' | WORD
#
# The resulting AST is a nested tuple:
#   ('term', 'pride')
#   ('and',  left, right)
#   ('or',   left, right)
#   ('not',  child)
#   ('true',)          # empty query — matches everything (but no ranking)
#
# A parse error raises ValueError with a human-readable message.

class _QueryParser:

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self):
        if not self.tokens:
            return ('true',)
        node = self.parse_or()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token: {self.peek()!r}")
        return node

    def parse_or(self):
        left = self.parse_and()
        while self.peek() == 'OR':
            self.consume()
            right = self.parse_and()
            left = ('or', left, right)
        return left

    def parse_and(self):
        left = self.parse_unary()
        while self.peek() == 'AND':
            self.consume()
            right = self.parse_unary()
            left = ('and', left, right)
        return left

    def parse_unary(self):
        if self.peek() == 'NOT':
            self.consume()
            return ('not', self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of query (missing term after operator?)")
        if tok == '(':
            self.consume()
            node = self.parse_or()
            if self.peek() != ')':
                raise ValueError("Missing closing ')' in query")
            self.consume()
            return node
        if tok in (')', 'AND', 'OR', 'NOT'):
            raise ValueError(f"Unexpected token {tok!r} — expected a word or '('")
        # plain word
        self.consume()
        return ('term', tok)


def parse_boolean_query(query):
    """
    Public helper: parse a query string into a boolean AST.
    Returns ('true',) for an empty query OR for a query with no
    uppercase boolean operators (so the old pure-cosine behaviour
    is preserved exactly).
    Raises ValueError on syntax errors.
    """
    tokens = _tokenize_query(query)

    # If there is NO uppercase boolean operator among the tokens,
    # there is nothing to parse — treat it as a plain bag-of-words query.
    if not any(tok in ('AND', 'OR', 'NOT', '(', ')') for tok in tokens):
        return ('true',)

    return _QueryParser(tokens).parse()


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class Searcher:

    def __init__(self, config, index_type='metadata'):

        self.config = config
        self.index_type = index_type
        self.indexer = Indexer(config, index_type=index_type)

        if not self.indexer.indexed_docs:
            raise RuntimeError(
                f"No {index_type} index found. Build it first "
                f"(menu option 4)."
            )
        if not self.indexer.doc_norms:
            raise RuntimeError(
                f"{index_type} index has no document norms (finalize was "
                f"not run). Rebuild the index without interrupting it."
            )

    # ------------------------------------------------------------------
    # Query vector (unchanged behaviour)
    # ------------------------------------------------------------------

    def _tokenize(self, query):
        return self.indexer.tokenize(query)

    def _query_vector(self, query, method):

        tokens = self._tokenize(query)
        if not tokens:
            return {}, 0.0

        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        vector = {}
        sumsq = 0.0

        for term, count in tf.items():
            df = self.indexer.df(term)
            if df == 0:
                # Term not in index — skip. It contributes nothing to
                # cosine similarity (no doc has it), and keeping it would
                # only inflate the query norm for no benefit.
                continue
            if method == 'tfidf':
                w = self.indexer.weight_tfidf(count, df)
            elif method == 'sublinear':
                w = self.indexer.weight_sublinear(count, df)
            else:
                raise ValueError(f"Unknown method: {method}")
            vector[term] = w
            sumsq += w * w

        norm = math.sqrt(sumsq)
        return vector, norm

    # ------------------------------------------------------------------
    # Boolean expression evaluation per document
    # ------------------------------------------------------------------

    def _collect_terms(self, ast):
        """Yield all terms that appear anywhere in the AST."""
        kind = ast[0]
        if kind == 'term':
            yield ast[1]
        elif kind in ('and', 'or'):
            yield from self._collect_terms(ast[1])
            yield from self._collect_terms(ast[2])
        elif kind == 'not':
            yield from self._collect_terms(ast[1])
        # 'true' yields nothing

    def _doc_has(self, doc_id, term):
        return doc_id in self.indexer.inverted_index.get(term, {})

    def _eval_ast(self, ast, doc_id):
        """Evaluate the boolean AST for a single document."""
        kind = ast[0]
        if kind == 'true':
            return True
        if kind == 'term':
            return self._doc_has(doc_id, ast[1])
        if kind == 'and':
            return self._eval_ast(ast[1], doc_id) and self._eval_ast(ast[2], doc_id)
        if kind == 'or':
            return self._eval_ast(ast[1], doc_id) or self._eval_ast(ast[2], doc_id)
        if kind == 'not':
            return not self._eval_ast(ast[1], doc_id)
        raise ValueError(f"Unknown AST node: {kind!r}")

    def _has_any_operator(self, ast):
        """True if the AST is more than a single term / 'true'."""
        if ast[0] == 'true':
            return False
        if ast[0] == 'term':
            return False
        return True

    def _candidate_docs(self, ast):
        """
        Return the set of doc_ids that could possibly satisfy the AST.
        This is just an optimisation — we only need to evaluate the
        boolean for docs that have at least one term from the query,
        otherwise they score 0 on cosine and would never be returned
        anyway (for non-degenerate queries). For 'true' (empty query),
        we return an empty set, meaning "no ranking available".
        """
        if ast[0] == 'true':
            return set()
        terms = set(self._collect_terms(ast))
        if not terms:
            return set()
        # Union of all postings for all terms mentioned
        docs = set()
        for term in terms:
            docs.update(self.indexer.inverted_index.get(term, {}).keys())
        return docs

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(self, query, method='tfidf', top_k=10):
        """
        Run a search.

        If the query contains boolean operators (AND / OR / NOT / parentheses),
        they are evaluated as a HARD FILTER over all documents that share at
        least one term with the query. The surviving documents are then ranked
        by cosine similarity (same as before).

        If the query has no operators, behaviour is identical to the old
        version: pure cosine over all docs that match any term (OR-like).

        Raises ValueError for syntactically invalid boolean queries.
        """

        # 1) Parse the boolean structure (raises ValueError on syntax error)
        ast = parse_boolean_query(query)

        # 2) Build the query vector as before, using the raw query text.
        #    Note: this includes operator words themselves, but they were
        #    uppercase in the input and _tokenize() lowercases them, so
        #    they become terms like 'and'/'or'/'not'. If the user typed
        #    them in uppercase, they'd be in the vector, which is not what
        #    we want. Fix: strip operator words before building the vector.
        cleaned_query = self._strip_operators(query)
        q_vec, q_norm = self._query_vector(cleaned_query, method)
        if not q_vec or q_norm == 0.0:
            return []

        # 3) Compute raw cosine numerators (dot products) as before.
        scores = {}
        for term, w_q in q_vec.items():
            postings = self.indexer.inverted_index.get(term, {})
            df = len(postings)
            for doc_id, tf in postings.items():
                if method == 'tfidf':
                    w_d = self.indexer.weight_tfidf(tf, df)
                else:
                    w_d = self.indexer.weight_sublinear(tf, df)
                scores[doc_id] = scores.get(doc_id, 0.0) + w_q * w_d

        # 4) Apply boolean filter if the query has any operator.
        #    Documents that fail the boolean expression are removed
        #    from the candidate set, even if they have a high cosine.
        if self._has_any_operator(ast):
            keep = {doc_id for doc_id in scores if self._eval_ast(ast, doc_id)}
            scores = {doc_id: s for doc_id, s in scores.items() if doc_id in keep}

        # 5) Normalise by ||d|| and rank by cosine.
        results = []
        for doc_id, dot in scores.items():
            d_norm = self.indexer.doc_norms.get(doc_id, {}).get(method, 0.0)
            if d_norm == 0.0:
                continue
            sim = dot / (q_norm * d_norm)
            results.append({'doc_id': doc_id, 'score': sim})

        results.sort(key=lambda r: r['score'], reverse=True)
        return results[:top_k]

    def _strip_operators(self, query):
        """
        Remove boolean operator words and parentheses from the query
        before passing it to the standard tokenizer, so they don't end
        up as terms in the query vector.

        'AND', 'OR', 'NOT' are only removed if they were UPPERCASE in the
        original query (same rule as the parser), so a query like
        'pride and prejudice' keeps 'and' as a normal (meaningless) term,
        exactly like before.
        """
        # Replace uppercase operators with spaces, keep the rest.
        # We also drop parentheses; they are syntax, not content.
        stripped = _re.sub(r'\b(AND|OR|NOT)\b', ' ', query)
        stripped = stripped.replace('(', ' ').replace(')', ' ')
        return stripped

    # ------------------------------------------------------------------
    # Pretty printing (unchanged, but shows the parsed query for debug)
    # ------------------------------------------------------------------

    def _metadata_for(self, doc_id, row_cache):
        return row_cache.get(doc_id, {})

    def print_results(self, query, results, method, top_k=10):
        """Print results with title + author from the TSV."""
        import csv
        import os

        row_cache = {}
        if os.path.exists(self.config.TSV_FILE):
            with open(self.config.TSV_FILE, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f, delimiter='\t'):
                    row_cache[row['gutenberg_id']] = row

        print()
        print("=" * 78)
        print(f"QUERY: {query!r}   [method={method}, top_k={top_k}]")
        print("=" * 78)
        if not results:
            print("  (no results)")
            return
        for rank, r in enumerate(results, 1):
            meta = self._metadata_for(r['doc_id'], row_cache)
            title = meta.get('title', '')[:70]
            author = meta.get('author', '')
            print(f"  {rank:2d}. score={r['score']:.4f}  id={r['doc_id']}")
            print(f"      {title}")
            if author:
                print(f"      by {author}")
        print("=" * 78)