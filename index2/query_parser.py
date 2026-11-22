"""
index2/query_parser.py — tiny Lucene-like query parser (pure Python, `re` only).

Supported syntax
    whale                   term (searched in all default fields)
    title:whale             term in one field
    "great expectations"    phrase (positions must be consecutive)
    title:whal*             prefix
    title:frankenstien~1    fuzzy (max edit distance 0-2, default 2)
    pub_year:[1850 TO 1900] inclusive range, '*' = open end
    a AND b, a OR b, NOT a  boolean operators, also +a / -a, parentheses
    title:whale^3           boost
    author:(dickens OR poe) field applied to a group

Implicit operator between clauses is AND (default_op='AND'; 'OR' is possible).
Precedence: NOT/+/- > AND (explicit or implicit) > OR.

AST (tuples, boost is always the LAST element):
    ('term',   field|None, text, boost)
    ('phrase', field|None, text, boost)
    ('prefix', field|None, text, boost)
    ('fuzzy',  field|None, text, max_edits, boost)
    ('range',  field, lo|None, hi|None, boost)
    ('bool',   must[], should[], must_not[], boost)
"""

import re


class QueryParseError(ValueError):
    pass


_TOKEN_RE = re.compile(r'''
    \s*(?:
        (?P<lpar>\()                                   |
        (?P<rpar>\))                                   |
        (?P<phrase>"[^"]*")                            |
        (?P<range>\[\s*\S+\s+TO\s+\S+?\s*\])           |
        (?P<field>[A-Za-z_]+):                         |
        (?P<boost>\^\d+(?:\.\d+)?)                     |
        (?P<fuzzy>~\d?)                                |
        (?P<op>[+-](?=\S))                             |
        (?P<word>[^\s()"\[\]^~:]+)
    )''', re.X)
_RANGE_RE = re.compile(r'\[\s*(\S+)\s+TO\s+(\S+?)\s*\]')


def _lex(query):
    out, pos = [], 0
    while pos < len(query):
        m = _TOKEN_RE.match(query, pos)
        if not m:
            pos += 1                      # skip stray character / trailing blanks
            continue
        out.append((m.lastgroup, m.group(m.lastgroup)))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens, default_op):
        self.toks = tokens
        self.i = 0
        self.default_op = default_op

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    # or_expr := and_expr (OR and_expr)*
    def parse_or(self, field=None):
        parts = [self.parse_and(field)]
        while self.peek() == ('word', 'OR'):
            self.i += 1
            parts.append(self.parse_and(field))
        if len(parts) == 1:
            return parts[0]
        return ('bool', [], parts, [], 1.0)

    # and_expr := clause*   (implicit / explicit AND, +, -, NOT)
    def parse_and(self, field=None):
        clauses = []
        while True:
            kind, val = self.peek()
            if kind is None or kind == 'rpar' or (kind, val) == ('word', 'OR'):
                break
            if (kind, val) == ('word', 'AND'):
                self.i += 1
                continue
            occur = ''
            if (kind, val) == ('word', 'NOT'):
                occur = '-'
                self.i += 1
            elif kind == 'op':
                occur = val
                self.i += 1
            clauses.append((occur, self.parse_primary(field)))

        if not clauses:
            return ('bool', [], [], [], 1.0)
        if len(clauses) == 1 and clauses[0][0] == '':
            return clauses[0][1]
        must, should, must_not = [], [], []
        for occur, node in clauses:
            if occur == '-':
                must_not.append(node)
            elif occur == '+' or (occur == '' and self.default_op == 'AND'):
                must.append(node)
            else:
                should.append(node)
        return ('bool', must, should, must_not, 1.0)

    def _boost(self, node):
        kind, val = self.peek()
        if kind == 'boost':
            self.i += 1
            return node[:-1] + (node[-1] * float(val[1:]),)
        return node

    def parse_primary(self, field=None):
        kind, val = self.next()
        if kind == 'lpar':
            node = self.parse_or(field)
            if self.next()[0] != 'rpar':
                raise QueryParseError("missing closing ')'")
            return self._boost(node)
        if kind == 'field':
            return self.parse_primary(val.lower())
        if kind == 'phrase':
            return self._boost(('phrase', field, val[1:-1], 1.0))
        if kind == 'range':
            if field is None:
                raise QueryParseError("a range needs a field, e.g. pub_year:[1850 TO 1900]")
            lo, hi = _RANGE_RE.match(val).groups()
            conv = lambda s: None if s == '*' else _number(s)
            return self._boost(('range', field, conv(lo), conv(hi), 1.0))
        if kind == 'word':
            if val.endswith('*') and len(val) > 1:
                return self._boost(('prefix', field, val[:-1], 1.0))
            k2, v2 = self.peek()
            if k2 == 'fuzzy':
                self.i += 1
                edits = min(int(v2[1:]) if len(v2) > 1 else 2, 2)
                return self._boost(('fuzzy', field, val, edits, 1.0))
            return self._boost(('term', field, val, 1.0))
        raise QueryParseError(f"unexpected token {val!r}")


def _number(s):
    try:
        return float(s)
    except ValueError:
        raise QueryParseError(f"range bound {s!r} is not a number")


def parse(query, default_op='AND'):
    tokens = _lex(query)
    if not tokens:
        raise QueryParseError("empty query")
    parser = _Parser(tokens, default_op.upper())
    node = parser.parse_or()
    if parser.i < len(tokens):
        raise QueryParseError("unbalanced parentheses")
    return node
