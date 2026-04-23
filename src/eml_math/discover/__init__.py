"""
EML Formula Discovery — find closed-form expressions from numerical data.

Uses the Rust eml_core extension when available (fast, parallel BFS).
Falls back to a pure-Python reference implementation otherwise.

Usage
-----
>>> from eml_math.discover import Searcher
>>> import math
>>> x = [i * 0.1 for i in range(1, 31)]
>>> y = [math.exp(xi) - math.log(xi) for xi in x]   # eml(x, x)
>>> result = Searcher(max_complexity=6).find(x, y)
>>> print(result.formula)   # "eml(x, x)" or equivalent
>>> print(result.error)     # < 1e-10
"""
from eml_math.discover.search import Searcher
from eml_math.discover.result import SearchResult

__all__ = ["Searcher", "SearchResult"]
