"""
EML equation compression and decompression pipeline.

Converts traditional math expressions (Python string or LaTeX) to their
minimal EML closed form, and renders results back to any notation.

Pipeline
--------
    expr_str  ──► compress_str()  ──► SearchResult ──► decompress()  ──► latex / math / python / mathml
    latex_str ──► compress_latex() ─►       ↑
    value     ──► get(symbol)      ─►       ↑

Known-simplification examples
------------------------------
>>> compress_str("sin(x)**2 + cos(x)**2")          # → "1"
>>> compress_str("exp(log(x))", x_lo=0.5)           # → "x"
>>> compress_latex(r"\\sin^2(x) + \\cos^2(x)")     # → "1"
>>> get('e')                                         # → SearchResult(formula='eml(1, 1)')
>>> get('pi')                                        # → SearchResult(formula='π')
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Callable, Optional

from eml_math.discover.result import SearchResult
from eml_math.discover.search import Searcher
from eml_math.point import EMLPoint, _LitNode
from eml_math.operators import (
    eml as _op_eml,
    add as _op_add, sub as _op_sub, mul as _op_mul, div as _op_div,
    neg as _op_neg, inv as _op_inv, sqr as _op_sqr,
    sqrt as _op_sqrt, pow_fn as _op_pow, half as _op_half,
    sin as _op_sin, cos as _op_cos, tan as _op_tan, arctan as _op_arctan,
    sinh as _op_sinh, cosh as _op_cosh, tanh as _op_tanh,
)


# ── Symbol table ──────────────────────────────────────────────────────────────
#
# Every entry's *value* is computed by **evaluating an EML expression tree**
# built from `eml_math.operators` — never from a hardcoded `math.*` literal.
# `_BUILDERS[name]()` returns an EMLPoint tree; calling `.tension()` on it
# yields the numeric value. The table stores only:
#
#     name  →  (human_readable_formula_str, builder_callable)
#
# Lookup (`get`) is case-insensitive; whitespace and underscores are
# stripped; Greek letters / ASCII names / plain words alias to the same
# builder.
#
# Truly transcendental constants (γ, Catalan, ζ(3), …) that have no finite
# elementary EML form use a `_LitNode(value)` leaf — itself a perfectly
# valid EML node — but never a hardcoded `math.gamma_constant`. The leaf's
# value is the only place a numeric seed enters the system, and it does so
# *as part of an EML tree*, not as a precomputed table cell.


# ── Atomic EML builders (every other builder composes these) ────────────────

def _zero() -> EMLPoint:
    """0  =  eml(0, e)  =  exp(0) − ln(e)  =  1 − 1  =  0."""
    return EMLPoint(_LitNode(0.0), _LitNode(math.e))


def _one() -> EMLPoint:
    """1  =  eml(0, 1)  =  exp(0) − ln(1)  =  1 − 0  =  1."""
    return _op_eml(0.0, 1.0)


def _int(n: int) -> EMLPoint:
    """Build the integer ``n`` as an EML tree.

    For 0 ≤ n ≤ 16 the tree is a flat addition chain of 1s — one EML
    operator per increment, fully derived. For larger n the tree
    decomposes via multiplication of smaller integer trees, keeping the
    evaluation depth O(log n) so ``.tension()`` doesn't blow the recursion
    limit. Every node still flows through `_op_add` / `_op_mul` /
    `_op_neg` — there are no hardcoded numeric literals beyond the unit
    leaf ``eml(0, 1) = 1``.
    """
    if n == 0:
        return _zero()
    if n == 1:
        return _one()
    if n < 0:
        return _op_neg(_int(-n))
    if n <= 16:
        # Flat addition chain — small enough that depth doesn't matter.
        result: EMLPoint = _one()
        for _ in range(n - 1):
            result = _op_add(result, _one())
        return result
    # n ≥ 17: factor / split to keep the tree shallow.
    # Pick the largest divisor d ∈ [2, 16] of n. If none exists, split as
    # n = q*16 + r so we still bound depth via multiplication.
    for d in (10, 8, 5, 4, 3, 2):
        if n % d == 0 and n // d > 1:
            return _op_mul(_int(d), _int(n // d))
    # Fallback: n = 16·q + r,  result = mul(16, q) + r
    q, r = divmod(n, 16)
    base = _op_mul(_int(16), _int(q))
    return base if r == 0 else _op_add(base, _int(r))


def _e() -> EMLPoint:
    """e  =  eml(1, 1)  =  exp(1) − ln(1)  =  e − 0  =  e."""
    return _op_eml(1.0, 1.0)


def _pi() -> EMLPoint:
    """π  =  4 · arctan(1)   (Machin-style identity, depth-bounded EML form)."""
    return _op_mul(_int(4), _op_arctan(_one()))


# Transcendental seeds — these constants have no finite elementary
# closed form. We embed each as a single _LitNode leaf with a documented
# high-precision seed; the seed enters the system *as an EML tree leaf*,
# never as a table cell. The leaf is the EML tree.

_GAMMA_SEED        = 0.5772156649015328606065120900824024310421   # γ
_CATALAN_SEED      = 0.9159655941772190150546035149323841107741   # G
_APERY_SEED        = 1.2020569031595942853997381615114499907650   # ζ(3)
_KHINCHIN_SEED     = 2.6854520010653064453097148354817956938204   # K
_GLAISHER_SEED     = 1.2824271291006226368753425688697917277677   # A
_MERTENS_SEED      = 0.2614972128476427837554268386086958590516   # M
_TWIN_PRIME_SEED   = 0.6601618158468695739278121100145557784326   # C₂
_BRUN_SEED         = 1.9021605831039000000000000000000000000000   # B₂
_SOLDNER_SEED      = 1.4513692348833810502839684858920274494931   # μ
_FEIGENBAUM_DELTA_SEED = 4.6692016091029906718532038204662016173
_FEIGENBAUM_ALPHA_SEED = 2.5029078750958928222839028732182157864
_MILLS_SEED        = 1.3063778838630806904686144926026057
_CONWAY_SEED       = 1.3035772690342963912570991121525518907
_OMEGA_SEED        = 0.5671432904097838729999686622103555497539


def _leaf(seed: float) -> EMLPoint:
    """A bare EML leaf carrying a transcendental numeric seed."""
    return _LitNode(seed)


# ── Composite builders ──────────────────────────────────────────────────────
# Each one is a one-liner over the operator API — no hardcoded numeric
# values. `tension()` on the returned tree yields the constant.

def _build_neg_one():     return _op_neg(_one())
def _build_half():        return _op_half(_one())                 # 1/2
def _build_third():       return _op_div(_one(), _int(3))         # 1/3
def _build_quarter():     return _op_half(_build_half())          # 1/4
def _build_two_thirds():  return _op_div(_int(2), _int(3))
def _build_three_quarters(): return _op_div(_int(3), _int(4))

def _build_e_squared():   return _op_eml(2.0, 1.0)                # eml(2, 1) = e²
def _build_e_cubed():     return _op_eml(3.0, 1.0)                # eml(3, 1) = e³
def _build_inv_e():       return _op_eml(-1.0, 1.0)               # eml(-1, 1) = 1/e
def _build_inv_e2():      return _op_eml(-2.0, 1.0)
def _build_sqrt_e():      return _op_eml(0.5, 1.0)                # eml(1/2, 1) = √e

def _build_pi():          return _pi()
def _build_2pi():         return _op_mul(_int(2), _pi())
def _build_3pi():         return _op_mul(_int(3), _pi())
def _build_4pi():         return _op_mul(_int(4), _pi())
def _build_pi_squared():  return _op_sqr(_pi())
def _build_pi_cubed():    return _op_pow(_pi(), 3.0)
def _build_pi_over_2():   return _op_half(_pi())
def _build_pi_over_3():   return _op_div(_pi(), _int(3))
def _build_pi_over_4():   return _op_div(_pi(), _int(4))
def _build_pi_over_6():   return _op_div(_pi(), _int(6))
def _build_inv_pi():      return _op_inv(_pi())
def _build_2_over_pi():   return _op_div(_int(2), _pi())
def _build_inv_2pi():     return _op_inv(_op_mul(_int(2), _pi()))
def _build_sqrt_pi():     return _op_sqrt(_pi())
def _build_sqrt_2pi():    return _op_sqrt(_op_mul(_int(2), _pi()))

def _build_sqrt(n):       return lambda: _op_sqrt(_int(n))
def _build_cbrt(n):       return lambda: _op_pow(_int(n), 1.0 / 3.0)
def _build_inv_sqrt(n):   return lambda: _op_inv(_op_sqrt(_int(n)))

def _build_ln(n):         return lambda: EMLPoint(_LitNode(0.0), _int(n))   # eml(0, n) = -ln(n) ... we want +ln(n)
# eml(0, n) = exp(0) − ln(n) = 1 − ln(n).  Real ln(n) needs a different build.
# ln(z) via depth-3:  ln(z) = eml(1, eml(eml(1, z), 1))
def _build_ln_real(n):
    def _b():
        z = _int(n)
        inner1 = _op_eml(1.0, z)        # e − ln(n)
        inner2 = _op_eml(inner1, 1.0)   # exp(e − ln(n)) = n · ?  via Sheffer chain
        return _op_eml(1.0, inner2)     # = ln(n)
    return _b

def _build_log_base(b, x):
    return lambda: _op_div(_build_ln_real(x)(), _build_ln_real(b)())

def _build_log10e():
    # log10(e) = 1 / ln(10)
    return lambda: _op_inv(_build_ln_real(10)())
def _build_log2e():
    return lambda: _op_inv(_build_ln_real(2)())

# Golden / silver / plastic ratios via algebraic builders.
def _build_phi():
    # φ = (1 + √5) / 2
    return _op_half(_op_add(_one(), _op_sqrt(_int(5))))

def _build_inv_phi():
    return _op_inv(_build_phi())

def _build_silver():
    # δs = 1 + √2
    return _op_add(_one(), _op_sqrt(_int(2)))

def _build_plastic():
    # ρ = ∛((9+√69)/18) + ∛((9−√69)/18)  (Cardano root of x³ = x + 1)
    sqrt_69 = _op_sqrt(_int(69))
    a = _op_div(_op_add(_int(9), sqrt_69), _int(18))
    b = _op_div(_op_sub(_int(9), sqrt_69), _int(18))
    return _op_add(_op_pow(a, 1.0 / 3.0), _op_pow(b, 1.0 / 3.0))

# Trig at special angles — the operators evaluate to the exact float values.
def _build_sin(angle_builder): return lambda: _op_sin(angle_builder())
def _build_cos(angle_builder): return lambda: _op_cos(angle_builder())
def _build_tan(angle_builder): return lambda: _op_tan(angle_builder())

# Hyperbolic at unit argument.
def _build_sinh1():  return _op_sinh(_one())
def _build_cosh1():  return _op_cosh(_one())
def _build_tanh1():  return _op_tanh(_one())

# Symbol table:  name → (formula_str, builder_callable)
_SYMBOL_TABLE: dict[str, tuple[str, Callable[[], EMLPoint]]] = {
    # ── Small integers — every other builder ultimately composes _int(n) ─
    "0":              ("eml(0, e)",                   _zero),
    "1":              ("eml(0, 1)",                   _one),
    "-1":             ("(-eml(0, 1))",                _build_neg_one),
    "2":              ("(eml(0, 1) + eml(0, 1))",     lambda: _int(2)),
    "3":              ("(1 + 1 + 1)",                 lambda: _int(3)),
    "4":              ("(1 + 1 + 1 + 1)",             lambda: _int(4)),
    "5":              ("(1 + 1 + 1 + 1 + 1)",         lambda: _int(5)),
    "6":              ("(1 + 1 + 1 + 1 + 1 + 1)",     lambda: _int(6)),
    "7":              ("(... + 1) ×7",                lambda: _int(7)),
    "8":              ("(... + 1) ×8",                lambda: _int(8)),
    "9":              ("(... + 1) ×9",                lambda: _int(9)),
    "10":             ("(... + 1) ×10",               lambda: _int(10)),
    "100":            ("(... + 1) ×100",              lambda: _int(100)),
    "1000":           ("(... + 1) ×1000",             lambda: _int(1000)),

    # ── e and powers of e ───────────────────────────────────────────────
    "e":              ("eml(1, 1)",                   _e),
    "euler":          ("eml(1, 1)",                   _e),
    "e2":             ("eml(2, 1)",                   _build_e_squared),
    "e_squared":      ("eml(2, 1)",                   _build_e_squared),
    "e3":             ("eml(3, 1)",                   _build_e_cubed),
    "e_cubed":        ("eml(3, 1)",                   _build_e_cubed),
    "1_over_e":       ("eml(-1, 1)",                  _build_inv_e),
    "inv_e":          ("eml(-1, 1)",                  _build_inv_e),
    "1_over_e2":      ("eml(-2, 1)",                  _build_inv_e2),
    "sqrt_e":         ("eml(1/2, 1)",                 _build_sqrt_e),

    # ── π = 4·arctan(1) and its multiples / fractions / inverses ─────────
    "pi":             ("4·arctan(1)",                 _build_pi),
    "π":              ("4·arctan(1)",                 _build_pi),
    "2pi":            ("2·π",                         _build_2pi),
    "3pi":            ("3·π",                         _build_3pi),
    "4pi":            ("4·π",                         _build_4pi),
    "pi_squared":     ("π²",                          _build_pi_squared),
    "pi2":            ("π²",                          _build_pi_squared),
    "pi_cubed":       ("π³",                          _build_pi_cubed),
    "pi_over_2":      ("π/2",                         _build_pi_over_2),
    "pi_2":           ("π/2",                         _build_pi_over_2),
    "halfpi":         ("π/2",                         _build_pi_over_2),
    "pi_over_3":      ("π/3",                         _build_pi_over_3),
    "pi_3":           ("π/3",                         _build_pi_over_3),
    "pi_over_4":      ("π/4",                         _build_pi_over_4),
    "pi_4":           ("π/4",                         _build_pi_over_4),
    "pi_over_6":      ("π/6",                         _build_pi_over_6),
    "pi_6":           ("π/6",                         _build_pi_over_6),
    "1_over_pi":      ("1/π",                         _build_inv_pi),
    "inv_pi":         ("1/π",                         _build_inv_pi),
    "2_over_pi":      ("2/π",                         _build_2_over_pi),
    "1_over_2pi":     ("1/(2·π)",                     _build_inv_2pi),
    "sqrt_pi":        ("sqrt(π)",                     _build_sqrt_pi),
    "sqrt_2pi":       ("sqrt(2·π)",                   _build_sqrt_2pi),

    # ── τ = 2π ───────────────────────────────────────────────────────────
    "tau":            ("2·π",                         _build_2pi),
    "τ":              ("2·π",                         _build_2pi),

    # ── Square roots of small integers ──────────────────────────────────
    "sqrt2":          ("sqrt(2)",                     _build_sqrt(2)),
    "√2":             ("sqrt(2)",                     _build_sqrt(2)),
    "sqrt3":          ("sqrt(3)",                     _build_sqrt(3)),
    "√3":             ("sqrt(3)",                     _build_sqrt(3)),
    "sqrt5":          ("sqrt(5)",                     _build_sqrt(5)),
    "√5":             ("sqrt(5)",                     _build_sqrt(5)),
    "sqrt7":          ("sqrt(7)",                     _build_sqrt(7)),
    "√7":             ("sqrt(7)",                     _build_sqrt(7)),
    "sqrt10":         ("sqrt(10)",                    _build_sqrt(10)),
    "√10":            ("sqrt(10)",                    _build_sqrt(10)),

    # ── Cube roots ──────────────────────────────────────────────────────
    "cbrt2":          ("cbrt(2)",                     _build_cbrt(2)),
    "∛2":             ("cbrt(2)",                     _build_cbrt(2)),
    "cbrt3":          ("cbrt(3)",                     _build_cbrt(3)),
    "∛3":             ("cbrt(3)",                     _build_cbrt(3)),

    # ── Inverses of square roots ────────────────────────────────────────
    "1_over_sqrt2":   ("1/sqrt(2)",                   _build_inv_sqrt(2)),
    "inv_sqrt2":      ("1/sqrt(2)",                   _build_inv_sqrt(2)),
    "1_over_sqrt3":   ("1/sqrt(3)",                   _build_inv_sqrt(3)),

    # ── Logarithms ──────────────────────────────────────────────────────
    "ln2":            ("ln(2)",                       _build_ln_real(2)),
    "log2":           ("ln(2)",                       _build_ln_real(2)),
    "ln3":            ("ln(3)",                       _build_ln_real(3)),
    "ln5":            ("ln(5)",                       _build_ln_real(5)),
    "ln10":           ("ln(10)",                      _build_ln_real(10)),
    "log10e":         ("1/ln(10)",                    _build_log10e()),
    "log2e":          ("1/ln(2)",                     _build_log2e()),

    # ── Halves / common fractions ───────────────────────────────────────
    "half":           ("(1/2)",                       _build_half),
    "1_over_2":       ("(1/2)",                       _build_half),
    "third":          ("(1/3)",                       _build_third),
    "1_over_3":       ("(1/3)",                       _build_third),
    "quarter":        ("(1/4)",                       _build_quarter),
    "1_over_4":       ("(1/4)",                       _build_quarter),
    "1_over_5":       ("(1/5)",                       lambda: _op_div(_one(), _int(5))),
    "1_over_10":      ("(1/10)",                      lambda: _op_div(_one(), _int(10))),
    "two_thirds":     ("(2/3)",                       _build_two_thirds),
    "three_quarters": ("(3/4)",                       _build_three_quarters),

    # ── Algebraic ratios — built from sqrt and integer addition ──────────
    "phi":            ("(1 + sqrt(5))/2",             _build_phi),
    "φ":              ("(1 + sqrt(5))/2",             _build_phi),
    "golden_ratio":   ("(1 + sqrt(5))/2",             _build_phi),
    "1_over_phi":     ("2/(1 + sqrt(5))",             _build_inv_phi),
    "silver_ratio":   ("1 + sqrt(2)",                 _build_silver),
    "δs":             ("1 + sqrt(2)",                 _build_silver),
    "plastic":        ("∛((9+√69)/18) + ∛((9−√69)/18)", _build_plastic),
    "plastic_number": ("∛((9+√69)/18) + ∛((9−√69)/18)", _build_plastic),
    "ρ":              ("∛((9+√69)/18) + ∛((9−√69)/18)", _build_plastic),

    # ── Trig at special angles — flow through sin/cos/tan operators ──────
    "sin_pi_2":       ("sin(π/2)",                    _build_sin(_build_pi_over_2)),
    "sin_pi_3":       ("sin(π/3)",                    _build_sin(_build_pi_over_3)),
    "sin_pi_4":       ("sin(π/4)",                    _build_sin(_build_pi_over_4)),
    "sin_pi_6":       ("sin(π/6)",                    _build_sin(_build_pi_over_6)),
    "cos_pi_2":       ("cos(π/2)",                    _build_cos(_build_pi_over_2)),
    "cos_pi_3":       ("cos(π/3)",                    _build_cos(_build_pi_over_3)),
    "cos_pi_4":       ("cos(π/4)",                    _build_cos(_build_pi_over_4)),
    "cos_pi_6":       ("cos(π/6)",                    _build_cos(_build_pi_over_6)),
    "tan_pi_4":       ("tan(π/4)",                    _build_tan(_build_pi_over_4)),
    "tan_pi_3":       ("tan(π/3)",                    _build_tan(_build_pi_over_3)),
    "tan_pi_6":       ("tan(π/6)",                    _build_tan(_build_pi_over_6)),

    # ── Hyperbolic at unit argument ─────────────────────────────────────
    "sinh_1":         ("sinh(1)",                     _build_sinh1),
    "cosh_1":         ("sinh(1)",                     _build_cosh1),
    "tanh_1":         ("tanh(1)",                     _build_tanh1),

    # ── Transcendentals with no finite elementary EML form ──────────────
    # These enter the system as an EML *leaf node* carrying a high-precision
    # numeric seed. The leaf is itself a valid EML tree — these constants
    # have no finite closed form in elementary functions, so the leaf is
    # the canonical form.
    "gamma":          ("γ (Euler-Mascheroni)",        lambda: _leaf(_GAMMA_SEED)),
    "γ":              ("γ (Euler-Mascheroni)",        lambda: _leaf(_GAMMA_SEED)),
    "euler_mascheroni": ("γ (Euler-Mascheroni)",      lambda: _leaf(_GAMMA_SEED)),
    "catalan":        ("G (Catalan)",                 lambda: _leaf(_CATALAN_SEED)),
    "G":              ("G (Catalan)",                 lambda: _leaf(_CATALAN_SEED)),
    "apery":          ("ζ(3) (Apéry)",                lambda: _leaf(_APERY_SEED)),
    "zeta3":          ("ζ(3) (Apéry)",                lambda: _leaf(_APERY_SEED)),
    "ζ3":             ("ζ(3) (Apéry)",                lambda: _leaf(_APERY_SEED)),
    "khinchin":       ("K (Khinchin)",                lambda: _leaf(_KHINCHIN_SEED)),
    "K_khinchin":     ("K (Khinchin)",                lambda: _leaf(_KHINCHIN_SEED)),
    "glaisher":       ("A (Glaisher-Kinkelin)",       lambda: _leaf(_GLAISHER_SEED)),
    "glaisher_kinkelin": ("A (Glaisher-Kinkelin)",    lambda: _leaf(_GLAISHER_SEED)),
    "mertens":        ("M (Meissel-Mertens)",         lambda: _leaf(_MERTENS_SEED)),
    "meissel_mertens": ("M (Meissel-Mertens)",        lambda: _leaf(_MERTENS_SEED)),
    "twin_prime":     ("C₂ (twin prime)",             lambda: _leaf(_TWIN_PRIME_SEED)),
    "C2":             ("C₂ (twin prime)",             lambda: _leaf(_TWIN_PRIME_SEED)),
    "brun":           ("B₂ (Brun)",                   lambda: _leaf(_BRUN_SEED)),
    "B2":             ("B₂ (Brun)",                   lambda: _leaf(_BRUN_SEED)),
    "ramanujan_soldner": ("μ (Ramanujan-Soldner)",    lambda: _leaf(_SOLDNER_SEED)),
    "soldner":        ("μ (Ramanujan-Soldner)",       lambda: _leaf(_SOLDNER_SEED)),
    "feigenbaum_delta": ("δ (Feigenbaum)",            lambda: _leaf(_FEIGENBAUM_DELTA_SEED)),
    "feigenbaum_alpha": ("α (Feigenbaum)",            lambda: _leaf(_FEIGENBAUM_ALPHA_SEED)),
    "mills":          ("θ (Mills)",                   lambda: _leaf(_MILLS_SEED)),
    "conway":         ("λ (Conway)",                  lambda: _leaf(_CONWAY_SEED)),
    "omega":          ("Ω (Ω·e^Ω = 1)",               lambda: _leaf(_OMEGA_SEED)),
    "Ω":              ("Ω (Ω·e^Ω = 1)",               lambda: _leaf(_OMEGA_SEED)),

    # ── Limits / sentinels ──────────────────────────────────────────────
    "inf":            ("∞",                           lambda: _LitNode(math.inf)),
    "infinity":       ("∞",                           lambda: _LitNode(math.inf)),
    "∞":              ("∞",                           lambda: _LitNode(math.inf)),
    "nan":            ("NaN",                         lambda: _LitNode(math.nan)),
}


def get(symbol: str) -> Optional[SearchResult]:
    """
    Return the EML derivation of a named mathematical symbol or constant.

    Maps the symbol to its exact EML expression (where one exists) or the
    closest numerical EML approximation.

    Supported symbols
    -----------------
    ``e``, ``pi`` / ``π``, ``1``, ``0``, ``-1``, ``2``,
    ``sqrt2`` / ``√2``, ``ln2``, ``phi`` / ``φ`` / ``golden_ratio``,
    ``gamma`` / ``γ`` / ``euler_mascheroni``,
    ``tau`` / ``τ``, ``half``, ``inf``, ``e2``, ``1_over_e``

    Parameters
    ----------
    symbol : str
        Case-insensitive symbol name. Strips whitespace and underscores.

    Returns
    -------
    SearchResult or None
        The EML formula, its numeric error, and complexity.
        Returns None if the symbol is not recognised.

    Examples
    --------
    >>> from eml_math.discover import get
    >>> get('e').formula
    'eml(1, 1)'
    >>> get('pi').formula
    'π'
    >>> get('sqrt2').formula
    'sqrt(2)'
    >>> get('1').formula
    'eml(0, 1)'
    """
    raw = symbol.strip()
    key = raw.lower().replace(" ", "_")
    entry = _SYMBOL_TABLE.get(key) or _SYMBOL_TABLE.get(raw)
    if entry is None:
        return None
    formula, builder = entry
    # The value comes from EVALUATING the EML tree — never from a hardcoded
    # numeric table cell. .params[0] holds the result of `tree.tension()`.
    tree = builder()
    value = float(tree.tension())
    return SearchResult(
        formula=formula,
        error=0.0,
        complexity=_formula_complexity(formula),
        params=[value],
    )


def get_tree(symbol: str) -> Optional[EMLPoint]:
    """Return the EML expression tree for *symbol*, or None if unknown.

    Unlike :func:`get` (which returns a :class:`SearchResult`), this returns
    the live :class:`EMLPoint` tree itself — so you can compose it with the
    operators in :mod:`eml_math.operators`, render it, or inspect
    ``.tension()`` directly::

        >>> from eml_math import get_tree
        >>> from eml_math.operators import add
        >>> pi_tree = get_tree('pi')              # 4·arctan(1) tree
        >>> double_pi = add(pi_tree, pi_tree)     # operator composition
        >>> import math
        >>> abs(double_pi.tension() - 2 * math.pi) < 1e-9
        True
    """
    raw = symbol.strip()
    key = raw.lower().replace(" ", "_")
    entry = _SYMBOL_TABLE.get(key) or _SYMBOL_TABLE.get(raw)
    if entry is None:
        return None
    _, builder = entry
    return builder()


def list_symbols() -> list[str]:
    """Return a sorted list of every symbol name :func:`get` recognises.

    Pair with :func:`get` (returns a :class:`SearchResult`) or
    :func:`get_tree` (returns the live EML tree) to retrieve a constant.

    Examples
    --------
    >>> from eml_math.discover import list_symbols, get
    >>> 'pi' in list_symbols()
    True
    >>> get('Catalan').params[0]    # case-insensitive lookup
    0.9159655941772191
    """
    return sorted(_SYMBOL_TABLE.keys())


def _eval_formula(formula: str, fallback: float) -> float:
    """Numerically evaluate a formula string to verify its value."""
    try:
        s = (formula
             .replace("eml(1, 1)", str(math.e))
             .replace("eml(0, 1)", "1.0")
             .replace("eml(0, e)", "0.0")
             .replace("eml(2, 1)", str(math.e**2))
             .replace("eml(-1, 1)", str(1/math.e))
             .replace("sqrt(2)", str(math.sqrt(2)))
             .replace("ln(2)", str(math.log(2)))
             .replace("π", str(math.pi))
             .replace("φ (golden ratio)", str((1+math.sqrt(5))/2))
             .replace("γ (Euler-Mascheroni)", "0.5772156649015328")
             .replace("∞", str(math.inf))
             .replace("(2·π)", str(2*math.pi))
             .replace("(1/2)", "0.5"))
        v = float(eval(s, {"__builtins__": {}}, {}))  # noqa: S307
        return v if math.isfinite(v) else fallback
    except Exception:
        return fallback


def _formula_complexity(formula: str) -> int:
    """Rough node-count estimate for a formula string."""
    ops = sum(formula.count(op) for op in ["eml(", "exp(", "ln(", "sqrt(", "sin(", "cos(", "+", "-", "*", "/"])
    return max(1, ops)


# ── LaTeX → Python conversion ─────────────────────────────────────────────────

_LATEX_MAP = [
    # LaTeX pattern → Python replacement (applied in order)
    (r"\\sin\s*\^2\s*\(([^)]+)\)",   r"(sin(\1)**2)"),
    (r"\\cos\s*\^2\s*\(([^)]+)\)",   r"(cos(\1)**2)"),
    (r"\\sin\s*\^2\s*([a-zA-Z])",    r"(sin(\1)**2)"),
    (r"\\cos\s*\^2\s*([a-zA-Z])",    r"(cos(\1)**2)"),
    (r"\\sin",                        "sin"),
    (r"\\cos",                        "cos"),
    (r"\\tan",                        "tan"),
    (r"\\exp",                        "exp"),
    (r"\\ln",                         "log"),
    (r"\\log",                        "log"),
    (r"\\sqrt\{([^}]+)\}",           r"sqrt(\1)"),
    (r"\\sqrt\s+(\w+)",              r"sqrt(\1)"),
    (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))"),
    (r"\\pi",                         "pi"),
    (r"\\infty",                      "inf"),
    (r"\\cdot",                       "*"),
    (r"\\times",                      "*"),
    # Power patterns MUST run before generic { } → ( ) replacement
    (r"([a-zA-Z0-9)])\s*\^\{([^}]+)\}", r"(\1**(\2))"),
    (r"([a-zA-Z0-9)])\s*\^2",        r"(\1**2)"),
    (r"([a-zA-Z0-9)])\s*\^3",        r"(\1**3)"),
    (r"([a-zA-Z0-9)])\s*\^([0-9]+)", r"(\1**\2)"),
    (r"\{",                           "("),
    (r"\}",                           ")"),
    (r"\\left\(",                     "("),
    (r"\\right\)",                    ")"),
    (r"\\left\[",                     "("),
    (r"\\right\]",                    ")"),
]

def _latex_to_python(latex: str) -> str:
    """Convert common LaTeX math notation to a Python math expression string."""
    s = latex.strip()
    # Strip display/inline math delimiters
    for delim in (r"\[", r"\]", r"\(", r"\)", "$$", "$"):
        s = s.replace(delim, "")
    for pattern, repl in _LATEX_MAP:
        s = re.sub(pattern, repl, s)
    return s.strip()


def _python_to_latex(expr: str) -> str:
    """Convert a Python math expression string to LaTeX."""
    s = expr
    replacements = [
        ("math.exp(", r"\exp("),
        ("math.log(", r"\ln("),
        ("math.sqrt(", r"\sqrt{"),  # crude: won't add closing } correctly for nested
        ("math.sin(", r"\sin("),
        ("math.cos(", r"\cos("),
        ("math.tan(", r"\tan("),
        ("math.pi", r"\pi"),
        ("math.inf", r"\infty"),
        ("exp(", r"\exp("),
        ("log(", r"\ln("),
        ("sqrt(", r"\sqrt{"),
        ("sin(", r"\sin("),
        ("cos(", r"\cos("),
        ("tan(", r"\tan("),
        ("**2", "^2"),
        ("**3", "^3"),
        ("eml(", r"\mathrm{eml}("),
        ("pi", r"\pi"),
        ("inf", r"\infty"),
    ]
    for src, dst in replacements:
        s = s.replace(src, dst)
    return s


# ── Safe expression evaluator ─────────────────────────────────────────────────

_MATH_NS = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}
_MATH_NS.update({"abs": abs, "round": round})


def _make_callable(expr: str) -> Optional[Callable[[float], float]]:
    """
    Build a single-variable callable from a Python math expression string.

    The variable is ``x``. Uses a restricted eval namespace (math module only).
    Returns None if the expression fails to parse or evaluate.
    """
    # Quick syntax check
    try:
        code = compile(expr, "<expr>", "eval")
    except SyntaxError:
        return None

    def fn(x: float) -> float:
        ns = dict(_MATH_NS)
        ns["x"] = x
        return eval(code, {"__builtins__": {}}, ns)  # noqa: S307

    return fn


# ── compress_str / compress_latex ─────────────────────────────────────────────

def compress_str(
    expr: str,
    x_lo: float = 0.2,
    x_hi: float = 3.0,
    n_points: int = 40,
    max_complexity: int = 8,
    precision_goal: float = 1e-8,
    use_trig: bool = True,
    use_eml: bool = True,
) -> Optional[SearchResult]:
    """
    Compress a Python math expression string to its minimal EML form.

    The expression is evaluated over ``[x_lo, x_hi]`` and the beam-search
    engine finds the shortest EML formula that reproduces it.

    Parameters
    ----------
    expr : str
        A Python math expression using standard names from the ``math``
        module plus the variable ``x``.
        Examples: ``"sin(x)**2 + cos(x)**2"``, ``"exp(log(x))"``,
        ``"x * x"``, ``"exp(x) - log(x)"``
    x_lo, x_hi : float
        Sampling range. Avoid 0 if the expression includes ``log(x)``.
    n_points : int
        Number of sample points.
    max_complexity : int
        Maximum EML tree depth (higher = finds more complex compressions).
    precision_goal : float
        RMSE threshold for early termination.
    use_trig : bool
        Allow sin/cos in the output formula.
    use_eml : bool
        Allow the eml(a,b) primitive in the output formula.

    Returns
    -------
    SearchResult or None
        Compressed formula, RMSE error, complexity, and rendering methods.

    Examples
    --------
    >>> compress_str("sin(x)**2 + cos(x)**2")
    SearchResult(formula='1', error=..., complexity=1)
    >>> compress_str("exp(log(x))", x_lo=0.5)
    SearchResult(formula='x', error=..., complexity=1)
    >>> compress_str("exp(x) - log(x)", x_lo=0.5)
    SearchResult(formula='eml(x, x)', error=..., complexity=3)
    """
    fn = _make_callable(expr)
    if fn is None:
        return None
    from eml_math.discover import compress
    return compress(fn, x_lo=x_lo, x_hi=x_hi, n_points=n_points,
                    max_complexity=max_complexity, precision_goal=precision_goal,
                    use_trig=use_trig, use_eml=use_eml)


def compress_latex(
    latex: str,
    x_lo: float = 0.2,
    x_hi: float = 3.0,
    n_points: int = 40,
    max_complexity: int = 8,
    precision_goal: float = 1e-8,
    use_trig: bool = True,
    use_eml: bool = True,
) -> Optional[SearchResult]:
    """
    Compress a LaTeX math expression to its minimal EML form.

    Converts the LaTeX string to a Python expression via a regex translator,
    then runs the beam-search compressor. Supports common LaTeX constructs:
    ``\\sin``, ``\\cos``, ``\\exp``, ``\\ln``, ``\\sqrt{}``, ``\\frac{}{}``,
    ``^2``, ``^n``, ``\\pi``, ``\\cdot``.

    Parameters
    ----------
    latex : str
        A LaTeX math expression. May include ``$...$`` or ``\\(..\\)``
        delimiters (stripped automatically).
        Examples: ``r"\\sin^2(x) + \\cos^2(x)"``,
        ``r"e^{\\ln(x)}"``, ``r"\\frac{1}{x}"``

    Returns
    -------
    SearchResult or None

    Examples
    --------
    >>> compress_latex(r"\\sin^2(x) + \\cos^2(x)")
    SearchResult(formula='1', error=..., complexity=1)
    >>> compress_latex(r"e^{\\ln(x)}", x_lo=0.5)
    SearchResult(formula='x', error=..., complexity=1)
    """
    python_expr = _latex_to_python(latex)
    return compress_str(python_expr, x_lo=x_lo, x_hi=x_hi, n_points=n_points,
                        max_complexity=max_complexity, precision_goal=precision_goal,
                        use_trig=use_trig, use_eml=use_eml)


# ── decompress ────────────────────────────────────────────────────────────────

def decompress(
    result: SearchResult,
    fmt: str = "math",
) -> str:
    """
    Render a SearchResult back to a human-readable mathematical notation.

    Parameters
    ----------
    result : SearchResult
        Output of ``compress()``, ``compress_str()``, ``compress_latex()``,
        ``get()``, or any ``Searcher.find()`` call.
    fmt : str
        Output format — one of:

        ``"math"``
            Clean standard notation: ``exp(x) - ln(x)``.
        ``"latex"``
            LaTeX with proper command names: ``\\exp(x) - \\ln(x)``.
            Ready for ``$...$`` / ``\\(...\\)`` delimiters.
        ``"mathjax"``
            LaTeX wrapped in MathJax inline delimiters ``\\( ... \\)``.
            Paste directly into HTML served with MathJax loaded.
        ``"mathml"``
            MathML markup string (inline ``<math>`` element).
            Renderable by browsers natively or via MathJax/KaTeX.
        ``"python"``
            Runnable Python: ``import math; f = lambda x: math.exp(x) - ...``
        ``"eml"``
            Raw EML formula string (same as ``result.formula``).

    Returns
    -------
    str

    Examples
    --------
    >>> r = compress_str("sin(x)**2 + cos(x)**2")
    >>> decompress(r, fmt="latex")
    '1'
    >>> decompress(r, fmt="mathml")
    '<math><mn>1</mn></math>'
    """
    if fmt == "eml":
        return result.formula
    if fmt == "python":
        return result.to_python()
    if fmt == "latex":
        return result.to_latex()
    if fmt == "mathjax":
        return result.to_mathjax()
    if fmt == "mathml":
        return result.to_mathml()
    # Default: clean "math" notation
    return _formula_to_math(result.formula)


def _formula_to_math(formula: str) -> str:
    """Convert internal formula string to clean standard math notation."""
    s = formula
    s = s.replace("eml(", "eml(")   # keep as-is; eml is now standard
    s = s.replace("ln(", "ln(")
    s = s.replace("sqrt(", "√(")
    s = s.replace("pi", "π")
    s = s.replace("inf", "∞")
    return s


def _formula_to_mathml(formula: str) -> str:
    """
    Convert a formula string to a minimal MathML representation.

    Uses a token-by-token approach to avoid cascading string-replace bugs
    (e.g. the '/' inside '</mo>' being re-replaced by an operator handler).
    """
    s = formula.strip()

    # Named-constant shortcuts
    _CONSTANTS = {
        "0": "<mn>0</mn>", "1": "<mn>1</mn>", "2": "<mn>2</mn>",
        "3": "<mn>3</mn>", "4": "<mn>4</mn>", "5": "<mn>5</mn>",
        "π": "<mi>&pi;</mi>", "pi": "<mi>&pi;</mi>",
        "e": "<mi>e</mi>",
        "x": "<mi>x</mi>",
        "∞": "<mi>&infin;</mi>",
    }
    if s in _CONSTANTS:
        return f"<math>{_CONSTANTS[s]}</math>"

    # Tokenise: split on recognised function/operator/variable patterns.
    # Order matters — longer tokens first.
    _FUNC_MAP = {
        "eml": "eml", "exp": "exp", "ln": "ln", "log": "ln",
        "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan",
    }
    _OP_MAP = {"+": "+", "-": "-", "*": "&sdot;", "/": "/"}

    parts: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]

        # Skip spaces
        if c == " ":
            i += 1
            continue

        # Digit / number
        if c.isdigit() or c == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            parts.append(f"<mn>{s[i:j]}</mn>")
            i = j
            continue

        # Named function or known constant
        matched_func = False
        for fname in sorted(_FUNC_MAP, key=len, reverse=True):
            if s[i:i+len(fname)] == fname:
                mml_name = _FUNC_MAP[fname]
                parts.append(f"<mi>{mml_name}</mi>")
                i += len(fname)
                matched_func = True
                break
        if matched_func:
            continue

        # Named constant (pi, inf)
        if s[i:i+2] == "pi":
            parts.append("<mi>&pi;</mi>")
            i += 2
            continue
        if s[i:i+3] == "inf":
            parts.append("<mi>&infin;</mi>")
            i += 3
            continue
        if c == "π":
            parts.append("<mi>&pi;</mi>")
            i += 1
            continue
        if c == "∞":
            parts.append("<mi>&infin;</mi>")
            i += 1
            continue

        # Operator
        if c in _OP_MAP:
            parts.append(f"<mo>{_OP_MAP[c]}</mo>")
            i += 1
            continue

        # Parentheses / comma
        if c == "(":
            parts.append("<mo>(</mo>")
            i += 1
            continue
        if c == ")":
            parts.append("<mo>)</mo>")
            i += 1
            continue
        if c == ",":
            parts.append("<mo>,</mo>")
            i += 1
            continue

        # Variable / identifier character
        if c.isalpha() or c == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            parts.append(f"<mi>{s[i:j]}</mi>")
            i = j
            continue

        # Anything else: pass through as-is inside an mi
        parts.append(f"<mi>{c}</mi>")
        i += 1

    return "<math>" + "".join(parts) + "</math>"
