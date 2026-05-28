"""Numerical equivalence tests for every entry in ``FAMOUS``.

For each famous equation we know the **traditional form** as a plain
Python lambda; the **EML form** lives on ``FamousEquation.eml`` and is
evaluated via :class:`EMLEvaluator`. The test draws 1000 random
variable assignments per equation and asserts the two agree to a
relative tolerance.

Random ranges are chosen to stay inside each formula's natural
domain (positive arguments for ln/sqrt, ``|v| < c`` for Lorentz
factors, valid triangle sides for Heron, etc.). Where the formula
diverges (geometric series at ``r == 1``, log at ``p == 0``) we
exclude those points.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, Iterable, Tuple

import pytest

from eml_math import FAMOUS
from eml_math.evaluator import EMLEvaluator


SAMPLES = 1000
RTOL = 1e-9
ATOL = 1e-9


# ---------------------------------------------------------------------------
# Reference Python lambdas — one per famous equation.  Each takes a context
# dict (parameter → value) and returns the expected scalar.
# ---------------------------------------------------------------------------

REFERENCE: Dict[str, Callable[[Dict[str, float]], float]] = {
    "einstein_e_mc2":       lambda c: c["m"] * c["c"] ** 2,
    "newton_f_ma":          lambda c: c["m"] * c["a"],
    "newton_gravity":       lambda c: c["G"] * c["M"] * c["m"] / c["r"] ** 2,
    "coulomb":              lambda c: c["k"] * c["q1"] * c["q2"] / c["r"] ** 2,
    "planck_e_hf":          lambda c: c["h"] * c["f"],
    "de_broglie":           lambda c: c["h"] / c["p"],
    "stefan_boltzmann":     lambda c: c["sigma"] * c["T"] ** 4,
    "lorentz_factor":       lambda c: 1.0 / math.sqrt(1.0 - (c["v"] / c["c"]) ** 2),
    "relativistic_energy":  lambda c: math.sqrt((c["p"] * c["c"]) ** 2
                                                + (c["m"] * c["c"] ** 2) ** 2),
    "kinetic_energy":       lambda c: 0.5 * c["m"] * c["v"] ** 2,
    "ohms_law":             lambda c: c["I"] * c["R"],
    "ideal_gas":            lambda c: c["n"] * c["R"] * c["T"] / c["V"],
    "pythagoras":           lambda c: math.sqrt(c["a"] ** 2 + c["b"] ** 2),
    "circle_area":          lambda c: math.pi * c["r"] ** 2,
    "circle_circumference": lambda c: 2.0 * math.pi * c["r"],
    "sphere_volume":        lambda c: (4.0 / 3.0) * math.pi * c["r"] ** 3,
    "sphere_surface_area":  lambda c: 4.0 * math.pi * c["r"] ** 2,
    "cone_volume":          lambda c: (1.0 / 3.0) * math.pi * c["r"] ** 2 * c["h"],
    "distance_2d":          lambda c: math.sqrt((c["x2"] - c["x1"]) ** 2
                                                + (c["y2"] - c["y1"]) ** 2),
    "quadratic_root_plus":  lambda c: ((-c["b"] + math.sqrt(c["b"] ** 2 - 4 * c["a"] * c["c"]))
                                       / (2 * c["a"])),
    "quadratic_root_minus": lambda c: ((-c["b"] - math.sqrt(c["b"] ** 2 - 4 * c["a"] * c["c"]))
                                       / (2 * c["a"])),
    "quadratic_formula":    lambda c: ((-c["b"] + c["sign"] * math.sqrt(c["b"] ** 2 - 4 * c["a"] * c["c"]))
                                       / (2 * c["a"])),
    "basel_term":           lambda c: 1.0 / c["n"] ** 2,
    "compound_interest":    lambda c: c["P"] * (1.0 + c["r"] / c["n"]) ** (c["n"] * c["t"]),
    "normal_distribution":  lambda c: math.exp(-c["x"] ** 2 / 2.0) / math.sqrt(2.0 * math.pi),
    "schwarzschild_radius": lambda c: 2.0 * c["G"] * c["M"] / c["c"] ** 2,
    "time_dilation":        lambda c: c["dt"] / math.sqrt(1.0 - (c["v"] / c["c"]) ** 2),
    "escape_velocity":      lambda c: math.sqrt(2.0 * c["G"] * c["M"] / c["r"]),
    "rydberg":              lambda c: c["R"] * (1.0 / c["n1"] ** 2 - 1.0 / c["n2"] ** 2),
    "wien_displacement":    lambda c: c["b"] / c["T"],
    "hubble_law":           lambda c: c["H0"] * c["d"],
    "distance_3d":          lambda c: math.sqrt((c["x2"] - c["x1"]) ** 2
                                                + (c["y2"] - c["y1"]) ** 2
                                                + (c["z2"] - c["z1"]) ** 2),
    "ellipse_area":         lambda c: math.pi * c["a"] * c["b"],
    "cylinder_volume":      lambda c: math.pi * c["r"] ** 2 * c["h"],
    # ``s`` (semi-perimeter) is passed as a pre-computed input — the EML
    # form doesn't recompute (a+b+c)/2 internally. Reference uses ``s``
    # directly to match.
    "triangle_area_heron":  lambda c: math.sqrt(
        c["s"] * (c["s"] - c["a"]) * (c["s"] - c["b"]) * (c["s"] - c["c"])
    ),
    # phi and psi are pre-computed inputs; both can be ANY real.
    "binet_fibonacci":      lambda c: (c["phi"] ** c["n"] - c["psi"] ** c["n"])
                                       / math.sqrt(5.0),
    "harmonic_term":        lambda c: 1.0 / c["n"],
    "geometric_series_sum_finite":
                            lambda c: (1.0 - c["r"] ** c["n"]) / (1.0 - c["r"]),
    "logistic_function":    lambda c: 1.0 / (1.0 + math.exp(-c["x"])),
    "entropy_shannon_term": lambda c: -c["p"] * math.log2(c["p"]),
    "haversine_central_angle":
                            lambda c: math.sin(c["theta"] / 2.0) ** 2,
    "bayes_rule":           lambda c: c["P_B_given_A"] * c["P_A"] / c["P_B"],
    "larmor_power":         lambda c: c["q"] ** 2 * c["a"] ** 2 / (6.0 * math.pi
                                                                    * c["eps0"] * c["c"] ** 3),
    "hawking_temperature":  lambda c: c["hbar"] * c["c"] ** 3 / (
        8.0 * math.pi * c["G"] * c["M"] * c["kB"]
    ),
    "bekenstein_hawking_entropy":
                            lambda c: c["kB"] * c["A"] / (4.0 * c["lP"] ** 2),
    "golden_ratio":         lambda c: (1.0 + math.sqrt(5.0)) / 2.0,
}


# ---------------------------------------------------------------------------
# Random sampling — each lambda produces a context dict.
# Ranges are chosen to stay inside each formula's natural domain.
# ---------------------------------------------------------------------------

def _u(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


SAMPLER: Dict[str, Callable[[random.Random], Dict[str, float]]] = {
    "einstein_e_mc2":       lambda r: {"m": _u(r, 0.1, 5), "c": _u(r, 0.1, 5)},
    "newton_f_ma":          lambda r: {"m": _u(r, 0.1, 5), "a": _u(r, 0.1, 5)},
    "newton_gravity":       lambda r: {"G": _u(r, 0.1, 5), "M": _u(r, 0.1, 5),
                                       "m": _u(r, 0.1, 5), "r": _u(r, 0.5, 5)},
    "coulomb":              lambda r: {"k": _u(r, 0.1, 5), "q1": _u(r, 0.1, 5),
                                       "q2": _u(r, 0.1, 5), "r": _u(r, 0.5, 5)},
    "planck_e_hf":          lambda r: {"h": _u(r, 0.1, 5), "f": _u(r, 0.1, 5)},
    "de_broglie":           lambda r: {"h": _u(r, 0.1, 5), "p": _u(r, 0.5, 5)},
    "stefan_boltzmann":     lambda r: {"sigma": _u(r, 0.1, 5), "T": _u(r, 0.1, 5)},
    "lorentz_factor":       lambda r: {"v": _u(r, 0.01, 0.9), "c": 1.0},
    "relativistic_energy":  lambda r: {"p": _u(r, 0.1, 5), "m": _u(r, 0.1, 5),
                                       "c": _u(r, 0.1, 5)},
    "kinetic_energy":       lambda r: {"m": _u(r, 0.1, 5), "v": _u(r, 0.1, 5)},
    "ohms_law":             lambda r: {"I": _u(r, 0.1, 5), "R": _u(r, 0.1, 5)},
    "ideal_gas":            lambda r: {"n": _u(r, 0.1, 5), "R": _u(r, 0.1, 5),
                                       "T": _u(r, 0.1, 5), "V": _u(r, 0.5, 5)},
    "pythagoras":           lambda r: {"a": _u(r, 0.1, 5), "b": _u(r, 0.1, 5)},
    "circle_area":          lambda r: {"r": _u(r, 0.1, 5)},
    "circle_circumference": lambda r: {"r": _u(r, 0.1, 5)},
    "sphere_volume":        lambda r: {"r": _u(r, 0.1, 5)},
    "sphere_surface_area":  lambda r: {"r": _u(r, 0.1, 5)},
    "cone_volume":          lambda r: {"r": _u(r, 0.1, 5), "h": _u(r, 0.1, 5)},
    "distance_2d":          lambda r: {"x1": _u(r, -5, 5), "y1": _u(r, -5, 5),
                                       "x2": _u(r, -5, 5), "y2": _u(r, -5, 5)},
    # Quadratic: must have non-negative discriminant. Choose a, b, c so it does.
    "quadratic_root_plus":  lambda r: _quad_ctx(r),
    "quadratic_root_minus": lambda r: _quad_ctx(r),
    "quadratic_formula":    lambda r: dict(_quad_ctx(r), sign=r.choice([1.0, -1.0])),
    "basel_term":           lambda r: {"n": float(r.randint(1, 100))},
    "compound_interest":    lambda r: {"P": _u(r, 100, 1000), "r": _u(r, 0.01, 0.5),
                                       "n": float(r.randint(1, 12)),
                                       "t": _u(r, 0.5, 10)},
    "normal_distribution":  lambda r: {"x": _u(r, -3, 3)},
    "schwarzschild_radius": lambda r: {"G": _u(r, 0.1, 5), "M": _u(r, 0.1, 5),
                                       "c": _u(r, 0.5, 5)},
    "time_dilation":        lambda r: {"dt": _u(r, 0.1, 5), "v": _u(r, 0.01, 0.9),
                                       "c": 1.0},
    "escape_velocity":      lambda r: {"G": _u(r, 0.1, 5), "M": _u(r, 0.1, 5),
                                       "r": _u(r, 0.5, 5)},
    "rydberg":              lambda r: _rydberg_ctx(r),
    "wien_displacement":    lambda r: {"b": _u(r, 0.1, 5), "T": _u(r, 0.1, 5)},
    "hubble_law":           lambda r: {"H0": _u(r, 0.1, 5), "d": _u(r, 0.1, 5)},
    "distance_3d":          lambda r: {"x1": _u(r, -5, 5), "y1": _u(r, -5, 5),
                                       "z1": _u(r, -5, 5), "x2": _u(r, -5, 5),
                                       "y2": _u(r, -5, 5), "z2": _u(r, -5, 5)},
    "ellipse_area":         lambda r: {"a": _u(r, 0.1, 5), "b": _u(r, 0.1, 5)},
    "cylinder_volume":      lambda r: {"r": _u(r, 0.1, 5), "h": _u(r, 0.1, 5)},
    "triangle_area_heron":  lambda r: _triangle_ctx(r),
    "binet_fibonacci":      lambda r: {
        "phi": (1.0 + math.sqrt(5.0)) / 2.0,
        "psi": (1.0 - math.sqrt(5.0)) / 2.0,
        "n":   float(r.randint(1, 15)),
    },
    "harmonic_term":        lambda r: {"n": float(r.randint(1, 100))},
    "geometric_series_sum_finite":
                            lambda r: {"r": r.choice([_u(r, 0.1, 0.9), _u(r, 1.1, 3)]),
                                       "n": float(r.randint(2, 10))},
    "logistic_function":    lambda r: {"x": _u(r, -5, 5)},
    "entropy_shannon_term": lambda r: {"p": _u(r, 0.01, 0.99)},
    "haversine_central_angle":
                            lambda r: {"theta": _u(r, 0.01, math.pi - 0.01)},
    "bayes_rule":           lambda r: {"P_B_given_A": _u(r, 0.1, 0.9),
                                       "P_A": _u(r, 0.1, 0.9),
                                       "P_B": _u(r, 0.2, 0.9)},
    "larmor_power":         lambda r: {"q": _u(r, 0.1, 5), "a": _u(r, 0.1, 5),
                                       "eps0": _u(r, 0.1, 5),
                                       "c": _u(r, 0.5, 5)},
    "hawking_temperature":  lambda r: {"hbar": _u(r, 0.1, 5), "c": _u(r, 0.5, 5),
                                       "G": _u(r, 0.1, 5), "M": _u(r, 0.5, 5),
                                       "kB": _u(r, 0.1, 5)},
    "bekenstein_hawking_entropy":
                            lambda r: {"kB": _u(r, 0.1, 5), "A": _u(r, 0.1, 5),
                                       "lP": _u(r, 0.5, 5)},
    "golden_ratio":         lambda r: {},
}


def _quad_ctx(rng: random.Random) -> Dict[str, float]:
    """Random quadratic with guaranteed-real roots (b² ≥ 4ac)."""
    a = rng.uniform(0.5, 3.0)
    c = rng.uniform(-3.0, 3.0)
    # Pick b so b² >= 4ac + slack.
    min_b2 = max(0.0, 4 * a * c) + 0.5
    b = math.copysign(math.sqrt(min_b2 + rng.uniform(0.0, 5.0)),
                      rng.choice([1.0, -1.0]))
    return {"a": a, "b": b, "c": c}


def _rydberg_ctx(rng: random.Random) -> Dict[str, float]:
    n1 = rng.randint(1, 5)
    n2 = rng.randint(n1 + 1, n1 + 5)
    return {"R": rng.uniform(0.1, 5.0), "n1": float(n1), "n2": float(n2)}


def _triangle_ctx(rng: random.Random) -> Dict[str, float]:
    """Valid triangle plus the pre-computed semi-perimeter ``s``."""
    a = rng.uniform(1.0, 5.0)
    b = rng.uniform(1.0, 5.0)
    # c bounded by |a-b| < c < a+b.
    c = rng.uniform(abs(a - b) + 0.1, a + b - 0.1)
    return {"a": a, "b": b, "c": c, "s": (a + b + c) / 2.0}


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

ALL_NAMES = sorted(FAMOUS.keys() & REFERENCE.keys() & SAMPLER.keys())


@pytest.mark.parametrize("name", ALL_NAMES)
def test_famous_eml_matches_reference(name: str) -> None:
    """1000 random combos × every famous equation × EML form ≡ reference form."""
    eq = FAMOUS[name]
    reference = REFERENCE[name]
    sampler = SAMPLER[name]
    rng = random.Random(hash(name) & 0xFFFFFFFF)   # deterministic per equation

    mismatches: list[Tuple[Dict[str, float], float, float]] = []
    for _ in range(SAMPLES):
        ctx = sampler(rng)
        try:
            expected = reference(ctx)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if not math.isfinite(expected):
            continue
        try:
            got = EMLEvaluator(ctx, strict=False).eval(eq.eml)
        except Exception as exc:                       # noqa: BLE001
            pytest.fail(f"{name}: EML eval raised {type(exc).__name__}: {exc}\n"
                        f"  context={ctx}")
        if not math.isfinite(got):
            mismatches.append((ctx, expected, got))
            continue
        if not math.isclose(got, expected, rel_tol=RTOL, abs_tol=ATOL):
            mismatches.append((ctx, expected, got))

    if mismatches:
        head = mismatches[:3]
        msg = "\n  ".join(
            f"ctx={ctx}  expected={exp:.6g}  EML={got:.6g}"
            for ctx, exp, got in head
        )
        pytest.fail(
            f"{name}: {len(mismatches)}/{SAMPLES} random combinations "
            f"disagreed between the EML form and the reference traditional form.\n"
            f"  first {len(head)} mismatches:\n  {msg}"
        )


def test_coverage_is_complete() -> None:
    """Every entry in FAMOUS must have BOTH a reference lambda AND a sampler."""
    missing_ref = set(FAMOUS) - set(REFERENCE)
    missing_sam = set(FAMOUS) - set(SAMPLER)
    assert not missing_ref, (
        f"FAMOUS equations without a reference Python form: {sorted(missing_ref)}"
    )
    assert not missing_sam, (
        f"FAMOUS equations without a random sampler: {sorted(missing_sam)}"
    )
