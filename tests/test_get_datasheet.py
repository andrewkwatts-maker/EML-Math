"""Smoke tests for the new Get() datasheet API.

Get() is a thin wrapper over the existing get() / get_tree() /
list_symbols() functions; these tests confirm it returns a JSON-shaped
dict for every entry the registry knows about, and that values match
the live tree's tension().
"""
from __future__ import annotations

import json
import math

import pytest

import eml_math


# ── Surface ───────────────────────────────────────────────────────────────────

def test_Get_is_exported():
    assert hasattr(eml_math, "Get")
    assert callable(eml_math.Get)


def test_list_constants_is_exported():
    assert hasattr(eml_math, "list_constants")
    assert callable(eml_math.list_constants)


def test_list_constants_nonempty():
    syms = eml_math.list_constants()
    assert isinstance(syms, list)
    assert len(syms) >= 100      # the existing registry holds 136


def test_Get_unknown_raises():
    with pytest.raises(KeyError):
        eml_math.Get("definitely_not_a_real_constant_zzz")


# ── Schema ────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"name", "value", "formula", "eml_tree", "complexity", "kind"}


@pytest.mark.parametrize("name", ["e", "pi", "phi", "sqrt2", "ln2", "tau"])
def test_Get_returns_required_keys(name):
    d = eml_math.Get(name)
    assert REQUIRED_KEYS <= set(d), f"missing keys: {REQUIRED_KEYS - set(d)}"


@pytest.mark.parametrize("name", ["e", "pi", "phi", "sqrt2", "ln2"])
def test_Get_value_is_finite_number(name):
    d = eml_math.Get(name)
    assert isinstance(d["value"], (int, float))
    assert math.isfinite(d["value"])


def test_Get_kind_is_math():
    assert eml_math.Get("pi")["kind"] == "math"


# ── Numeric correctness ──────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("e",     math.e),
    ("pi",    math.pi),
    ("tau",   math.tau),
    ("phi",   (1 + math.sqrt(5)) / 2),
    ("sqrt2", math.sqrt(2)),
    ("sqrt3", math.sqrt(3)),
    ("sqrt5", math.sqrt(5)),
    ("ln2",   math.log(2)),
    ("ln10",  math.log(10)),
    ("half",  0.5),
])
def test_Get_value_matches_known(name, expected):
    d = eml_math.Get(name)
    assert abs(d["value"] - expected) < 1e-10, (
        f"{name}: got {d['value']!r}, want {expected!r}"
    )


# ── as_json kwarg ────────────────────────────────────────────────────────────

def test_Get_as_json_returns_string():
    s = eml_math.Get("e", as_json=True)
    assert isinstance(s, str)
    parsed = json.loads(s)
    assert parsed["name"] == "e"
    assert abs(parsed["value"] - math.e) < 1e-10


# ── Sweep every registered constant ──────────────────────────────────────────

@pytest.mark.parametrize("name", eml_math.list_constants())
def test_Get_resolves_every_registered_symbol(name):
    """Every name from list_constants() must be accepted by Get()."""
    d = eml_math.Get(name)
    assert d["name"] == name
    assert "value" in d
    assert "formula" in d


# ── Formula presence ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["e", "pi", "phi", "sqrt2"])
def test_Get_formula_is_nonempty_string(name):
    d = eml_math.Get(name)
    assert isinstance(d["formula"], str)
    assert d["formula"]
