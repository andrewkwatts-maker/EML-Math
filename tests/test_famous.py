"""
Tests for eml_math.famous — celebrated equations as EML expressions.

Each equation must:
  1. parse without error (compact tree),
  2. render to SVG / PNG / PDF without error,
  3. evaluate to the known closed-form value when supplied with test inputs.
"""
import math
import pytest

from eml_math.famous import (
    FAMOUS, FamousEquation, get, by_category, all_equations,
)


# ── Registry sanity ─────────────────────────────────────────────────────────

class TestRegistry:
    def test_at_least_20_equations(self):
        assert len(FAMOUS) >= 20

    def test_categories_present(self):
        cats = {eq.category for eq in FAMOUS.values()}
        assert "physics" in cats
        assert "geometry" in cats
        assert "math" in cats

    def test_get_known_returns(self):
        eq = get("einstein_e_mc2")
        assert isinstance(eq, FamousEquation)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown equation"):
            get("not_an_equation")

    def test_by_category(self):
        physics = by_category("physics")
        assert len(physics) >= 8
        assert all(eq.category == "physics" for eq in physics)

    def test_unique_names(self):
        names = [eq.name for eq in all_equations()]
        assert len(set(names)) == len(names)


# ── Each equation: parse + render + evaluate ────────────────────────────────

# Numeric reference values used to verify the EML expression actually
# computes the textbook formula.
_C = 299_792_458.0           # speed of light (m/s)
_G = 6.674e-11               # gravitational constant
_H = 6.626e-34               # Planck constant
_K_E = 8.99e9                # Coulomb constant
_SIGMA_SB = 5.670e-8         # Stefan-Boltzmann constant
_R_GAS = 8.314               # ideal gas constant

REFERENCE = {
    "einstein_e_mc2":     ({"m": 1.0, "c": _C},                 1.0 * _C ** 2),
    "newton_f_ma":        ({"m": 2.0, "a": 9.81},               2.0 * 9.81),
    "newton_gravity":     ({"G": _G, "M": 5.972e24, "m": 70.0, "r": 6.371e6},
                           _G * 5.972e24 * 70.0 / 6.371e6 ** 2),
    "coulomb":            ({"k": _K_E, "q1": 1e-6, "q2": 2e-6, "r": 0.1},
                           _K_E * 1e-6 * 2e-6 / 0.1 ** 2),
    "planck_e_hf":        ({"h": _H, "f": 5e14},                _H * 5e14),
    "de_broglie":         ({"h": _H, "p": 1e-24},               _H / 1e-24),
    "stefan_boltzmann":   ({"sigma": _SIGMA_SB, "T": 5778.0},   _SIGMA_SB * 5778.0 ** 4),
    "lorentz_factor":     ({"v": 0.6 * _C, "c": _C},            1.0 / math.sqrt(1 - 0.36)),
    "relativistic_energy":({"p": 1e-21, "m": 1e-27, "c": _C},
                           math.sqrt((1e-21 * _C) ** 2 + (1e-27 * _C ** 2) ** 2)),
    "kinetic_energy":     ({"m": 2.0, "v": 3.0},                0.5 * 2.0 * 9.0),
    "ohms_law":           ({"I": 0.5, "R": 100.0},              50.0),
    "ideal_gas":          ({"n": 1.0, "R": _R_GAS, "T": 300.0, "V": 0.025},
                           1.0 * _R_GAS * 300.0 / 0.025),
    "pythagoras":         ({"a": 3.0, "b": 4.0},                5.0),
    "circle_area":        ({"r": 2.0},                          math.pi * 4.0),
    "circle_circumference":({"r": 2.0},                         2 * math.pi * 2.0),
    "sphere_volume":      ({"r": 1.0},                          (4.0 / 3.0) * math.pi),
    "sphere_surface_area":({"r": 1.0},                          4.0 * math.pi),
    "cone_volume":        ({"r": 1.0, "h": 3.0},                math.pi),
    "distance_2d":        ({"x1": 0.0, "y1": 0.0, "x2": 3.0, "y2": 4.0}, 5.0),
    "quadratic_formula":  ({"a": 1.0, "b": -3.0, "c": 2.0, "sign": 1.0}, 2.0),
    "golden_ratio":       ({},                                  (1 + math.sqrt(5)) / 2),
    "basel_term":         ({"n": 6.0},                          1.0 / 36.0),
    "compound_interest":  ({"P": 1000.0, "r": 0.05, "n": 12.0, "t": 10.0},
                           1000.0 * (1 + 0.05 / 12) ** (12 * 10)),
    "normal_distribution":({"x": 0.0},                          1.0 / math.sqrt(2 * math.pi)),
}


@pytest.mark.parametrize("name", sorted(FAMOUS.keys()))
def test_parse(name: str) -> None:
    """Every equation must parse without error in compact mode."""
    eq = get(name)
    tree = eq.parse()
    assert tree is not None
    # tree should have at least one node and a label
    assert tree.label != ""


@pytest.mark.parametrize("name", sorted(FAMOUS.keys()))
def test_renders_svg(name: str) -> None:
    """Every equation produces a non-empty SVG flow diagram."""
    eq = get(name)
    svg = eq.flow_svg(width=600, height=400)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert eq.output in svg     # the output label must appear


@pytest.mark.parametrize("name", sorted(FAMOUS.keys()))
def test_renders_png(name: str) -> None:
    eq = get(name)
    png = eq.flow_png(width=400, height=300)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("name", sorted(FAMOUS.keys()))
def test_renders_pdf(name: str) -> None:
    eq = get(name)
    pdf = eq.flow_pdf(width=400, height=300)
    assert pdf[:5] == b"%PDF-"


@pytest.mark.parametrize("name", sorted(REFERENCE.keys()))
def test_evaluate_matches_textbook(name: str) -> None:
    """Numeric evaluation reproduces the textbook value."""
    eq = get(name)
    ctx, expected = REFERENCE[name]
    got = eq.evaluate(ctx)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-12), (
        f"{name}: got {got}, expected {expected}"
    )


# ── Specifically: quadratic formula gives both roots via sign branch ────────

class TestQuadraticBothRoots:
    """x² − 3x + 2 = 0 has roots 1 and 2."""
    EQ = "quadratic_formula"
    BASE = {"a": 1.0, "b": -3.0, "c": 2.0}

    def test_plus_root(self):
        ctx = dict(self.BASE, sign=+1.0)
        assert get(self.EQ).evaluate(ctx) == pytest.approx(2.0)

    def test_minus_root(self):
        ctx = dict(self.BASE, sign=-1.0)
        assert get(self.EQ).evaluate(ctx) == pytest.approx(1.0)

    def test_irrational_roots_x2_minus_2(self):
        # x² − 2 = 0 → x = ±√2
        ctx = {"a": 1.0, "b": 0.0, "c": -2.0, "sign": +1.0}
        assert get(self.EQ).evaluate(ctx) == pytest.approx(math.sqrt(2))
        ctx["sign"] = -1.0
        assert get(self.EQ).evaluate(ctx) == pytest.approx(-math.sqrt(2))


# ── Output label appears in renderer output ──────────────────────────────────

class TestOutputLabel:
    def test_default_uses_equation_output(self):
        eq = get("einstein_e_mc2")
        svg = eq.flow_svg()
        assert "E" in svg

    def test_override_works(self):
        eq = get("einstein_e_mc2")
        svg = eq.flow_svg(output_label="energy")
        assert "energy" in svg
