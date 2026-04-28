"""
eml_math.famous — celebrated equations of physics, geometry, and mathematics
expressed as EML descriptions ready for parsing, evaluation, and flow-diagram
rendering.

Each entry is a :class:`FamousEquation` carrying:

* ``name``        — short identifier (file-safe)
* ``title``       — pretty display title
* ``description`` — one-line summary
* ``inputs``      — ordered list of input names (matches eml_vec(...) names)
* ``output``      — name of the output quantity (used as flow-diagram label)
* ``eml``         — the ``"EML: …"`` expression string
* ``category``    — ``"physics"`` | ``"geometry"`` | ``"math"``
* ``notes``       — optional commentary (e.g. how ± is handled)

A few design notes:

* The quadratic formula's ± is encoded as an explicit ``sign`` input that
  takes ±1.  Setting ``sign = +1`` gives one root, ``sign = −1`` the other —
  faithful to the EML algebra (which is real-valued; complex numbers would
  require leaving the binary primitive).
* All formulas are parsed and (with appropriate test inputs) evaluated by
  the test suite to confirm they reduce to the expected numeric result.

Quick start
-----------
>>> from eml_math.famous import FAMOUS, get
>>> einstein = get("einstein_e_mc2")
>>> tree = einstein.parse()
>>> tree.flow_svg(width=600, height=400, output_label=einstein.output)
'<svg …</svg>'
>>> einstein.evaluate({"m": 1.0, "c": 299792458}) == 299792458 ** 2
True
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "FamousEquation",
    "FAMOUS",
    "get",
    "by_category",
    "all_equations",
]


# ── Record ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FamousEquation:
    """A famous equation expressed as an EML description."""
    name:        str
    title:       str
    description: str
    inputs:      List[str]
    output:      str
    eml:         str
    category:    str               # 'physics' | 'geometry' | 'math'
    notes:       str = ""

    def parse(self, *, expand_eml: bool = False):
        """Return the parsed :class:`EMLTreeNode` (compact / ops-level by default)."""
        from eml_math.tree import parse_eml_tree
        return parse_eml_tree(self.eml, expand_eml=expand_eml)

    def evaluate(self, context: Dict[str, float]) -> float:
        """Evaluate against a parameter context. Convenience wrapper."""
        from eml_math.evaluator import EMLEvaluator
        return EMLEvaluator(context, strict=False).eval(self.eml)

    def flow_svg(self, **kw) -> str:
        """Render the flow diagram as SVG; output label defaults to ``self.output``."""
        kw.setdefault("output_label", self.output)
        return self.parse().flow_svg(**kw)

    def flow_png(self, **kw) -> bytes:
        kw.setdefault("output_label", self.output)
        return self.parse().flow_png(**kw)

    def flow_pdf(self, **kw) -> bytes:
        kw.setdefault("output_label", self.output)
        return self.parse().flow_pdf(**kw)


# ── Registry ─────────────────────────────────────────────────────────────────

FAMOUS: Dict[str, FamousEquation] = {}


def _register(eq: FamousEquation) -> None:
    FAMOUS[eq.name] = eq


def get(name: str) -> FamousEquation:
    if name not in FAMOUS:
        raise KeyError(f"unknown equation: {name!r}. Available: {sorted(FAMOUS)}")
    return FAMOUS[name]


def by_category(category: str) -> List[FamousEquation]:
    return [eq for eq in FAMOUS.values() if eq.category == category]


def all_equations() -> List[FamousEquation]:
    return list(FAMOUS.values())


# ── PHYSICS ──────────────────────────────────────────────────────────────────

_register(FamousEquation(
    name="einstein_e_mc2",
    title="Einstein mass-energy equivalence",
    description="E = m c²",
    inputs=["m", "c"],
    output="E",
    eml="EML: ops.mul(eml_vec('m'), ops.pow(eml_vec('c'), eml_scalar(2.0))) — E = mc²",
    category="physics",
))

_register(FamousEquation(
    name="newton_f_ma",
    title="Newton's second law",
    description="F = m a",
    inputs=["m", "a"],
    output="F",
    eml="EML: ops.mul(eml_vec('m'), eml_vec('a')) — F = ma",
    category="physics",
))

_register(FamousEquation(
    name="newton_gravity",
    title="Newton's law of universal gravitation",
    description="F = G M m / r²",
    inputs=["G", "M", "m", "r"],
    output="F",
    eml="EML: ops.div(ops.mul(eml_vec('G'), ops.mul(eml_vec('M'), eml_vec('m'))), "
        "ops.pow(eml_vec('r'), eml_scalar(2.0))) — F = GMm/r²",
    category="physics",
))

_register(FamousEquation(
    name="coulomb",
    title="Coulomb's law",
    description="F = k q₁ q₂ / r²",
    inputs=["k", "q1", "q2", "r"],
    output="F",
    eml="EML: ops.div(ops.mul(eml_vec('k'), ops.mul(eml_vec('q1'), eml_vec('q2'))), "
        "ops.pow(eml_vec('r'), eml_scalar(2.0))) — F = kq₁q₂/r²",
    category="physics",
))

_register(FamousEquation(
    name="planck_e_hf",
    title="Planck-Einstein relation",
    description="E = h f",
    inputs=["h", "f"],
    output="E",
    eml="EML: ops.mul(eml_vec('h'), eml_vec('f')) — E = hf",
    category="physics",
))

_register(FamousEquation(
    name="de_broglie",
    title="de Broglie wavelength",
    description="λ = h / p",
    inputs=["h", "p"],
    output="lambda",
    eml="EML: ops.div(eml_vec('h'), eml_vec('p')) — λ = h/p",
    category="physics",
))

_register(FamousEquation(
    name="stefan_boltzmann",
    title="Stefan-Boltzmann law",
    description="P = σ T⁴",
    inputs=["sigma", "T"],
    output="P",
    eml="EML: ops.mul(eml_vec('sigma'), ops.pow(eml_vec('T'), eml_scalar(4.0))) — P = σT⁴",
    category="physics",
))

_register(FamousEquation(
    name="lorentz_factor",
    title="Lorentz factor",
    description="γ = 1 / √(1 − v²/c²)",
    inputs=["v", "c"],
    output="gamma",
    eml="EML: ops.inv(ops.sqrt(ops.sub(eml_scalar(1.0), "
        "ops.div(ops.pow(eml_vec('v'), eml_scalar(2.0)), "
        "ops.pow(eml_vec('c'), eml_scalar(2.0)))))) — γ = 1/√(1−v²/c²)",
    category="physics",
))

_register(FamousEquation(
    name="relativistic_energy",
    title="Relativistic energy-momentum relation",
    description="E = √((pc)² + (mc²)²)",
    inputs=["p", "m", "c"],
    output="E",
    eml="EML: ops.sqrt(ops.add(ops.pow(ops.mul(eml_vec('p'), eml_vec('c')), eml_scalar(2.0)), "
        "ops.pow(ops.mul(eml_vec('m'), ops.pow(eml_vec('c'), eml_scalar(2.0))), eml_scalar(2.0)))) "
        "— E² = (pc)² + (mc²)²",
    category="physics",
))

_register(FamousEquation(
    name="kinetic_energy",
    title="Classical kinetic energy",
    description="KE = ½ m v²",
    inputs=["m", "v"],
    output="KE",
    eml="EML: ops.mul(eml_scalar(0.5), ops.mul(eml_vec('m'), "
        "ops.pow(eml_vec('v'), eml_scalar(2.0)))) — KE = ½mv²",
    category="physics",
))

_register(FamousEquation(
    name="ohms_law",
    title="Ohm's law",
    description="V = I R",
    inputs=["I", "R"],
    output="V",
    eml="EML: ops.mul(eml_vec('I'), eml_vec('R')) — V = IR",
    category="physics",
))

_register(FamousEquation(
    name="ideal_gas",
    title="Ideal gas law (solved for P)",
    description="P = n R T / V",
    inputs=["n", "R", "T", "V"],
    output="P",
    eml="EML: ops.div(ops.mul(eml_vec('n'), ops.mul(eml_vec('R'), eml_vec('T'))), "
        "eml_vec('V')) — P = nRT/V",
    category="physics",
))


# ── GEOMETRY ─────────────────────────────────────────────────────────────────

_register(FamousEquation(
    name="pythagoras",
    title="Pythagorean theorem",
    description="c = √(a² + b²)",
    inputs=["a", "b"],
    output="c",
    eml="EML: ops.sqrt(ops.add(ops.pow(eml_vec('a'), eml_scalar(2.0)), "
        "ops.pow(eml_vec('b'), eml_scalar(2.0)))) — c = √(a²+b²)",
    category="geometry",
))

_register(FamousEquation(
    name="circle_area",
    title="Area of a circle",
    description="A = π r²",
    inputs=["r"],
    output="A",
    eml="EML: ops.mul(eml_pi(), ops.pow(eml_vec('r'), eml_scalar(2.0))) — A = πr²",
    category="geometry",
    notes="π enters via eml_pi() rather than a 3.14159… literal so the diagram shows it as a named symbol.",
))

_register(FamousEquation(
    name="circle_circumference",
    title="Circumference of a circle",
    description="C = 2π r",
    inputs=["r"],
    output="C",
    eml="EML: ops.mul(eml_scalar(2.0), ops.mul(eml_pi(), eml_vec('r'))) — C = 2πr",
    category="geometry",
))

_register(FamousEquation(
    name="sphere_volume",
    title="Volume of a sphere",
    description="V = (4/3) π r³",
    inputs=["r"],
    output="V",
    eml="EML: ops.mul(ops.div(eml_scalar(4.0), eml_scalar(3.0)), "
        "ops.mul(eml_pi(), ops.pow(eml_vec('r'), eml_scalar(3.0)))) — V = (4/3)πr³",
    category="geometry",
))

_register(FamousEquation(
    name="sphere_surface_area",
    title="Surface area of a sphere",
    description="A = 4π r²",
    inputs=["r"],
    output="A",
    eml="EML: ops.mul(eml_scalar(4.0), ops.mul(eml_pi(), "
        "ops.pow(eml_vec('r'), eml_scalar(2.0)))) — A = 4πr²",
    category="geometry",
))

_register(FamousEquation(
    name="cone_volume",
    title="Volume of a cone",
    description="V = (1/3) π r² h",
    inputs=["r", "h"],
    output="V",
    eml="EML: ops.mul(ops.div(eml_scalar(1.0), eml_scalar(3.0)), "
        "ops.mul(eml_pi(), ops.mul(ops.pow(eml_vec('r'), eml_scalar(2.0)), eml_vec('h')))) "
        "— V = πr²h/3",
    category="geometry",
))

_register(FamousEquation(
    name="distance_2d",
    title="Euclidean distance (2D)",
    description="d = √((x₂−x₁)² + (y₂−y₁)²)",
    inputs=["x1", "y1", "x2", "y2"],
    output="d",
    eml="EML: ops.sqrt(ops.add("
        "ops.pow(ops.sub(eml_vec('x2'), eml_vec('x1')), eml_scalar(2.0)), "
        "ops.pow(ops.sub(eml_vec('y2'), eml_vec('y1')), eml_scalar(2.0)))) "
        "— d = √((Δx)² + (Δy)²)",
    category="geometry",
))


# ── MATHEMATICS ──────────────────────────────────────────────────────────────

_register(FamousEquation(
    name="quadratic_formula",
    title="Quadratic formula",
    description="x = (−b + sign·√(b² − 4ac)) / 2a",
    inputs=["a", "b", "c", "sign"],
    output="x",
    eml="EML: ops.div(ops.add(ops.neg(eml_vec('b')), "
        "ops.mul(eml_vec('sign'), ops.sqrt(ops.sub(ops.pow(eml_vec('b'), eml_scalar(2.0)), "
        "ops.mul(eml_scalar(4.0), ops.mul(eml_vec('a'), eml_vec('c'))))))), "
        "ops.mul(eml_scalar(2.0), eml_vec('a'))) — quadratic root via sign branch",
    category="math",
    notes=(
        "EML is real-valued, so the ± of the textbook quadratic formula is "
        "encoded as an explicit `sign` input ∈ {-1, +1}. Evaluating with "
        "sign=+1 gives one root, sign=-1 the other. (Complex numbers are "
        "avoided because they would break the single-real-axis EML primitive.)"
    ),
))

_register(FamousEquation(
    name="golden_ratio",
    title="Golden ratio",
    description="φ = (1 + √5) / 2",
    inputs=[],
    output="phi",
    eml="EML: ops.div(ops.add(eml_scalar(1.0), ops.sqrt(eml_scalar(5.0))), "
        "eml_scalar(2.0)) — φ = (1+√5)/2",
    category="math",
))

_register(FamousEquation(
    name="basel_term",
    title="Basel-problem term (single n)",
    description="aₙ = 1 / n²",
    inputs=["n"],
    output="a_n",
    eml="EML: ops.div(eml_scalar(1.0), ops.pow(eml_vec('n'), eml_scalar(2.0))) — 1/n²",
    category="math",
    notes="The Basel sum Σ 1/n² = π²/6 — the term shown is the summand.",
))

_register(FamousEquation(
    name="compound_interest",
    title="Compound interest",
    description="A = P (1 + r/n)^(nt)",
    inputs=["P", "r", "n", "t"],
    output="A",
    eml="EML: ops.mul(eml_vec('P'), ops.pow("
        "ops.add(eml_scalar(1.0), ops.div(eml_vec('r'), eml_vec('n'))), "
        "ops.mul(eml_vec('n'), eml_vec('t')))) — A = P(1 + r/n)^(nt)",
    category="math",
))

_register(FamousEquation(
    name="normal_distribution",
    title="Standard-normal density",
    description="f(x) = exp(−x²/2) / √(2π)",
    inputs=["x"],
    output="f",
    eml="EML: ops.div(ops.exp(ops.neg(ops.div(ops.pow(eml_vec('x'), eml_scalar(2.0)), "
        "eml_scalar(2.0)))), ops.sqrt(ops.mul(eml_scalar(2.0), eml_pi()))) "
        "— f(x) = e^(−x²/2) / √(2π)",
    category="math",
))
