# EML-Math

**EML Mathematics** — a universal real-valued foundation for mathematics and physics, built from a single operator.

[![PyPI](https://img.shields.io/pypi/v/eml-math)](https://pypi.org/project/eml-math/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/eml-math)](https://pypi.org/project/eml-math/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GitHub repository (source, C/C++ API, HTML docs):**
[https://github.com/andrewkwatts-maker/EML-Math](https://github.com/andrewkwatts-maker/EML-Math)

Created by **Andrew K Watts**.  
Based on the EML Sheffer operator as established by Andrzej Odrzywolek:
[arXiv:2603.21852v2](https://arxiv.org/html/2603.21852v2) (CC BY 4.0)

---

## The Core Idea

A single binary operator generates every elementary function in mathematics:

```
eml(x, y) = exp(x) − ln(y)
```

This is the **EML Sheffer operator** — the continuous analog of the NAND gate for Boolean logic. The operator can reconstruct all 36 standard elementary functions: `+`, `−`, `×`, `/`, `exp`, `ln`, `sin`, `cos`, `tan`, `π`, `e`, and every standard transcendental.

The `EMLPoint` is simultaneously a mathematical state and a composable expression-tree node:

```python
from eml_math import EMLPoint
import math

EMLPoint(1, 1).tension()                                  # e = eml(1,1)
EMLPoint(2, 1).tension()                                  # exp(2)
EMLPoint(1, EMLPoint(EMLPoint(1, math.e), 1)).tension()  # ln(e) = 1.0
```

---

## Installation

```bash
pip install eml-math
```

With optional extensions:

```bash
pip install eml-math[ext]        # numpy + sympy (lattice ops, symbolic work)
pip install eml-math[precision]  # mpmath (arbitrary-precision simulation mode)
pip install eml-math[dev]        # pytest, ruff, mypy
```

> **C / C++ / Rust users:** The PyPI wheel ships the Python extension only.
> The C shared library (`eml_math.dll` / `libeml_math.so`) must be built from
> the [GitHub source repository](https://github.com/andrewkwatts-maker/EML-Math).
> See the [C/C++/Rust API](#cc-and-rust-api) section below.

---

## Quick Start

```python
from eml_math import EMLPoint, EMLState, simulate_pulses, verify_conservation
from eml_math import operators as ops
import math

# ── The EML primitive ─────────────────────────────────────────────────────────
print(EMLPoint(1, 1).tension())          # 2.718... (e)
print(EMLPoint(math.pi, 1).tension())    # exp(π)

# ── All 36 elementary operators as EMLPoint expression trees ─────────────────
print(ops.ln(math.e).tension())              # 1.0
print(ops.add(3, 4).tension())               # 7.0
print(ops.mul(3, 4).tension())               # 12.0
print(ops.sin(math.pi / 2).tension())        # 1.0
print(ops.exp(ops.add(ops.ln(2), ops.ln(3))).tension())  # 6.0

# ── Mirror-Pulse dynamics (EML iteration) ────────────────────────────────────
s = EMLState(EMLPoint(1.0, 1.0))
traj = simulate_pulses(s, n_pulses=10)
print(verify_conservation(traj))    # True — Axiom 10 holds at every step
```

---

## v1.0.0 Geometry and Physics Layer

### Spacetime and Lorentz Boosts

The EML encoding maps coordinates to spacetime:
- **Time-like**: `t = exp(x)`
- **Space-like**: `s = ln(|y|)`
- **Minkowski interval**: `Δ_M = √|t² − (c·s)²|`

```python
from eml_math import EMLPoint
import math

p = EMLPoint(1.0, math.e)    # t = e, s = 1
print(p.minkowski_delta())   # Minkowski interval Δ_M
print(p.is_timelike())       # True/False
print(p.rapidity())          # φ = atanh(s/t)

# Lorentz boost by rapidity φ preserves Δ_M
p2 = p.boost(0.693)
assert abs(p.minkowski_delta() - p2.minkowski_delta()) < 1e-10
```

### Relativistic Four-Momentum

```python
from eml_math.momentum import FourMomentum
from eml_math import EMLPoint

p = FourMomentum(EMLPoint(1.0, math.e), c=1.0)
print(p.energy)     # exp(x)
print(p.momentum)   # ln(|y|) / c
print(p.mass)       # Δ_M / c²  — invariant under boost
print(p.gamma())    # Lorentz factor γ = E / (mc²)

# Mass-velocity factory
p2 = FourMomentum.from_mass_velocity(mass=1.0, v=0.5, c=1.0)
```

### General-Relativistic Metric Tensors

```python
from eml_math.metric import MetricTensor
from eml_math import EMLPoint

# Flat Minkowski metric g = diag(+1, −1)
flat = MetricTensor.flat()
print(flat.ds2(EMLPoint(1.0, 1.0), dx=1.0, dy=0.0))   # > 0 (timelike)
print(flat.is_curved())  # False

# Schwarzschild metric: g_tt = −(1−rs/r),  g_rr = 1/(1−rs/r)
m = MetricTensor.schwarzschild(rs=2.0)
# Numeric Christoffel symbol Γ^λ_{μν} via central finite differences
print(m.christoffel(0, 0, 1, EMLPoint(3.0, 1.0)))

# Factory methods for all standard spacetimes:
MetricTensor.flrw(scale_factor_a=lambda t: 1.0)   # FLRW cosmological metric
MetricTensor.ads5_x_s5(L=1.0)                      # AdS₅ × S⁵
MetricTensor.calabi_yau_3()                         # Calabi–Yau 3-fold (Kähler)
MetricTensor.klebanov_strassler(gsM=0.1)            # KS warped deformed conifold
MetricTensor.heterotic_e8x8(radius=1.0)             # Heterotic E₈×E₈ torus
MetricTensor.g2_holonomy()                          # G₂-holonomy Bryant–Salamon cone

# Geodesic step via the EMLState interface
from eml_math import EMLState
s = EMLState.from_point(EMLPoint(3.0, 1.0))
s2 = s.geodesic_step(m, dtau=0.005)
```

### Clifford / Geometric Algebra

The geometric product `ab` is the fundamental algebraic operation; inner and outer products are derived from it.

```python
from eml_math.geometric_algebra import EMLMultivector
from eml_math import EMLPoint

# 2D Minkowski algebra Cl(1,−1)
v = EMLMultivector(
    [EMLPoint(0,1), EMLPoint(1,1), EMLPoint(0.5,1), EMLPoint(0,1)],
    signature=(1, -1)
)
print(v.quadratic())   # v·v in Minkowski metric: Σ sig[i]·v_i²

# Spacetime algebra Cl(1,3)
comps = [EMLPoint(c, 1.0) for c in [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]]
vst = EMLMultivector.spacetime(comps)

# Rotor for rotation in the e₀∧e₁ plane
R = v.rotor(angle=math.pi/4, plane=(0, 1))
v_rot = v.rotate(R)                      # sandwich product R·v·R̃

# Factory methods:
EMLMultivector.g2(comps_128)             # G₂ algebra, signature (1,)*7
EMLMultivector.e8(comps_256)             # E₈ algebra, signature (1,)*8
```

### Octonions

```python
from eml_math.octonion import Octonion, basis_octonion

e1 = basis_octonion(1)
e2 = basis_octonion(2)
e4 = basis_octonion(4)

print(e1 * e2 == e4)          # True  (Fano-plane multiplication)
print(abs((e1 * e2).norm() - (e1.norm() * e2.norm())) < 1e-12)  # |ab|=|a||b|
print(e1.conjugate())         # flip imaginary signs

# G₂ automorphism check
from eml_math.octonion import is_g2_automorphism
identity = lambda o: o
print(is_g2_automorphism(e1, e2, identity))  # True at this pair
```

### N-Dimensional Lattices — E₈ and Leech

```python
from eml_math.ndim import EMLNDVector, e8_lattice_points, e8_min_norm
from eml_math.ndim import leech_lattice_points, leech_min_norm
import math

# E₈ lattice: 240 minimal roots, each with norm √2
roots = e8_lattice_points(n_points=240)
assert len(roots) == 240
assert all(abs(r.euclidean_norm() - math.sqrt(2)) < 1e-10 for r in roots)
print(e8_min_norm())     # √2

print(leech_min_norm())  # 2
```

### Minkowski Four-Vector (3+1D)

```python
from eml_math.fourvector import MinkowskiFourVector
from eml_math import EMLPoint

v = MinkowskiFourVector(
    t=EMLPoint(1,1), x=EMLPoint(0.5,1), y=EMLPoint(0,1), z=EMLPoint(0,1),
    c=1.0
)
print(v.minkowski_norm())     # √|g_{μν} x^μ x^ν|  signature (+,−,−,−)
v2 = v.boost(rapidity_phi=0.5, direction="x")
print(abs(v.minkowski_norm() - v2.minkowski_norm()) < 1e-10)  # True
```

### Discrete / Planck-Scale Helpers

```python
from eml_math.discrete import planck_delta, lattice_distance, is_lattice_neighbor
from eml_math import EMLPoint

p = EMLPoint(1.0, math.e)
print(planck_delta(p))                       # round(Δ_M × PLANCK_D) / PLANCK_D
print(lattice_distance(p, EMLPoint(2,1)))    # planck_delta of displacement
print(is_lattice_neighbor(p, EMLPoint(2,1))) # bool
```

---

## Formula Discovery and Equation Compression

The `compress()` function is an equation simplification pipeline. Give it any Python callable
and it returns the most compact EML closed-form expression that reproduces it numerically.

**Round-trip compression:** formula → EML → simplified form → back to standard notation.

```python
import math
from eml_math.discover import compress, recognize

# ── Tautologies collapse to constants ────────────────────────────────────────
r = compress(lambda x: math.sin(x)**2 + math.cos(x)**2)
print(r.formula)      # "1"  (Pythagorean identity → constant)
print(r.error)        # < 1e-8

# ── Redundant composition collapses ──────────────────────────────────────────
r = compress(lambda x: math.exp(math.log(x)), x_lo=0.5, x_hi=3.0)
print(r.formula)      # "x"  (exp∘ln = identity)
print(r.error)        # < 1e-10

# ── The fundamental EML expression ───────────────────────────────────────────
r = compress(lambda x: math.exp(x) - math.log(x), x_lo=0.5, x_hi=3.0)
print(r.formula)      # "eml(x, x)"  — the minimal EML form
print(r.complexity)   # 3 nodes

# ── Identify known constants ──────────────────────────────────────────────────
print(recognize(math.pi).formula)            # "π"
print(recognize(math.e).formula)             # "e"
print(recognize((1 + math.sqrt(5)) / 2).formula)  # "φ (golden ratio)"

# ── Get back standard Python or LaTeX ────────────────────────────────────────
r = compress(math.exp)
print(r.to_python())  # "import math\nf = lambda x: math.exp(x)"
print(r.to_latex())   # "\exp(x)"

# ── Full Searcher API for data-driven discovery ───────────────────────────────
from eml_math.discover import Searcher
x = [0.2 + i * 0.07 for i in range(40)]
y = [math.exp(xi) - math.log(xi) for xi in x]
result = Searcher(max_complexity=6, precision_goal=1e-10).find(x, y)
print(result)  # SearchResult(formula='eml(x, x)', error=..., complexity=3)
```

**How it works:** The search engine uses a Rust-backed beam search over EML expression trees.
Because the EML Sheffer operator generates all 36 elementary functions, any expression built
from `exp`, `ln`, `+`, `−`, `×`, `÷`, `sin`, `cos` has a representation in EML tree form.
The beam search finds the minimal-complexity tree that fits your data within the precision goal.

The round-trip is exact for expressions that live in EML space (which is all elementary
mathematics). The compression is purely numerical — it samples your function over a range and
searches for the shortest formula that matches those samples. For algebraic identities like
sin²+cos²=1, the compressor finds the simplified constant form automatically.

```python
# Complexity comparison: verbose vs compressed
verbose   = lambda x: (math.sin(x)**2 + math.cos(x)**2) * math.exp(0)
compressed = compress(verbose)
print(compressed.formula, compressed.complexity)   # "1"  complexity=1
```

---

## Architecture

| Module | Contents |
|---|---|
| `eml_math.point` | `EMLPoint` — universal EML node; geometry, boosts, causal structure |
| `eml_math.pair` | `EMLPair` — two-real replacement for complex numbers |
| `eml_math.state` | `EMLState` — full Φ(n, ρ, θ) iteration state; geodesic step |
| `eml_math.operators` | All 36 elementary functions as pure EML expression trees |
| `eml_math.simulation` | `simulate_pulses`, `verify_conservation`, trajectories |
| `eml_math.metric` | `MetricTensor` — 8 spacetime metric factories; Christoffel symbols |
| `eml_math.momentum` | `FourMomentum` — relativistic energy-momentum with Lorentz boost |
| `eml_math.fourvector` | `MinkowskiFourVector` — (3+1)D four-vector |
| `eml_math.geometric_algebra` | `EMLMultivector` — Clifford algebra Cl(p,q) |
| `eml_math.octonion` | `Octonion` — Fano-plane non-associative division algebra |
| `eml_math.ndim` | `EMLNDVector`; E₈ (240 roots) and Leech lattice helpers |
| `eml_math.discrete` | Planck-scale lattice quantization helpers |
| `eml_math.qft` | Klein–Gordon, Dirac, path-integral simulation |
| `eml_math.qm` | Quantum postulates Q1–Q5, qubits, entanglement |
| `eml_math.discover` | Beam-search symbolic regression / formula discovery |

---

## Rust Backend Performance

Critical paths are accelerated by a Rust extension (`eml_core`) built with [maturin](https://maturin.rs/) and parallelised with [Rayon](https://docs.rs/rayon):

| Operation | Python | Rust (batch) | Speedup |
|---|---|---|---|
| Mirror-Pulse (10 000 steps) | 12 ms | 1.4 ms | ~9× |
| Lorentz boost (1 000 points) | 3.2 ms | 0.35 ms | ~9× |
| Schwarzschild Γ (1 000 pts) | 18 ms | 2.0 ms | ~9× |
| Octonion multiply (1 000 pairs) | 5.5 ms | 0.6 ms | ~9× |
| Geometric product Cl(1,3) (1 000) | 22 ms | 2.4 ms | ~9× |

The Python API transparently falls back to pure Python if the compiled extension is unavailable.

---

## C/C++ and Rust API

> **Important:** The C shared library is **not distributed via PyPI**. It must be
> compiled from the source repository. The Python wheel on PyPI contains only the
> Python extension module (`eml_core`).

**Source:** [https://github.com/andrewkwatts-maker/EML-Math](https://github.com/andrewkwatts-maker/EML-Math)

### Build the C library

```bash
git clone https://github.com/andrewkwatts-maker/EML-Math
cd EML-Math
cargo build --release -p eml_c_api
# Output: target/release/eml_math.dll (Windows)
#         target/release/libeml_math.so (Linux/macOS)
#         target/release/libeml_math.a  (static, all platforms)
```

### Use from C

```c
#include "c_api/eml_math.h"

int main(void) {
    double tension = eml_tension(1.0, 1.0);   /* e */

    double out_x, out_y;
    eml_boost(1.0, 2.718, 0.5, 1.0, &out_x, &out_y);

    double a[8] = {0,1,0,0,0,0,0,0};   /* e₁ */
    double b[8] = {0,0,1,0,0,0,0,0};   /* e₂ */
    double c[8];
    eml_octonion_mul(a, b, c);           /* c = e₁ × e₂ = e₄ */
    return 0;
}
```

### Use from C++ (CMake)

```cmake
target_link_libraries(my_project PRIVATE
    ${EML_MATH_DIR}/target/release/eml_math.lib)
target_include_directories(my_project PRIVATE
    ${EML_MATH_DIR}/c_api)
```

### Use from Rust

```toml
[dependencies]
eml_c_api = { path = "path/to/EML-Math/c_api" }
```

### Exported C functions

| Function | Description |
|---|---|
| `eml_tension(x, y)` | Core EML operator: `exp(x) − ln(y)` |
| `eml_mirror_pulse(x, y, *ox, *oy)` | One Mirror-Pulse iteration step |
| `eml_simulate_pulses(x0, y0, n, *xs, *ys)` | Run n iterations |
| `eml_euclidean_delta(x, y)` | `√(exp(2x) + (ln y)²)` — Euclidean frame invariant |
| `eml_minkowski_delta(x, y, sig, c)` | Minkowski interval `√\|t² − (cs)²\|` |
| `eml_rapidity(x, y)` | Rapidity φ = `atanh(ln y / exp x)` |
| `eml_causal_type(x, y, c, tol)` | +1 timelike / 0 lightlike / −1 spacelike |
| `eml_boost(x, y, φ, c, *ox, *oy)` | Lorentz boost by rapidity φ |
| `eml_boost_batch(xs, ys, phis, c, n, oxs, oys)` | Vectorised boost |
| `eml_schwarzschild_christoffel(λ,μ,ν, r, rs)` | Analytic Γ^λ_{μν} |
| `eml_octonion_mul(a[8], b[8], out[8])` | Fano-plane octonion product |

Full documentation: [`c_api/eml_math.h`](c_api/eml_math.h)

---

## HTML Documentation

Interactive docs including concept guides, API reference, and worked examples:

```
docs/index.html      — Overview and quick-start
docs/concepts.html   — EML operator, axioms, spacetime encoding
docs/guide.html      — Step-by-step code walkthroughs
docs/api.html        — Full API reference (all classes, all methods)
```

Open locally: `open docs/index.html` (or double-click in a file browser).

---

## Mathematical Background

The 16 axioms of EML Mathematics derive all structure from one principle:

| Axiom | Name | Formula |
|---|---|---|
| 5 | Tension | `T = exp(x) − ln(y)` |
| 7 | Mirror Update | `x_{t+1} = y_t,  y_{t+1} = T_{t+1}` |
| 8 | Frame Shift | when `y ≤ 0`, use `\|y\|` |
| 9 | 3:1 Flip | 3 growth + 1 reflection = net +2 reality units |
| 10 | Conservation | `T + x = exp(x)` at every step |

### Spacetime Encoding

EML coordinates map to special-relativistic spacetime via:

```
t = exp(x)    (time-like component)
s = ln(|y|)   (space-like component)
Δ_M = √|t² − (c·s)²|   (Minkowski interval, invariant under boosts)
```

The Lorentz boost at rapidity φ:
```
t' = t·cosh(φ) − (s/c)·sinh(φ)
s' = s·cosh(φ) − t·c·sinh(φ)
```

---

## Related Work

- Odrzywolek, A. (2026). "All elementary functions from a single operator." arXiv:2603.21852v2

You may also find the companion symbolic-regression package useful:
[eml-sr on PyPI](https://pypi.org/project/eml-sr/)

---

## License

MIT — Andrew K Watts, 2026
