# EMLMath

**EML Mathematics (EML-Math)** — a universal real-valued foundation for all of mathematics, built from a single operator.

Created by **Andrew K Watts**.
Based on the work by Andrzej Odrzywołek in the below paper
https://arxiv.org/html/2603.21852v2
which is supplied under https://creativecommons.org/licenses/by/4.0/

you may find that another pypi library I found on the same topic is more useful for your tasks 
https://pypi.org/project/eml-sr/

---

## The Core Idea

A single binary operator generates every elementary function in mathematics:

```
eml(x, y) = exp(x) − ln(y)
```

This is the **EML Sheffer operator** — the continuous analog of NAND for Boolean logic. A peer-reviewed paper (Odrzywolek 2026, arXiv:2603.21852v2) proves that together with the constant `1`, `eml` can reconstruct all 36 elementary functions: `+`, `−`, `×`, `/`, `exp`, `ln`, `sin`, `cos`, `tan`, `π`, `e`, and every standard transcendental.

MPM's tension formula is identical: `T = exp(x) − ln(y)`.

The **EMLPoint** is both the mathematical state and the expression tree node. Nesting EMLPoints *is* the computation:

```python
from eml_math import EMLPoint
import math

EMLPoint(1, 1).tension()                                  # e = eml(1,1)
EMLPoint(2, 1).tension()                                  # exp(2)

# ln(e) as a nested EMLPoint tree — no imports needed
EMLPoint(1, EMLPoint(EMLPoint(1, math.e), 1)).tension()  # 1.0
```

## Installation

```bash
pip install eml_math
```

With optional extensions (sympy for prime tensions, numpy for lattice fields):

```bash
pip install eml-math[ext]
```

## Quick Start

```python
from eml_math import EMLPoint, EMLState, simulate_pulses, verify_conservation
from eml_math import operators as ops
import math

# ── The EML primitive ─────────────────────────────────────────────────────────
print(EMLPoint(1, 1).tension())          # 2.718... (e)
print(EMLPoint(math.pi, 1).tension())    # exp(π)

# ── Operators as EMLPoint trees ──────────────────────────────────────────
print(ops.ln(math.e).tension())              # 1.0
print(ops.add(3, 4).tension())               # 7.0
print(ops.mul(3, 4).tension())               # 12.0
print(ops.sqrt(2).tension())                 # 1.414...
print(ops.sin(math.pi / 2).tension())        # 1.0

# ── Chaining: exp(ln(6)) = 6 ─────────────────────────────────────────────────
print(ops.exp(ops.add(ops.ln(2), ops.ln(3))).tension())  # 6.0

# ── Mirror-Pulse dynamics ─────────────────────────────────────────────────────
knot = EMLState(EMLPoint(1.0, 1.0))
traj = simulate_pulses(knot, n_pulses=10)
print(verify_conservation(traj))    # True — Axiom 10 holds at every step

for k in traj[:5]:
    print(f"n={k.flip_count}  rho={k.rho:.6f}  T={k.point.tension():.6f}")

# ── Discrete mode (opt-in) ────────────────────────────────────────────────────
knot_d = EMLState(EMLPoint(1.0, 1.0, D=100))   # D=100 toy scale
traj_d = simulate_pulses(knot_d, n_pulses=7)
# Reproduces the D=100 table from MPM.txt (lines ~600-643)
```

## The EMLPair — Replacing Complex Numbers

The EML paper notes that `sin`, `cos`, and `π` require complex intermediates. MPM's solution: use two real EMLStates in phase relationship:

```python
from eml_math import EMLPair

# i = EMLPair(real=0, imag=1) — no complex arithmetic
i = EMLPair.unit_i()
print(i)   # EMLPair(0 + 1i)

# Complex multiplication stays real throughout
z1 = EMLPair.from_values(3.0, 4.0)
z2 = EMLPair.from_values(1.0, 2.0)
z3 = z1 * z2
print(z3.real_tension, z3.imag_tension)  # -5.0, 10.0  ✓
print(z1.modulus)                         # 5.0
```

## Architecture

| Module | Contents |
|---|---|
| `eml_math.point` | `EMLPoint` — universal EML node, continuous by default |
| `eml_math.knot` | `EMLState` — full Φ(n, ρ, θ) kinematic entity |
| `eml_math.pair` | `EMLPair` — real replacement for complex numbers |
| `eml_math.operators` | All 36 elementary ops as pure EML EMLPoint nestings |
| `eml_math.simulation` | `simulate_pulses`, `verify_conservation`, trajectories |
| `eml_math.convert` | Bidirectional MPM ↔ traditional notation converter (v0.3.0) |
| `eml_math.qft` | Klein-Gordon, Dirac, Path Integral simulation (v0.4.0) |
| `eml_math.qm` | Quantum postulates Q1-Q5, qubits, entanglement (v0.5.0) |

## Continuous vs Discrete Mode

Continuous (default) — smooth float arithmetic, only frame-shift guard:

```python
p = EMLPoint(1.0, 1.0)          # D=None — continuous
```

Discrete (opt-in) — Planck-scale quantization via `round(T × D)`:

```python
p = EMLPoint(1.0, 1.0, D=100)   # toy discrete
p = EMLPoint(1.0, 1.0, D=6.187e34)  # physical Planck scale
```

## Mathematical Background

The 16 axioms of Mirror Phase Mathematics derive everything from one principle:

- **Axiom 5 (Tension):** `T = exp(x) − ln(y)` — the EML Sheffer operator
- **Axiom 7 (Mirror Update):** `x_{t+1} = y_t,  y_{t+1} = T_{t+1}`
- **Axiom 8 (Frame Shift):** when `y ≤ 0`, use `|y|` — keeps all values real
- **Axiom 9 (3:1 Flip):** 3 growth steps + 1 reflection = net +2 reality units
- **Axiom 10 (Conservation):** `T + x = exp(x)` — holds at every step

Full documentation: [eml_math.readthedocs.io](https://eml_math.readthedocs.io)

## Related Work

- Odrzywolek, A. (2026). "All elementary functions from a single operator." arXiv:2603.21852v2

## License

MIT — Andrew K Watts, 2026
