# Changelog

---

## [2.3.0] - 2026-08-22

### Fixed — numeric core

- **`ln()` was catastrophically wrong below the normal range.** The depth-3
  chain evaluates `exp(e - ln x)` internally; once `x < ~1.1e-307` that exceeds
  `DBL_MAX`, the Slipping-Wheel clamp rewrites the exponent, and the outer
  cancellation no longer holds:

      ops.ln(1e-320)        ->  -3.8878     (correct: -736.827)
      ops.mul(1e-320, 1e10) ->   2.05e8     (correct:  9.9999e-311)

  `_LnChainNode` evaluates the analytically-equal closed form `e - t` whenever
  `exp(t)` is unrepresentable, keeping identical children/`repr`/`is_leaf` so
  tree walkers are unaffected. Normal range is bit-identical; the subnormal
  result now matches plain float arithmetic exactly.

- **Rust and Python numeric cores had drifted**, so the same `EMLPoint` gave
  different answers depending on whether it evaluated as a leaf (Rust) or from
  `_LitNode`s (Python). `OVERFLOW_THRESHOLD` was the rounded literal `709.78`
  rather than `f64::MAX.ln() = 709.782712893384`, and the frame-shift guard
  used `abs(y).max(1e-300)`, crushing every negative subnormal instead of
  flooring only exact zero. 6 of 10 probe cases diverged before; 0 after.
  Requires a rebuild of the Rust extension.

- **Unbounded trial division in `is_prime_tension`.** The sympy-less fallback
  had no iteration cap; at the module's own documented scale `D ~ 6.19e34` it
  needs 1.24e17 iterations — roughly 395 years, uninterruptible. Now capped at
  `n <= 1e18` with a clear error. Also guards `round()` against inf/NaN.

- `_fmt_num` crashed on non-finite literals: `v == int(v)` is evaluated before
  the magnitude guard, and the parser only catches `SyntaxError`.
- The sign-aware `ops` shim skipped `inv`, so `ops.div(1, -2)` and
  `ops.inv(-2)` disagreed on sign within one class.
- `compress.py` labelled `cosh_1` as `"sinh(1)"` — the value was right, the
  rendered LaTeX was wrong and indistinguishable from `sinh_1`.

### Changed

- Stopped tracking a compiled `.pyc`. A stale one whose `(mtime, size)` matched
  a fresh source edit was served instead of the edit, producing a phantom test
  failure that survived a correct fix.

### Tests

- 2199 -> 2249 passing (50 new edge-case tests covering subnormals, exact
  zeros, non-finite literals, and Rust/Python parity).

### Documented, not changed — these are by design

- Sign loss in `mul`/`div`/`pow`/`sqrt` for negatives (`ops.mul(-3, 4) = +12`)
  follows from the documented frame-shift guard; `_SignedOps` is the intended
  escape hatch, which is why the `inv` hole in it *was* fixed.
- Slipping Wheel through the operator layer (`ops.exp(1000) = 1000.0`) is
  documented behaviour, load-bearing for `iterate()`'s orbit bounding.
- Zero does not survive ln-based ops (`mul(0, 5) = 5e-300`). Worth a design
  decision, not silently altered.

---

## [2.0.1] — 2026-05-17

### Added

- **`eml-math-app` CLI command** — companion-app launcher installed as a console script
  alongside the library. On first run it locates or clones the
  [EML-Math-App](https://github.com/andrewkwatts-maker/EML-Math-App) KivyMD desktop/Android
  explorer at the matching version tag (`v2.0.1`) and launches it. Subsequent runs go straight
  to launch. Developer checkouts are detected automatically via sibling-directory search from
  `__file__`; end-user installs clone to `~/.eml-math-app`.

### Fixed

- **CI:** Removed stale `arithmos_core` path dependency from `rust/eml_core/Cargo.toml`.
  The path only exists in the local engine workspace; its presence caused all PyPI CI builds to
  fail at cargo manifest load time, even when the feature was disabled.
- **Dispatch:** `_dispatch.py` fallback path now correctly routes to pure Python when the
  Rust extension is absent at runtime.

---

## [2.0.0] — 2026-05-14

### Added

- **`src/eml_math/_dispatch.py`** — Runtime dispatch layer. Exposes `_HAS_RUST: bool` and a
  `rust_accelerated(fn_name)` decorator that transparently routes calls to the Rust extension
  and silently falls back to the Python implementation when the extension is not built.

- **`EMLPoint.tension()` — Rust fast path.** Leaf nodes in continuous mode (`D=None`) are now
  dispatched to `eml_core::EMLPoint::tension()` via the Rust extension. Discrete-mode points
  (`D` is set) continue to use the pure-Python path, which includes quantisation logic absent
  from the Rust implementation.

- **`EMLPoint.iterate()` — Rust fast path.** Same dispatch pattern; routes to
  `eml_core::EMLPoint::mirror_pulse()` and reconstructs a Python `EMLPoint` from the result.

- **`src/eml_math/tree.py`** — Expression tree module:
  - `normalize_input()` — normalises raw Python expressions before parsing
  - `tree_to_python()` — reconstructs a Python expression string from a parsed tree
  - Full `ast.BinOp` and `ast.UnaryOp(UAdd)` parser support

- **`tests/test_rust_python_parity.py`** — Rust/Python parity suite:
  - 100-point randomised `tension()` parity at ±1e-12 tolerance
  - 10-step `iterate()` chain parity at ±1e-12 tolerance
  - Batch operation parity: `tension_n`, `exp_n`, `ln_n`

- **`tests/test_input_helpers.py`** — 124 assertions covering input normalisation helpers.

- **`tests/test_tree_binop.py`** — 147 assertions covering binary-op tree parsing and
  round-trip reconstruction.

### Changed

- Crate version bumped from `1.4.x` → `2.0.0` to align with the PyPI package version.

---

## [1.4.1] — 2026-05-12

### Changed

- Renamed `rust/eml_core/src/arithma_bridge.rs` → `arithmos_bridge.rs` to align with the
  upstream crate rename (`arithma_core` → `arithmos_core`).
- Updated `[features]` dep reference from `arithma_core` → `arithmos_core`.
- `Cargo.lock` regenerated against `arithmos_core 1.4.1`.

---

## [1.4.0] — 2026-05-10

### Added

- **`with-arithmos` Cargo feature** *(off by default)* — optional `arithmos_core` path
  dependency for engine consumers. Allows `eml_core` to carry an `ArithmosExpression` payload
  alongside the native `EMLPoint` / RPN tree. Strictly absent from the PyPI dependency tree —
  `pip install eml-math` is unaffected.
- **`rust/eml_core/src/arithmos_bridge.rs`** — `ArithmosPayload` trait + converter skeleton.
  Bodies populate as the Arithmos public surface stabilises.

### Changed

- `crate-type` extended to `["cdylib", "rlib"]` — adds `rlib` so engine consumers can link
  `eml_core` directly without the PyO3 GIL serialisation penalty. The `cdylib` output
  (maturin-built Python extension) is unchanged.
- Crate version bumped `1.0.0` → `1.4.0` to match `pyproject.toml` for downstream
  version-parity tooling.

### Notes

- No functional regressions. Existing PyO3 attributes (`#[pyclass]`, `#[pymethods]`,
  `#[pyo3(get)]`, `#[new]`) are untouched.
- A `--no-default-features` Rust build does not yet compile cleanly; engine consumers
  enable the bridge via their own `with-eml` feature flag, which inherits the defaults.

---

## [1.3.0] — 2026-05-03

- Render pipeline, datasheet `Get()` API, 136 named constants, 2092 tests, CI green.
- PNG/PDF rendering tests skip gracefully when Pillow is not installed.
