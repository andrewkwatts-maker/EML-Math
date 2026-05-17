# Changelog

---

## [2.0.1] — 2026-05-17

### Fixed

- **CI:** Removed stale `arithmos_core` path dependency from `rust/eml_core/Cargo.toml`.
  The path only exists in the local engine workspace; its presence caused all PyPI CI builds to fail
  at cargo manifest load time, even when the feature was disabled.
- **Dispatch:** `_dispatch.py` fallback path now correctly routes to pure Python when
  `_arithma_core` is absent at runtime.

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
