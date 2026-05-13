# Changelog

## [1.4.0] — 2026-05-10

EML / Arithmos / metaphysica / periodica synchronised v1.4.0 cut. Adds an
optional Arithmos symbolic-substrate bridge for engine consumers; the
PyPI install path is unchanged.

### Added

- **`with-arithmos` Cargo feature** (off by default) — pulls in
  `arithmos_core` as an optional dependency via a `git-submodule`-only
  path so engine consumers can carry an `ArithmosExpression` payload
  alongside the native `EMLPoint` / RPN tree. Strictly absent from the
  PyPI dep tree — `pip install eml-math` is unaffected.
- **`src/arithmos_bridge.rs`** — converters + `ArithmosPayload` trait
  contract. Skeleton only for v1.4.0; bodies populate as the Arithmos
  surface stabilises.

### Changed

- `crate-type = ["cdylib", "rlib"]` — adds `rlib` so engine consumers
  can link `eml_core` directly without the PyO3 GIL serialisation
  penalty. The `cdylib` line still produces the maturin-built Python
  extension as before.
- Crate version bumped from `1.0.0` → `1.4.0` so it matches the
  `pyproject.toml` PyPI version line for downstream sanity-check
  tooling that compares both.

### Notes

- No functional regressions; existing PyO3 attributes (`#[pyclass]`,
  `#[pymethods]`, `#[pyo3(get)]`, `#[new]`) are untouched. A future
  release may introduce a dual-impl pattern that decouples the Rust
  inherent surface from PyO3, but the v1.4.0 line keeps PyO3 on the
  default-build path.
- `default = ["python"]` — a `--no-default-features` build of the
  Rust crate does not currently compile because the source still uses
  bare `#[pyclass]` / `#[pymethods]` attributes. Engine consumers do
  not hit that path; they enable the bridge via their own
  `with-eml` feature flag (which inherits the default features).
