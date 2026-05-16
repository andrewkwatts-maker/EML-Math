#====== EML-Math/src/eml_math/_dispatch.py ======#
#!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
#!
#!This is the intellectual property of Andrew Keith Watts. Unauthorized
#!reproduction, distribution, or modification of this code, in whole or in part,
#!without the express written permission of Andrew Keith Watts is strictly prohibited.
#!
#!For inquiries, please contact AndrewKWatts@Gmail.com

# Rust impl: rust/eml_core/src/lib.rs
"""Dispatch helpers: route public API functions to the Rust backend when available.

Every public function decorated with ``@rust_accelerated("fn_name")`` calls the
named function from ``eml_math.eml_core`` when the Rust wheel is present,
falling back transparently to the Python body otherwise.
"""
from __future__ import annotations

import functools

_native = None
_HAS_RUST: bool = False
try:
    from eml_math import eml_core as _native  # type: ignore[import-not-found]
    _HAS_RUST = True
except ImportError:
    pass


def rust_accelerated(rust_fn_name: str):
    """Dispatch to Rust backend fn by name; fall back to Python impl silently.

    Usage::

        @rust_accelerated("tension_n")
        def batch_tension(xs, ys):
            # Python fallback body — only runs when Rust is unavailable.
            ...
    """
    def decorator(py_fn):
        @functools.wraps(py_fn)
        def wrapper(*args, **kwargs):
            if _HAS_RUST and _native is not None:
                fn = getattr(_native, rust_fn_name, None)
                if fn is not None:
                    return fn(*args, **kwargs)
            return py_fn(*args, **kwargs)
        return wrapper
    return decorator
