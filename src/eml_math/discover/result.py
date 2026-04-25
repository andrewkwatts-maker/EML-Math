from __future__ import annotations

import re


class SearchResult:
    """Result of a formula discovery search."""

    __slots__ = ("formula", "error", "complexity", "params")

    def __init__(
        self,
        formula: str,
        error: float,
        complexity: int,
        params: list[float],
    ) -> None:
        self.formula = formula
        self.error = error
        self.complexity = complexity
        self.params = params

    def to_latex(self) -> str:
        """Convert formula to LaTeX. Output is ready for $ $, \\( \\), or MathJax."""
        # Use regex word-boundary replacements to avoid partial matches
        s = self.formula
        s = re.sub(r'\beml\(', r'\\mathrm{eml}(', s)
        s = re.sub(r'\bexp\(', r'\\exp(', s)
        s = re.sub(r'\bln\(', r'\\ln(', s)
        s = re.sub(r'\bsqrt\(', r'\\sqrt{', s)
        s = re.sub(r'\bsin\(', r'\\sin(', s)
        s = re.sub(r'\bcos\(', r'\\cos(', s)
        s = re.sub(r'\btan\(', r'\\tan(', s)
        s = re.sub(r'\bsinh\(', r'\\sinh(', s)
        s = re.sub(r'\bcosh\(', r'\\cosh(', s)
        s = re.sub(r'\btanh\(', r'\\tanh(', s)
        s = re.sub(r'\barcsin\(', r'\\arcsin(', s)
        s = re.sub(r'\barccos\(', r'\\arccos(', s)
        s = re.sub(r'\barctan\(', r'\\arctan(', s)
        s = re.sub(r'\bpi\b', r'\\pi', s)
        s = re.sub(r'\binf\b', r'\\infty', s)
        s = re.sub(r'∞', r'\\infty', s)
        s = re.sub(r'π', r'\\pi', s)
        return s

    def to_mathjax(self) -> str:
        """Return LaTeX wrapped in MathJax inline delimiters \\( ... \\)."""
        return r'\(' + self.to_latex() + r'\)'

    def to_mathml(self) -> str:
        """Convert formula to MathML markup (inline <math> element)."""
        from eml_math.discover.compress import _formula_to_mathml
        return _formula_to_mathml(self.formula)

    def to_python(self) -> str:
        """Return runnable Python with import math header."""
        s = self.formula
        s = re.sub(r'\beml\(([^,]+),\s*([^)]+)\)', r'(math.exp(\1) - math.log(\2))', s)
        s = re.sub(r'\bexp\(', 'math.exp(', s)
        s = re.sub(r'\bln\(', 'math.log(', s)
        s = re.sub(r'\bsqrt\(', 'math.sqrt(', s)
        s = re.sub(r'\bsin\(', 'math.sin(', s)
        s = re.sub(r'\bcos\(', 'math.cos(', s)
        s = re.sub(r'\btan\(', 'math.tan(', s)
        s = re.sub(r'\bsinh\(', 'math.sinh(', s)
        s = re.sub(r'\bcosh\(', 'math.cosh(', s)
        s = re.sub(r'\btanh\(', 'math.tanh(', s)
        s = re.sub(r'\bpi\b', 'math.pi', s)
        s = re.sub(r'\binf\b', 'math.inf', s)
        return f"import math\nf = lambda x: {s}"

    def __repr__(self) -> str:
        return (
            f"SearchResult(formula={self.formula!r}, "
            f"error={self.error:.2e}, complexity={self.complexity})"
        )
