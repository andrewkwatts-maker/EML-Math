from __future__ import annotations


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
        return (
            self.formula
            .replace("eml(", r"\mathrm{eml}(")
            .replace("exp(", r"\exp(")
            .replace("ln(", r"\ln(")
            .replace("sqrt(", r"\sqrt{")
            .replace("sin(", r"\sin(")
            .replace("cos(", r"\cos(")
            .replace("tan(", r"\tan(")
            .replace("pi", r"\pi")
        )

    def to_python(self) -> str:
        expr = (
            self.formula
            .replace("exp(", "math.exp(")
            .replace("ln(", "math.log(")
            .replace("sqrt(", "math.sqrt(")
            .replace("sin(", "math.sin(")
            .replace("cos(", "math.cos(")
        )
        return f"import math\nf = lambda x: {expr}"

    def __repr__(self) -> str:
        return (
            f"SearchResult(formula={self.formula!r}, "
            f"error={self.error:.2e}, complexity={self.complexity})"
        )
