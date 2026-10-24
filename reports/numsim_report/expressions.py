"""Expressions of a report document and the checks built from them."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import CaseData


class Expressions:
    """Evaluates the document's expressions: strings are Python over numpy (`np`), the parameters,
    the derived values and, given a case, its columns; numbers pass through.

    Documents are trusted input: `np` alone gives an expression access to the file system."""

    BUILTINS = {"abs": abs, "max": max, "min": min}

    def __init__(self, spec: dict, cases: dict[str, CaseData]):
        self.spec = spec
        self.cases = cases
        self.scope = {"np": np, **spec.get("parameters", {})}
        for name, expr in spec.get("derived", {}).items():
            self.scope[name] = self(expr)

    def __call__(self, expr, case: str | None = None):
        if not isinstance(expr, str):
            return expr
        scope = self.scope if case is None else {**self.scope, **self.cases[case].columns}
        return eval(expr, {"__builtins__": self.BUILTINS}, scope)

    def case_names(self, item: dict) -> list[str]:
        """Cases an item applies to: "case", "cases" (a list or "*"), default all; missing ones dropped."""
        names = item.get("cases", [item["case"]] if "case" in item else "*")
        if names == "*":
            names = [c["name"] for c in self.spec["cases"]]
        return [n for n in names if n in self.cases]


@dataclass
class CheckResult:
    name: str
    case: str
    error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.error <= self.tolerance


def evaluate_checks(expr: Expressions) -> list[CheckResult]:
    """Per check and case: max |value − expected| over the loaded steps, relative to max |expected| if asked."""
    results = []
    for check in expr.spec.get("checks", []):
        for case in expr.case_names(check):
            step = expr.cases[case].columns["step"]
            mask = step > 0
            if "where" in check:
                mask &= np.broadcast_to(np.asarray(expr(check["where"], case), bool), step.shape)
            value = np.broadcast_to(np.asarray(expr(check["value"], case), float), step.shape)[mask]
            expected = np.broadcast_to(np.asarray(expr(check.get("expected", 0), case), float), step.shape)[mask]
            error = float(np.max(np.abs(value - expected), initial=0.0))
            if check.get("relative", False):
                error /= max(float(np.max(np.abs(expected), initial=0.0)), 1e-300)
            results.append(CheckResult(check["name"], case, error if np.isfinite(error) else math.inf,
                                       float(check["tolerance"])))
    return results
