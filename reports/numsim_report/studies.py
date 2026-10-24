"""Step-size studies: a case run with several step counts, the error of its final state against the finest run."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np

from .data import CaseData
from .expressions import Expressions

EXACT = 1e-9  # relative error below which a scheme counts as exact: ten times the driver tolerance


@dataclass
class StudyRow:
    steps: int
    increment: float  # fraction of a path segment per step, 1 / steps
    error: float      # |quantity − quantity of the finest run|


@dataclass
class StudyResult:
    name: str
    quantity: str
    reference_steps: int
    rows: list[StudyRow]
    order: float
    exact: bool
    min_order: float

    @property
    def passed(self) -> bool:
        return self.exact or self.order >= self.min_order


def fit_order(steps: list[int], errors: list[float]) -> float:
    """Slope of log(error) against log(increment) by least squares."""
    x, y = np.log(1.0 / np.asarray(steps, float)), np.log(np.asarray(errors, float))
    return float(np.polyfit(x, y, 1)[0]) if len(steps) > 1 else math.nan


def _final(expr: Expressions, run: CaseData, quantity: str) -> float:
    return float(np.broadcast_to(np.asarray(expr_with(expr, run)(quantity), float), run.columns["step"].shape)[-1])


def expr_with(expr: Expressions, run: CaseData):
    """The document's expressions evaluated over one study run."""
    def evaluate(e):
        return e if not isinstance(e, str) else eval(e, {"__builtins__": Expressions.BUILTINS}, {**expr.scope, **run.columns})
    return evaluate


def evaluate_studies(expr: Expressions, runs: dict[str, dict[int, CaseData]]) -> list[StudyResult]:
    results = []
    for study in expr.spec.get("studies", []):
        by_steps = runs.get(study["name"], {})
        if len(by_steps) < 2:
            continue
        quantity = study.get("quantity", "stress_11")
        finest = max(by_steps)
        reference = _final(expr, by_steps[finest], quantity)
        rows = [StudyRow(n, 1.0 / n, abs(_final(expr, by_steps[n], quantity) - reference)) for n in sorted(by_steps) if n != finest]
        exact = all(r.error <= EXACT * max(abs(reference), 1e-300) for r in rows)
        order = math.nan if exact else fit_order([r.steps for r in rows], [max(r.error, 1e-300) for r in rows])
        results.append(StudyResult(study["name"], quantity, finest, rows, order, exact, float(study.get("min_order", 0.9))))
    return results


def render_study_figure(res: StudyResult) -> bytes:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.6))
    try:
        h = [r.increment for r in res.rows]
        e = [max(r.error, 1e-18) for r in res.rows]
        ax.loglog(h, e, "o-", label=f"error of {res.quantity}")
        if not res.exact and math.isfinite(res.order):
            ax.loglog(h, [e[-1] * (hi / h[-1]) ** round(res.order) for hi in h], "k--", lw=1, label=f"slope {round(res.order)}")
        ax.set(xlabel="segment fraction per step", ylabel=f"|{res.quantity} − reference|",
               title=f"{res.name}: order {res.order:.2f}" if not res.exact else f"{res.name}: exact on this path")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        fig.tight_layout()
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=100)
        return png.getvalue()
    finally:
        plt.close(fig)
