"""What material_report produces, loaded into arrays."""
from __future__ import annotations

import csv
import json
import pathlib
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np

COMPONENTS = ("11", "22", "33", "12", "13", "23")
SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@dataclass
class CaseData:
    name: str
    columns: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]  # step, iteration, residual


@dataclass
class RunOutput:
    spec: dict        # the resolved report document
    graph: list       # materials with parameters, produced properties and inputs
    cases: dict[str, CaseData]
    converged: bool
    studies: dict[str, dict[int, CaseData]] = field(default_factory=dict)  # study -> steps -> run


class ToolError(RuntimeError):
    """material_report rejected the input."""


def read_csv(path: pathlib.Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def with_invariants(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Adds p = tr(σ)/3 and the von Mises stress q."""
    s = {c: columns["stress_" + c] for c in COMPONENTS}
    columns["p"] = (s["11"] + s["22"] + s["33"]) / 3
    columns["q"] = np.sqrt(0.5 * ((s["11"] - s["22"]) ** 2 + (s["22"] - s["33"]) ** 2 + (s["33"] - s["11"]) ** 2)
                           + 3 * (s["12"] ** 2 + s["13"] ** 2 + s["23"] ** 2))
    return columns


def pretty_label(column: str) -> str:
    """strain_11 -> ε₁₁, stress_12 -> σ₁₂; anything else unchanged."""
    m = re.fullmatch(r"(strain|stress)_(\d\d)", column)
    return {"strain": "ε", "stress": "σ"}[m[1]] + m[2].translate(SUBSCRIPTS) if m else column


def run_tool(tool: str, report: pathlib.Path) -> RunOutput:
    """Runs material_report on `report` and loads its output."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp)
        code = subprocess.run([tool, report, out]).returncode
        if code == 2:
            raise ToolError(f"{report}: material_report rejected the document")
        spec = json.loads((out / "resolved.json").read_text())
        def load(stem: str) -> CaseData | None:
            if not (out / f"{stem}.csv").exists():
                return None
            return CaseData(stem, with_invariants(read_csv(out / f"{stem}.csv")), read_csv(out / f"{stem}_iterations.csv"))

        cases = {c["name"]: d for c in spec["cases"] if (d := load(c["name"]))}
        studies = {s["name"]: {n: d for n in s["steps"] if (d := load(f"{s['name']}_{n}"))} for s in spec.get("studies", [])}
        return RunOutput(spec, json.loads((out / "graph.json").read_text()), cases, code == 0, studies)
