#!/usr/bin/env python3
"""Build material reports from their report.json: run the cases, evaluate the checks, draw the figures.

usage: make_report.py --tool <material_report> [--check] docs/materials/<material> [...]

--check runs the cases and checks only (no figures, README untouched) and exits 1 if a check fails.
Expressions (figures, references, checks, "derived") see numpy as np, the parameters, the derived
values and the columns of the case as arrays, including p (pressure) and q (von Mises stress).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import pathlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

BEGIN, END = "<!-- report:begin -->", "<!-- report:end -->"
COMPONENTS = ("11", "22", "33", "12", "13", "23")


# --- data ---------------------------------------------------------------------------------------

@dataclass
class CaseData:
    name: str
    columns: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]


@dataclass
class RunOutput:
    spec: dict
    graph: list
    cases: dict[str, CaseData]
    converged: bool


@dataclass
class CheckResult:
    name: str
    case: str
    error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.error <= self.tolerance


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


def run_tool(tool: str, report: pathlib.Path) -> RunOutput:
    """Runs material_report and reads what it wrote."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp)
        code = subprocess.run([tool, report, out]).returncode
        if code == 2:
            raise SystemExit(f"{report}: material_report failed")
        spec = json.loads((out / "resolved.json").read_text())
        cases = {c["name"]: CaseData(c["name"], with_invariants(read_csv(out / f"{c['name']}.csv")),
                                     read_csv(out / f"{c['name']}_iterations.csv"))
                 for c in spec["cases"] if (out / f"{c['name']}.csv").exists()}
        return RunOutput(spec, json.loads((out / "graph.json").read_text()), cases, code == 0)


# --- expressions --------------------------------------------------------------------------------

class Expressions:
    """Evaluates the report's expressions; strings are expressions, numbers pass through."""

    BUILTINS = {"abs": abs, "max": max, "min": min}

    def __init__(self, spec: dict, cases: dict[str, CaseData]):
        self.cases = cases
        self.scope = {"np": np, **spec.get("parameters", {})}
        for name, expr in spec.get("derived", {}).items():
            self.scope[name] = self(expr)

    def __call__(self, expr, case: str | None = None):
        if not isinstance(expr, str):
            return expr
        scope = self.scope if case is None else {**self.scope, **self.cases[case].columns}
        return eval(expr, {"__builtins__": self.BUILTINS}, scope)  # report authors' expressions

    def case_names(self, item: dict, spec: dict) -> list[str]:
        names = item.get("cases", [item["case"]] if "case" in item else "*")
        if names == "*":
            names = [c["name"] for c in spec["cases"]]
        return [n for n in names if n in self.cases]


def evaluate_checks(spec: dict, expr: Expressions) -> list[CheckResult]:
    """max |value − expected| over the loaded steps (relative to max |expected| if asked) per check and case."""
    results = []
    for check in spec.get("checks", []):
        for case in expr.case_names(check, spec):
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


# --- figures ------------------------------------------------------------------------------------

PanelRenderer = Callable[["FigureContext", object, dict, list[str]], None]
PANELS: dict[str, PanelRenderer] = {}


def panel_kind(name: str):
    """Registers a panel renderer under `kind`."""
    def register(fn: PanelRenderer) -> PanelRenderer:
        PANELS[name] = fn
        return fn
    return register


@dataclass
class FigureContext:
    spec: dict
    expr: Expressions
    plt: object = field(repr=False)
    ticker: object = field(repr=False)


@panel_kind("line")
def line_panel(ctx: FigureContext, ax, panel: dict, cases: list[str]) -> None:
    ys = panel["y"] if isinstance(panel["y"], list) else [panel["y"]]
    plot = ax.semilogy if panel.get("logy") else ax.plot
    for n, case in enumerate(cases):
        x = ctx.expr(panel["x"], case)
        for y in ys:
            expr, label = (y["expr"], y.get("label", y["expr"])) if isinstance(y, dict) else (y, y)
            plot(x, ctx.expr(expr, case), label=label if len(cases) == 1 else f"{case}: {label}")
        for ref in panel.get("reference", []):
            ax.plot(ctx.expr(ref.get("x", panel["x"]), case), ctx.expr(ref["y"], case), "k--", lw=1,
                    label=ref.get("label", "reference") if n == 0 else None)
    ax.set(xlabel=panel.get("xlabel", panel["x"]), ylabel=panel.get("ylabel", ""))
    if not panel.get("logy"):
        ax.ticklabel_format(style="sci", scilimits=(-2, 4), useMathText=True)


@panel_kind("residuals")
def residual_panel(ctx: FigureContext, ax, panel: dict, cases: list[str]) -> None:
    for case in cases:
        data = ctx.expr.cases[case]
        stress = np.sqrt(sum(data.columns["stress_" + c] ** 2 for c in COMPONENTS))
        scale = max(float(np.max(stress)), 1e-300)
        steps = sorted(set(int(s) for s in data.residuals["step"]) - {0})
        for s in steps:
            r = data.residuals["residual"][data.residuals["step"] == s] / scale
            if len(r) > 1:
                ax.semilogy(range(len(r)), np.maximum(r, 1e-18), color=ctx.plt.get_cmap("viridis")(s / len(steps)),
                            lw=0.8, alpha=0.6)
    ax.set(xlabel="Newton iteration", ylabel="|P:(σ − σ*)| / max|σ|")
    ax.xaxis.set_major_locator(ctx.ticker.MaxNLocator(integer=True))


@panel_kind("iterations")
def iteration_panel(ctx: FigureContext, ax, panel: dict, cases: list[str]) -> None:
    for case in cases:
        c = ctx.expr.cases[case].columns
        ax.step(c["step"][1:], c["iterations"][1:], where="mid", label=case if len(cases) > 1 else None)
    ax.set(xlabel="step", ylabel="iterations")
    ax.yaxis.set_major_locator(ctx.ticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)


def render_figure(ctx: FigureContext, figure: dict, path: pathlib.Path) -> None:
    panels = figure["panels"]
    ncols = min(figure.get("columns", 2), len(panels))
    fig, axes = ctx.plt.subplots(math.ceil(len(panels) / ncols), ncols, figsize=(5 * ncols, 3.6 * math.ceil(len(panels) / ncols)),
                                 squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        kind = panel.get("kind", "line")
        if kind not in PANELS:
            raise SystemExit(f"unknown panel kind '{kind}' ({', '.join(PANELS)})")
        PANELS[kind](ctx, ax, panel, ctx.expr.case_names(panel, ctx.spec))
        ax.set_title(panel.get("title", ""))
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    ctx.plt.close(fig)


def render_figures(run: RunOutput, expr: Expressions, folder: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    (folder / "figures").mkdir(exist_ok=True)
    ctx = FigureContext(run.spec, expr, plt, ticker)
    for i, figure in enumerate(run.spec.get("figures", [])):
        render_figure(ctx, figure, folder / "figures" / figure.get("file", f"figure_{i}.png"))


# --- markdown -----------------------------------------------------------------------------------

def provenance(folder: pathlib.Path) -> list[str]:
    rev = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=folder, capture_output=True,
                         text=True).stdout.strip()
    return [f"Generated by `reports/make_report.py` from [`report.json`](report.json) at `{rev or 'unknown'}` "
            f"on {datetime.date.today().isoformat()}; do not edit.", ""]


def load_case_table(run: RunOutput) -> list[str]:
    md = ["### Load cases", "", "| case | control | steps | iterations (max / mean) | max tangent error | converged |",
          "|---|---|---|---|---|---|"]
    for c in run.spec["cases"]:
        data = run.cases.get(c["name"])
        if data is None:
            md.append(f"| {c['name']} | {c.get('control', 'strain')} | – | – | – | **no** |")
            continue
        it, err = data.columns["iterations"][1:], data.columns["tangent_error"][1:]
        md.append(f"| {c['name']} | {c.get('control', 'strain')} | {len(it)} | {int(np.max(it, initial=0))} / "
                  f"{np.mean(it) if len(it) else 0:.2f} | {np.nanmax(err) if len(err) else 0:.1e} | "
                  f"{'yes' if np.all(data.columns['converged'] == 1) else '**no**'} |")
    return md


def check_table(checks: list[CheckResult]) -> list[str]:
    if not checks:
        return []
    md = ["", "### Checks", "", "| check | cases | max error (case) | tolerance | result |", "|---|---|---|---|---|"]
    for name in dict.fromkeys(c.name for c in checks):
        rows = [c for c in checks if c.name == name]
        worst = max(rows, key=lambda c: c.error)
        failed = [c.case for c in rows if not c.passed]
        result = "pass" if not failed else "**FAIL** (" + ", ".join(failed) + ")"
        md.append(f"| {name} | {len(rows)} | {worst.error:.1e} ({worst.case}) | {worst.tolerance:.0e} | {result} |")
    return md


def figure_sections(run: RunOutput) -> list[str]:
    md = []
    for i, f in enumerate(run.spec.get("figures", [])):
        md += ["", f"### {f['title']}", ""] + ([f["text"], ""] if "text" in f else [])
        md.append(f"![{f['title']}](figures/{f.get('file', f'figure_{i}.png')})")
    return md


def graph_section(run: RunOutput) -> list[str]:
    md = ["", "### Graph", "", "Properties wired between the materials of this report (`material::property`).", "",
          "```mermaid", "flowchart LR"]
    md += [f'  {m["name"]}["{m["name"]}<br/><small>{m["type"]}</small>"]' for m in run.graph]
    for m in run.graph:
        for i in m["inputs"]:
            src, prop = i["source"].split("::")
            md.append(f"  {src} -->|{prop}{' (local)' if i['kind'] == 'local' else ''}| {m['name']}")
    md += ["```", "", "| material | type | produces | consumes |", "|---|---|---|---|"]
    for m in run.graph:
        produced = ", ".join(p["name"] + (" (history)" if p["history"] else "") for p in m["properties"]
                             if not p["name"].startswith("_"))
        md.append(f"| {m['name']} | `{m['type']}` | {produced} | {', '.join(i['source'] for i in m['inputs'])} |")
    return md


def parameter_table(run: RunOutput) -> list[str]:
    md = ["", "### Parameters", "", "| material | parameter | value |", "|---|---|---|"]
    md += [f"| {m['name']} | `{p['name']}` | {json.dumps(p['value'])} |"
           for m in run.graph for p in m["parameters"] if p["name"] != "name" and p["value"] is not None]
    return md


def results_markdown(run: RunOutput, checks: list[CheckResult], folder: pathlib.Path) -> str:
    return "\n".join(["## Results", ""] + provenance(folder) + load_case_table(run) + check_table(checks)
                     + figure_sections(run) + graph_section(run) + parameter_table(run))


def replace_results(text: str, body: str) -> str:
    """Replaces everything between the report markers, leaving the hand-written part alone."""
    block = f"{BEGIN}\n{body}\n{END}"
    if BEGIN not in text:
        return text.rstrip("\n") + "\n\n" + block + "\n"
    return re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: block, text, flags=re.S)


# --- main ---------------------------------------------------------------------------------------

def build(tool: str, folder: pathlib.Path, check_only: bool) -> bool:
    run = run_tool(tool, folder / "report.json")
    expr = Expressions(run.spec, run.cases)
    checks = evaluate_checks(run.spec, expr)
    failed = [c for c in checks if not c.passed]
    for c in failed:
        print(f"{folder.name}: check '{c.name}' on {c.case} failed: {c.error:.3e} > {c.tolerance:.1e}", file=sys.stderr)
    if not run.converged:
        print(f"{folder.name}: a load case did not converge", file=sys.stderr)
    if not check_only:
        render_figures(run, expr, folder)
        readme = folder / "README.md"
        text = readme.read_text() if readme.exists() else f"# {run.spec.get('title', folder.name)}\n"
        readme.write_text(replace_results(text, results_markdown(run, checks, folder)))
    print(f"{folder.name}: {len(checks) - len(failed)}/{len(checks)} checks passed")
    return run.converged and not failed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", required=True, help="path to the material_report executable")
    p.add_argument("--check", action="store_true", help="checks only; exit 1 on a failed check")
    p.add_argument("folders", nargs="+", type=pathlib.Path)
    a = p.parse_args()
    ok = [build(a.tool, folder, a.check) for folder in a.folders]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
