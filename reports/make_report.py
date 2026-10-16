#!/usr/bin/env python3
"""Build material reports from their report.json: run the cases, evaluate the checks, draw the figures.

usage: make_report.py --tool <material_report> --out <directory> docs/materials/<material> [...]
       make_report.py --tool <material_report> --check docs/materials/<material> [...]

Writes <directory>/<material>.html, self-contained (figures embedded). --check runs the cases and
checks only and exits 1 if a check fails or a step does not converge.

report.json: "include" (merged first), "parameters" ("$name" strings take their values), "derived"
(expressions), "model" (json_model document), "point" (strain_source, stress_source, ...), "record"
(columns wired to "material::property[i,j]"), "cases", "figures" (panels of kind line, residuals,
iterations), "checks" (max |value - expected| <= tolerance, optionally relative / where).
Expressions see numpy as np, the parameters, the derived values and the columns of the case as
arrays: step, time, load, iterations, converged, tangent_error, strain_11..23, stress_11..23,
p (pressure), q (von Mises stress) and the recorded columns.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime
import html
import io
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


def render_figure(ctx: FigureContext, figure: dict) -> bytes:
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
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=100)
    ctx.plt.close(fig)
    return png.getvalue()


def render_figures(run: RunOutput, expr: Expressions) -> list[bytes]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ctx = FigureContext(run.spec, expr, plt, ticker)
    return [render_figure(ctx, figure) for figure in run.spec.get("figures", [])]


# --- html ---------------------------------------------------------------------------------------

STYLE = """
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 16px; color: #222; }
h1 { margin-bottom: 0.2em; } .meta { color: #666; margin-bottom: 1.5em; }
table { border-collapse: collapse; margin: 0.5em 0 1.5em; font-size: 0.92em; }
th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: left; vertical-align: top; }
th { background: #f3f3f3; } td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: #17692c; } .fail { color: #b00020; font-weight: bold; }
img { max-width: 100%; border: 1px solid #eee; } figure { margin: 0 0 2em; } figcaption { color: #444; margin-bottom: 0.5em; }
"""


def html_table(head: list[str], rows: list[list], classes: list[str] | None = None) -> str:
    """A table; cells are escaped unless they are already markup (tuple ("html", text))."""
    def cell(value, cls):
        text = value[1] if isinstance(value, tuple) and value[0] == "html" else html.escape(str(value))
        return f"<td{f' class={cls}' if cls else ''}>{text}</td>"
    classes = classes or [""] * len(head)
    body = "".join("<tr>" + "".join(cell(v, c) for v, c in zip(row, classes)) + "</tr>" for row in rows)
    return "<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>" + body + "</table>"


def readable_path(path: str, spec: dict) -> str:
    """model.materials[1].K -> elastic.K, cases[0].scale -> uniaxial_strain.scale"""
    m = re.fullmatch(r"model\.materials\[(\d+)\]\.(.+)", path)
    if m:
        return f"{spec['model']['materials'][int(m[1])].get('name', m[1])}.{m[2]}"
    m = re.fullmatch(r"cases\[(\d+)\]\.(.+)", path)
    if m:
        return f"{spec['cases'][int(m[1])].get('name', m[1])}.{m[2]}"
    m = re.fullmatch(r"checks\[(\d+)\]\.(.+)", path)
    if m:
        return f"check '{spec['checks'][int(m[1])].get('name', m[1])}'.{m[2]}"
    return path


def parameter_rows(spec: dict) -> list[tuple[str, str, list[str]]]:
    usage = spec.get("parameter_usage", {})
    return [(name, json.dumps(value), [readable_path(p, spec) for p in usage.get(name, [])])
            for name, value in spec.get("parameters", {}).items()]


def derived_rows(spec: dict, expr: Expressions) -> list[tuple[str, str, str]]:
    return [(name, e, f"{expr.scope[name]:.6g}") for name, e in spec.get("derived", {}).items()]


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def parameter_section(run: RunOutput, expr: Expressions) -> str:
    out = "<h2>Parameters</h2>" + html_table(["parameter", "value", "used in"],
                                             [[n, v, ", ".join(u) or "–"] for n, v, u in parameter_rows(run.spec)],
                                             ["", "num", ""])
    if run.spec.get("derived"):
        out += "<h3>Derived</h3>" + html_table(["name", "expression", "value"],
                                               [list(r) for r in derived_rows(run.spec, expr)], ["", "", "num"])
    rows = [[m["name"], m["type"], p["name"], p["value"] if isinstance(p["value"], str) else json.dumps(p["value"])]
            for m in run.graph for p in m["parameters"] if p["name"] != "name" and p["value"] is not None]
    return out + "<h3>Material parameters as used</h3>" + html_table(["material", "type", "parameter", "value"], rows)


def load_case_section(run: RunOutput) -> str:
    rows = []
    for c in run.spec["cases"]:
        data = run.cases.get(c["name"])
        if data is None:
            rows.append([c["name"], c.get("control", "strain"), "–", "–", "–", ("html", '<span class="fail">no</span>')])
            continue
        it, err = data.columns["iterations"][1:], data.columns["tangent_error"][1:]
        ok = bool(np.all(data.columns["converged"] == 1))
        rows.append([c["name"], c.get("control", "strain"), len(it),
                     f"{int(np.max(it, initial=0))} / {np.mean(it) if len(it) else 0:.2f}",
                     f"{np.nanmax(err) if len(err) else 0:.1e}",
                     ("html", f'<span class="{"pass" if ok else "fail"}">{"yes" if ok else "no"}</span>')])
    return "<h2>Load cases</h2>" + html_table(
        ["case", "control", "steps", "iterations (max / mean)", "max tangent error", "converged"], rows,
        ["", "", "num", "num", "num", ""])


def check_section(checks: list[CheckResult]) -> str:
    if not checks:
        return ""
    rows = []
    for name in dict.fromkeys(c.name for c in checks):
        group = [c for c in checks if c.name == name]
        worst = max(group, key=lambda c: c.error)
        failed = [c.case for c in group if not c.passed]
        result = ("html", '<span class="pass">pass</span>' if not failed
                  else f'<span class="fail">FAIL ({html.escape(", ".join(failed))})</span>')
        rows.append([name, len(group), f"{worst.error:.1e} ({worst.case})", f"{worst.tolerance:.0e}", result])
    return "<h2>Checks</h2>" + html_table(["check", "cases", "max error (case)", "tolerance", "result"], rows,
                                          ["", "num", "num", "num", ""])


def figure_section(run: RunOutput, figures: list[bytes]) -> str:
    out = ""
    for f, png in zip(run.spec.get("figures", []), figures):
        caption = f"<figcaption>{html.escape(f['text'])}</figcaption>" if "text" in f else ""
        out += f"<h2>{html.escape(f['title'])}</h2><figure>{caption}<img alt=\"{html.escape(f['title'])}\" src=\"{data_uri(png)}\"></figure>"
    return out


def graph_section(run: RunOutput) -> str:
    rows = [[m["name"], m["type"],
             ", ".join(p["name"] + (" (history)" if p["history"] else "") for p in m["properties"] if not p["name"].startswith("_")),
             ", ".join(i["source"] + (" (local)" if i["kind"] == "local" else "") for i in m["inputs"])] for m in run.graph]
    return "<h2>Graph</h2>" + html_table(["material", "type", "produces", "consumes"], rows)


def revision(folder: pathlib.Path) -> str:
    return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=folder, capture_output=True,
                          text=True).stdout.strip() or "unknown"


def html_report(run: RunOutput, expr: Expressions, checks: list[CheckResult], figures: list[bytes],
                folder: pathlib.Path) -> str:
    title = html.escape(run.spec.get("title", folder.name))
    passed = sum(c.passed for c in checks)
    meta = (f"{passed}/{len(checks)} checks passed · {'all steps converged' if run.converged else 'a case did not converge'}"
            f" · {html.escape(revision(folder))} · {datetime.date.today().isoformat()}")
    body = (parameter_section(run, expr) + load_case_section(run) + check_section(checks)
            + figure_section(run, figures) + graph_section(run))
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{title}</title>"
            f"<style>{STYLE}</style></head><body><h1>{title}</h1><div class=\"meta\">{meta}</div>{body}</body></html>\n")


# --- main ---------------------------------------------------------------------------------------

def index_page(entries: list[tuple[str, int, int, bool]]) -> str:
    """entries: (material, checks passed, checks, converged)"""
    rows = [[("html", f'<a href="{html.escape(n)}.html">{html.escape(n)}</a>'), f"{p}/{t}",
             ("html", f'<span class="{"pass" if p == t and c else "fail"}">{"pass" if p == t and c else "FAIL"}</span>')]
            for n, p, t, c in entries]
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Material reports</title>"
            f"<style>{STYLE}</style></head><body><h1>Material reports</h1>"
            + html_table(["material", "checks", "result"], rows, ["", "num", ""]) + "</body></html>\n")


def build(tool: str, folder: pathlib.Path, out: pathlib.Path | None) -> tuple[str, int, int, bool]:
    run = run_tool(tool, folder / "report.json")
    expr = Expressions(run.spec, run.cases)
    checks = evaluate_checks(run.spec, expr)
    failed = [c for c in checks if not c.passed]
    for c in failed:
        print(f"{folder.name}: check '{c.name}' on {c.case} failed: {c.error:.3e} > {c.tolerance:.1e}", file=sys.stderr)
    if not run.converged:
        print(f"{folder.name}: a load case did not converge", file=sys.stderr)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        page = out / f"{folder.name}.html"
        page.write_text(html_report(run, expr, checks, render_figures(run, expr), folder))
        print(f"{folder.name}: wrote {page}")
    print(f"{folder.name}: {len(checks) - len(failed)}/{len(checks)} checks passed")
    return folder.name, len(checks) - len(failed), len(checks), run.converged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", required=True, help="path to the material_report executable")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=pathlib.Path, help="directory for the HTML reports")
    mode.add_argument("--check", action="store_true", help="checks only; exit 1 on a failed check")
    p.add_argument("folders", nargs="+", type=pathlib.Path)
    a = p.parse_args()
    entries = [build(a.tool, folder, a.out) for folder in a.folders]
    if a.out is not None:
        (a.out / "index.html").write_text(index_page(entries))
    sys.exit(0 if all(p == t and c for _, p, t, c in entries) else 1)


if __name__ == "__main__":
    main()
