"""The HTML page of a report and the index of several."""
from __future__ import annotations

import base64
import datetime
import html
import json
import pathlib
import re
import subprocess
from dataclasses import dataclass

import numpy as np

from .data import RunOutput
from .expressions import CheckResult, Expressions

STYLE = """
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 16px; color: #222; }
h1 { margin-bottom: 0.2em; } .meta { color: #666; margin-bottom: 1.5em; }
table { border-collapse: collapse; margin: 0.5em 0 1.5em; font-size: 0.92em; }
th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: left; vertical-align: top; }
th { background: #f3f3f3; } td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: #17692c; } .fail { color: #b00020; font-weight: bold; }
img { max-width: 100%; border: 1px solid #eee; } figure { margin: 0 0 2em; } figcaption { color: #444; margin-bottom: 0.5em; }
"""


class Raw(str):
    """Markup that goes into a table cell unescaped."""


def esc(value) -> str:
    return value if isinstance(value, Raw) else html.escape(str(value))


def verdict(ok: bool, text: str | None = None) -> Raw:
    return Raw(f'<span class="{"pass" if ok else "fail"}">{esc(text or ("pass" if ok else "FAIL"))}</span>')


def table(head: list[str], rows: list[list], numeric: list[int] = ()) -> str:
    """`numeric` lists the right-aligned columns."""
    def cell(i, value):
        return f'<td class="num">{esc(value)}</td>' if i in numeric else f"<td>{esc(value)}</td>"
    body = "".join("<tr>" + "".join(cell(i, v) for i, v in enumerate(row)) + "</tr>" for row in rows)
    return "<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr>" + body + "</table>"


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>'
            f"<style>{STYLE}</style></head><body><h1>{esc(title)}</h1>{body}</body></html>\n")


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


# --- parameters -------------------------------------------------------------------------------

def readable_path(path: str, spec: dict) -> str:
    """model.materials[1].K -> elastic.K; cases[0].scale -> uniaxial_strain.scale; checks[0].tolerance -> check 'tangent'.tolerance"""
    for pattern, items, fmt in ((r"model\.materials\[(\d+)\]\.(.+)", spec.get("model", {}).get("materials", []), "{}.{}"),
                                (r"cases\[(\d+)\]\.(.+)", spec.get("cases", []), "{}.{}"),
                                (r"checks\[(\d+)\]\.(.+)", spec.get("checks", []), "check '{}'.{}")):
        m = re.fullmatch(pattern, path)
        if m:
            return fmt.format(items[int(m[1])].get("name", m[1]), m[2])
    return path


def parameter_rows(spec: dict) -> list[tuple[str, str, list[str]]]:
    usage = spec.get("parameter_usage", {})
    return [(name, json.dumps(value), [readable_path(p, spec) for p in usage.get(name, [])])
            for name, value in spec.get("parameters", {}).items()]


def derived_rows(spec: dict, expr: Expressions) -> list[tuple[str, str, str]]:
    return [(name, e, f"{expr.scope[name]:.6g}") for name, e in spec.get("derived", {}).items()]


def parameter_section(run: RunOutput, expr: Expressions) -> str:
    out = "<h2>Parameters</h2>" + table(["parameter", "value", "used in"],
                                        [[n, v, ", ".join(u) or "–"] for n, v, u in parameter_rows(run.spec)], [1])
    if run.spec.get("derived"):
        out += "<h3>Derived</h3>" + table(["name", "expression", "value"], [list(r) for r in derived_rows(run.spec, expr)], [2])
    rows = [[m["name"], m["type"], p["name"], p["value"] if isinstance(p["value"], str) else json.dumps(p["value"])]
            for m in run.graph for p in m["parameters"] if p["name"] != "name" and p["value"] is not None]
    return out + "<h3>Material parameters as used</h3>" + table(["material", "type", "parameter", "value"], rows)


# --- results ----------------------------------------------------------------------------------

def load_case_section(run: RunOutput) -> str:
    rows = []
    for c in run.spec["cases"]:
        data = run.cases.get(c["name"])
        if data is None:
            rows.append([c["name"], c.get("control", "strain"), "–", "–", "–", verdict(False, "no")])
            continue
        it, err = data.columns["iterations"][1:], data.columns["tangent_error"][1:]
        ok = bool(np.all(data.columns["converged"] == 1))
        rows.append([c["name"], c.get("control", "strain"), len(it),
                     f"{int(np.max(it, initial=0))} / {np.mean(it) if len(it) else 0:.2f}",
                     f"{np.nanmax(err) if len(err) else 0:.1e}", verdict(ok, "yes" if ok else "no")])
    return "<h2>Load cases</h2>" + table(
        ["case", "control", "steps", "iterations (max / mean)", "max tangent error", "converged"], rows, [2, 3, 4])


def check_section(checks: list[CheckResult]) -> str:
    if not checks:
        return ""
    rows = []
    for name in dict.fromkeys(c.name for c in checks):
        group = [c for c in checks if c.name == name]
        worst = max(group, key=lambda c: c.error)
        failed = [c.case for c in group if not c.passed]
        rows.append([name, len(group), f"{worst.error:.1e} ({worst.case})", f"{worst.tolerance:.0e}",
                     verdict(not failed, None if not failed else f"FAIL ({', '.join(failed)})")])
    return "<h2>Checks</h2>" + table(["check", "cases", "max error (case)", "tolerance", "result"], rows, [1, 2, 3])


def figure_section(spec: dict, figures: list[bytes]) -> str:
    out = ""
    for f, png in zip(spec.get("figures", []), figures):
        caption = f"<figcaption>{esc(f['text'])}</figcaption>" if "text" in f else ""
        out += f"<h2>{esc(f['title'])}</h2><figure>{caption}<img alt=\"{esc(f['title'])}\" src=\"{data_uri(png)}\"></figure>"
    return out


def graph_section(run: RunOutput) -> str:
    rows = [[m["name"], m["type"],
             ", ".join(p["name"] + (" (history)" if p["history"] else "") for p in m["properties"] if not p["name"].startswith("_")),
             ", ".join(i["source"] + (" (local)" if i["kind"] == "local" else "") for i in m["inputs"])] for m in run.graph]
    return "<h2>Graph</h2>" + table(["material", "type", "produces", "consumes"], rows)


def revision(folder: pathlib.Path) -> str:
    return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=folder, capture_output=True,
                          text=True).stdout.strip() or "unknown"


def report_page(run: RunOutput, expr: Expressions, checks: list[CheckResult], figures: list[bytes],
                folder: pathlib.Path) -> str:
    passed = sum(c.passed for c in checks)
    meta = (f"{passed}/{len(checks)} checks passed · {'all steps converged' if run.converged else 'a case did not converge'}"
            f" · {esc(revision(folder))} · {datetime.date.today().isoformat()}")
    body = (f'<div class="meta">{meta}</div>' + parameter_section(run, expr) + load_case_section(run)
            + check_section(checks) + figure_section(run.spec, figures) + graph_section(run))
    return page(run.spec.get("title", folder.name), body)


# --- index ------------------------------------------------------------------------------------

@dataclass
class ReportSummary:
    name: str
    passed: int
    checks: int
    converged: bool

    @property
    def ok(self) -> bool:
        return self.passed == self.checks and self.converged


def index_page(entries: list[ReportSummary]) -> str:
    rows = [[Raw(f'<a href="{esc(e.name)}.html">{esc(e.name)}</a>'), f"{e.passed}/{e.checks}", verdict(e.ok)] for e in entries]
    return page("Material reports", table(["material", "checks", "result"], rows, [1]))
