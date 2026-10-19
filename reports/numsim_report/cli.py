"""make_report.py --tool <material_report> (--out <directory> | --check) docs/materials/<material> [...]

Writes <directory>/<material>.html, self-contained (figures embedded), and an index.html.
--check runs the cases and checks only; the exit code is 1 if a check fails or a step does not converge.

report.json
  include      files merged first (arrays appended, local keys win)
  parameters   values; every string "$name" in the document is replaced by parameters.name
  derived      expressions of the parameters ("E": "9 * K * G / (3 * K + G)")
  model        the material graph, a json_model document {"materials": [...]}
  point        strain_source (or deformation_gradient_source), stress_source; optional tangent_source,
               stress_property, tangent_property ("damaged_stress"), time_source
  record       extra columns wired to graph properties: "kappa": "j2::equivalent_plastic_strain";
               a rank-2 tensor gives six columns name_11 ... name_23, a component is selected with
               "j2::plastic_strain[1,1]" (1-based); columns are ordered by name
  cases        name; control: strain | uniaxial_stress | mixed with stress_components ["22", "33"];
               proportional: direction (strain per unit load) and path (load factors); non-proportional:
               points (strain states in order, the load factor counts segments); stress (target per
               unit load factor); tensors as components [11, 22, 33, 12, 13, 23]; steps per segment,
               scale (applies to the strains), dt;
               tangent_every: finite-difference tangent check every n-th step (1; 0 = never)
  figures      title, text, columns, panels: kind line (x, y, reference, logy, xlabel, ylabel),
               residuals, iterations; case or cases (list or "*")
  checks       name, case/cases, value, expected (default 0), tolerance, relative, where;
               plot: true or {"x": ...} draws value, expected and the error per case
  studies      step-size studies: name, case, steps (list of step counts), quantity (expression of
               the final state, default stress_11), min_order (default 0.9); the error against the
               finest run is fitted with a power of the load increment

Expressions are Python over numpy (np), the parameters, the derived values and the columns of the
case: step, time, load, iterations, converged, tangent_error, strain_11..23 and stress_11..23
(tensor components), p (pressure), q (von Mises stress) and the recorded columns.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from . import html
from .data import run_tool
from .expressions import Expressions, evaluate_checks
from .studies import evaluate_studies


def build(tool: str, folder: pathlib.Path, out: pathlib.Path | None) -> html.ReportSummary:
    run = run_tool(tool, folder / "report.json")
    expr = Expressions(run.spec, run.cases)
    checks = evaluate_checks(expr)
    studies = evaluate_studies(expr, run.studies)
    failed = [c for c in checks if not c.passed]
    for c in failed:
        print(f"{folder.name}: check '{c.name}' on {c.case} failed: {c.error:.3e} > {c.tolerance:.1e}", file=sys.stderr)
    for s in studies:
        if not s.passed:
            print(f"{folder.name}: study '{s.name}': order {s.order:.2f} < {s.min_order:g}", file=sys.stderr)
    failed_studies = sum(not s.passed for s in studies)
    if not run.converged:
        print(f"{folder.name}: a load case did not converge", file=sys.stderr)
    if out is not None:
        from .figures import render_check_figures, render_figures  # matplotlib only when figures are wanted
        from .studies import render_study_figure

        out.mkdir(parents=True, exist_ok=True)
        target = out / f"{folder.name}.html"
        target.write_text(html.report_page(run, expr, checks, render_figures(expr), folder, render_check_figures(expr),
                                           studies, [render_study_figure(s) for s in studies]))
        print(f"{folder.name}: wrote {target}")
    total, ok = len(checks) + len(studies), len(checks) - len(failed) + len(studies) - failed_studies
    print(f"{folder.name}: {ok}/{total} checks passed")
    return html.ReportSummary(folder.name, ok, total, run.converged)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", required=True, help="path to the material_report executable")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=pathlib.Path, help="directory for the HTML reports")
    mode.add_argument("--check", action="store_true", help="checks only; exit 1 on a failed check")
    p.add_argument("folders", nargs="+", type=pathlib.Path)
    a = p.parse_args(argv)
    entries = [build(a.tool, folder, a.out) for folder in a.folders]
    if a.out is not None:
        (a.out / "index.html").write_text(html.index_page(entries))
    return 0 if all(e.ok for e in entries) else 1
