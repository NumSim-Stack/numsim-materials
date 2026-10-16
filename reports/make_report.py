#!/usr/bin/env python3
"""Build material reports from their report.json: run the cases, evaluate the checks, draw the figures.

usage: make_report.py --tool <material_report> [--check] docs/materials/<material> [...]

--check runs the cases and checks only (no figures, README untouched) and exits 1 if a check fails.
Expressions (figures, references, checks, "derived") see numpy as np, the parameters, the derived
values and the columns of the case as arrays, including p (pressure) and q (von Mises stress).
"""
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

import numpy as np

BEGIN, END = "<!-- report:begin -->", "<!-- report:end -->"
COMPONENTS = ("11", "22", "33", "12", "13", "23")


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def with_invariants(d):
    s = {c: d["stress_" + c] for c in COMPONENTS}
    d["p"] = (s["11"] + s["22"] + s["33"]) / 3
    d["q"] = np.sqrt(0.5 * ((s["11"] - s["22"]) ** 2 + (s["22"] - s["33"]) ** 2 + (s["33"] - s["11"]) ** 2)
                     + 3 * (s["12"] ** 2 + s["13"] ** 2 + s["23"] ** 2))
    return d


class Report:
    def __init__(self, tool, folder):
        self.folder = folder
        self.tmp = tempfile.TemporaryDirectory()
        out = pathlib.Path(self.tmp.name)
        run = subprocess.run([tool, folder / "report.json", out])
        if run.returncode == 2:
            raise SystemExit(f"{folder}: material_report failed")
        self.converged = run.returncode == 0
        self.spec = json.loads((out / "resolved.json").read_text())
        self.graph = json.loads((out / "graph.json").read_text())
        self.data, self.its = {}, {}
        for c in self.spec["cases"]:
            if (out / (c["name"] + ".csv")).exists():
                self.data[c["name"]] = with_invariants(read_csv(out / (c["name"] + ".csv")))
                self.its[c["name"]] = read_csv(out / (c["name"] + "_iterations.csv"))
        self.scope = {"np": np, **self.spec.get("parameters", {})}
        for k, expr in self.spec.get("derived", {}).items():
            self.scope[k] = self.eval(expr)

    def eval(self, expr, case=None):
        if not isinstance(expr, str):
            return expr
        scope = dict(self.scope)
        if case is not None:
            scope.update(self.data[case])
        return eval(expr, {"__builtins__": {}, "abs": abs, "max": max, "min": min}, scope)  # report authors' expressions

    def cases_of(self, item):
        names = item.get("cases", [item["case"]] if "case" in item else "*")
        if names == "*":
            names = [c["name"] for c in self.spec["cases"]]
        return [n for n in names if n in self.data]

    # --- checks ---------------------------------------------------------------------------------
    def checks(self):
        rows = []
        for chk in self.spec.get("checks", []):
            for case in self.cases_of(chk):
                d = self.data[case]
                shape = d["step"].shape
                mask = d["step"] > 0
                if "where" in chk:
                    mask &= np.broadcast_to(np.asarray(self.eval(chk["where"], case), bool), shape)
                value = np.broadcast_to(np.asarray(self.eval(chk["value"], case), float), shape)[mask]
                expected = np.broadcast_to(np.asarray(self.eval(chk.get("expected", 0), case), float), shape)[mask]
                err = float(np.max(np.abs(value - expected), initial=0.0))
                if chk.get("relative", False):
                    err /= max(float(np.max(np.abs(expected), initial=0.0)), 1e-300)
                if not np.isfinite(err):
                    err = math.inf
                rows.append({"name": chk["name"], "case": case, "error": err, "tolerance": chk["tolerance"],
                             "passed": err <= chk["tolerance"]})
        return rows

    # --- figures --------------------------------------------------------------------------------
    def figure(self, fig_spec, path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker

        panels = fig_spec["panels"]
        ncols = min(fig_spec.get("columns", 2), len(panels))
        nrows = math.ceil(len(panels) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
        for ax, panel in zip(axes.flat, panels):
            self.panel(ax, panel, plt, matplotlib.ticker)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)

    def panel(self, ax, panel, plt, ticker):
        kind = panel.get("kind", "line")
        cases = self.cases_of(panel)
        if kind == "residuals":
            for case in cases:
                its, d = self.its[case], self.data[case]
                scale = max(float(np.max(np.sqrt(sum(d["stress_" + c] ** 2 for c in COMPONENTS)))), 1e-300)
                steps = sorted(set(int(s) for s in its["step"]) - {0})
                for s in steps:
                    r = its["residual"][its["step"] == s] / scale
                    if len(r) > 1:
                        ax.semilogy(range(len(r)), np.maximum(r, 1e-18), color=plt.get_cmap("viridis")(s / len(steps)),
                                    lw=0.8, alpha=0.6)
            ax.set(xlabel="Newton iteration", ylabel="|P:(σ − σ*)| / max|σ|")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        elif kind == "iterations":
            for case in cases:
                d = self.data[case]
                ax.step(d["step"][1:], d["iterations"][1:], where="mid", label=case if len(cases) > 1 else None)
            ax.set(xlabel="step", ylabel="iterations")
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_ylim(bottom=0)
        else:
            ys = panel["y"] if isinstance(panel["y"], list) else [panel["y"]]
            for n, case in enumerate(cases):
                x = self.eval(panel["x"], case)
                for y in ys:
                    expr, label = (y["expr"], y.get("label", y["expr"])) if isinstance(y, dict) else (y, y)
                    lab = label if len(cases) == 1 else f"{case}: {label}"
                    (ax.semilogy if panel.get("logy") else ax.plot)(x, self.eval(expr, case), label=lab)
                for ref in panel.get("reference", []):
                    rx = self.eval(ref.get("x", panel["x"]), case)
                    ax.plot(rx, self.eval(ref["y"], case), "k--", lw=1, label=ref.get("label", "reference") if n == 0 else None)
            ax.set(xlabel=panel.get("xlabel", panel["x"]), ylabel=panel.get("ylabel", ""))
            if not panel.get("logy"):
                ax.ticklabel_format(style="sci", scilimits=(-2, 4), useMathText=True)
        ax.set_title(panel.get("title", ""))
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0] and kind != "residuals":
            ax.legend(fontsize=7)

    # --- markdown -------------------------------------------------------------------------------
    def markdown(self, checks):
        md = ["## Results", "", self.provenance(), ""]
        md += ["### Load cases", "", "| case | control | steps | iterations (max / mean) | max tangent error | converged |",
               "|---|---|---|---|---|---|"]
        for c in self.spec["cases"]:
            d = self.data.get(c["name"])
            if d is None:
                md.append(f"| {c['name']} | {c.get('control', 'strain')} | – | – | – | **no** |")
                continue
            it, err = d["iterations"][1:], d["tangent_error"][1:]
            conv = "yes" if np.all(d["converged"] == 1) else "**no**"
            md.append(f"| {c['name']} | {c.get('control', 'strain')} | {len(it)} | {int(np.max(it, initial=0))} / "
                      f"{np.mean(it) if len(it) else 0:.2f} | {np.nanmax(err) if len(err) else 0:.1e} | {conv} |")
        if checks:
            md += ["", "### Checks", "", "| check | cases | max error (case) | tolerance | result |", "|---|---|---|---|---|"]
            for name in dict.fromkeys(r["name"] for r in checks):
                rows = [r for r in checks if r["name"] == name]
                worst = max(rows, key=lambda r: r["error"])
                failed = [r["case"] for r in rows if not r["passed"]]
                result = "pass" if not failed else "**FAIL** (" + ", ".join(failed) + ")"
                md.append(f"| {name} | {len(rows)} | {worst['error']:.1e} ({worst['case']}) | {worst['tolerance']:.0e} | {result} |")
        for i, f in enumerate(self.spec.get("figures", [])):
            name = f.get("file", f"figure_{i}.png")
            md += ["", f"### {f['title']}", ""] + ([f["text"], ""] if "text" in f else []) + [f"![{f['title']}](figures/{name})"]
        md += ["", "### Graph", "", "Properties wired between the materials of this report (`material::property`).", "",
               "```mermaid", "flowchart LR"]
        for m in self.graph:
            md.append(f'  {m["name"]}["{m["name"]}<br/><small>{m["type"]}</small>"]')
        for m in self.graph:
            for i in m["inputs"]:
                src, prop = i["source"].split("::")
                md.append(f"  {src} -->|{prop}{' (local)' if i['kind'] == 'local' else ''}| {m['name']}")
        md += ["```", "", "| material | type | produces | consumes |", "|---|---|---|---|"]
        for m in self.graph:
            produced = ", ".join(p["name"] + (" (history)" if p["history"] else "") for p in m["properties"]
                                 if not p["name"].startswith("_"))
            consumed = ", ".join(i["source"] for i in m["inputs"])
            md.append(f"| {m['name']} | `{m['type']}` | {produced} | {consumed} |")
        md += ["", "### Parameters", "", "| material | parameter | value |", "|---|---|---|"]
        for m in self.graph:
            for p in m["parameters"]:
                if p["name"] != "name" and p["value"] is not None:
                    md.append(f"| {m['name']} | `{p['name']}` | {json.dumps(p['value'])} |")
        return md

    def provenance(self):
        rev = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=self.folder, capture_output=True,
                             text=True).stdout.strip()
        return (f"Generated by `reports/make_report.py` from [`report.json`](report.json) at `{rev or 'unknown'}` "
                f"on {datetime.date.today().isoformat()}; do not edit.")

    def write(self, checks):
        figures = self.folder / "figures"
        figures.mkdir(exist_ok=True)
        for i, f in enumerate(self.spec.get("figures", [])):
            self.figure(f, figures / f.get("file", f"figure_{i}.png"))
        readme = self.folder / "README.md"
        text = readme.read_text() if readme.exists() else f"# {self.spec.get('title', self.folder.name)}\n\n{BEGIN}\n{END}\n"
        body = BEGIN + "\n" + "\n".join(self.markdown(checks)) + "\n" + END
        readme.write_text(re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: body, text, flags=re.S))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", required=True, help="path to the material_report executable")
    p.add_argument("--check", action="store_true", help="checks only; exit 1 on a failed check")
    p.add_argument("folders", nargs="+", type=pathlib.Path)
    a = p.parse_args()
    failed = False
    for folder in a.folders:
        rep = Report(a.tool, folder)
        checks = rep.checks()
        bad = [c for c in checks if not c["passed"]]
        failed = failed or bool(bad) or not rep.converged
        for c in bad:
            print(f"{folder.name}: check '{c['name']}' on {c['case']} failed: {c['error']:.3e} > {c['tolerance']:.1e}",
                  file=sys.stderr)
        if not rep.converged:
            print(f"{folder.name}: a load case did not converge", file=sys.stderr)
        if not a.check:
            rep.write(checks)
        print(f"{folder.name}: {len(checks) - len(bad)}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
