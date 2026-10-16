#!/usr/bin/env python3
"""Run the load cases of material reports and update their figures and tables.

usage: make_report.py --tool <material_report> docs/materials/<material> [...]
"""
import argparse
import csv
import json
import pathlib
import re
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

SLOTS = ["11", "22", "33", "12", "13", "23"]
BEGIN, END = "<!-- report:begin -->", "<!-- report:end -->"


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


def from_origin(values):
    return [0.0] + list(values)


def style(ax):
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style="sci", scilimits=(-2, 4), useMathText=True)


def invariants(d):
    """Pressure p = tr(σ)/3 and von Mises stress q per row."""
    p, q = [], []
    for k in range(len(d["sig11"])):
        s11, s22, s33 = d["sig11"][k], d["sig22"][k], d["sig33"][k]
        s12, s13, s23 = d["sig12"][k], d["sig13"][k], d["sig23"][k]
        p.append((s11 + s22 + s33) / 3)
        q.append((0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
                  + 3 * (s12 ** 2 + s13 ** 2 + s23 ** 2)) ** 0.5)
    return p, q


def active(data, prefix, tol=1e-12):
    cols = [prefix + s for s in SLOTS]
    scale = max((abs(v) for c in cols for v in data[c]), default=0.0) or 1.0
    return [c for c in cols if max(abs(v) for v in data[c]) > tol * scale]


def driving(case):
    """Label and values of the driven strain slot."""
    i = max(range(6), key=lambda k: abs(case["direction"][k]))
    return i, ("γ" if i > 2 else "ε") + SLOTS[i]


def stress_figure(cases, data, out):
    n = len(cases)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(10, 3.4 * ((n + 1) // 2)), squeeze=False)
    for ax, case in zip(axes.flat, cases):
        d = data[case["name"]]
        i, label = driving(case)
        x = d["eps" + SLOTS[i]]
        for c in active(d, "sig"):
            ax.plot(from_origin(x), from_origin(d[c]), label="σ" + c[3:])
        ax.set_title(case["name"].replace("_", " "))
        ax.set_xlabel(label)
        ax.set_ylabel("stress")
        style(ax)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def uniaxial_stress_figure(d, its, out):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    x = from_origin(d["eps11"])
    ax[0, 0].plot(x, from_origin(d["sig11"]))
    ax[0, 0].set(xlabel="ε11", ylabel="σ11", title="stress-strain")
    ax[0, 1].plot(x, from_origin(d["eps22"]), label="ε22")
    ax[0, 1].plot(x, from_origin(d["eps33"]), "--", label="ε33")
    ax[0, 1].set(xlabel="ε11", ylabel="lateral strain", title="lateral strains")
    ax[0, 1].legend()
    scale = max(sum(d["sig" + c][k] ** 2 for c in SLOTS) ** 0.5 for k in range(len(d["sig11"]))) or 1.0
    steps = sorted(set(int(s) for s in its["step"]))
    cmap = plt.get_cmap("viridis")
    for s in steps:
        rows = [k for k, v in enumerate(its["step"]) if int(v) == s]
        res = [max(its["residual"][k] / scale, 1e-18) for k in rows]
        if len(res) > 1:
            ax[1, 0].semilogy(range(len(res)), res, color=cmap(s / len(steps)), alpha=0.6, lw=0.8)
    ax[1, 0].set(xlabel="Newton iteration", ylabel="|P(σ − σ*)| / max|σ|", title="residual per step (colour: step)")
    ax[1, 1].step(d["step"], d["iterations"], where="mid")
    ax[1, 1].set(xlabel="step", ylabel="iterations", title="iterations per step")
    ax[1, 1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[1, 1].set_ylim(bottom=0)
    for a in ax.flat:
        a.grid(alpha=0.3)
    for a in (ax[0, 0], ax[0, 1]):
        style(a)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def pq_figure(cases, data, out):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for case in cases:
        p, q = invariants(data[case["name"]])
        ax.plot(from_origin(p), from_origin(q), label=case["name"].replace("_", " "))
    ax.set(xlabel="p = tr σ / 3", ylabel="q = von Mises stress", title="stress paths")
    style(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def tangent_figure(cases, data, out):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for case in cases:
        d = data[case["name"]]
        ax.semilogy(d["step"], [max(e, 1e-17) for e in d["tangent_error"]], label=case["name"].replace("_", " "))
    ax.set(xlabel="step", ylabel="‖C − C_fd‖ / ‖C‖", title="consistent tangent vs central differences")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def history_figure(case, d, out):
    cols = [c for c in d if "::" in c and max(abs(v) for v in d[c]) > 0]
    if not cols:
        return False
    i, label = driving(case)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for c in cols:
        ax.plot(d["step"], d[c], label=c)
    ax.set(xlabel="step", ylabel="history", title="history variables, " + case["name"].replace("_", " "))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def report(tool, folder):
    spec = json.loads((folder / "case.json").read_text())
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([tool, folder / "case.json", tmp], check=True, stderr=subprocess.DEVNULL)
        data = {c["name"]: read_csv(pathlib.Path(tmp) / (c["name"] + ".csv")) for c in spec["cases"]}
        its = {c["name"]: read_csv(pathlib.Path(tmp) / (c["name"] + "_iterations.csv")) for c in spec["cases"]}

    md = ["## Results", "", "Generated by `reports/make_report.py` from [`case.json`](case.json); do not edit.", ""]
    md += ["| case | control | steps | iterations (max / mean) | max tangent error | converged |", "|---|---|---|---|---|---|"]
    for c in spec["cases"]:
        d = data[c["name"]]
        n = len(d.get("step", []))
        it = d.get("iterations", [0])
        conv = "yes" if n == len(c["path"][1:]) * c.get("steps", 50) and all(d["converged"]) else "**no**"
        md.append(f"| {c['name']} | {c.get('control', 'strain')} | {n} | {int(max(it))} / {sum(it) / max(n, 1):.2f} "
                  f"| {max(d.get('tangent_error', [0])):.1e} | {conv} |")

    stress_figure(spec["cases"], data, figures / "stress_strain.png")
    md += ["", "### Stress for prescribed strain paths", "", "![stress-strain](figures/stress_strain.png)"]
    pq_figure(spec["cases"], data, figures / "stress_paths.png")
    md += ["", "### Stress paths in the p–q plane", "", "![stress paths](figures/stress_paths.png)"]
    uni = [c for c in spec["cases"] if c.get("control") == "uniaxial_stress"]
    for c in uni:
        uniaxial_stress_figure(data[c["name"]], its[c["name"]], figures / f"{c['name']}.png")
        md += ["", "### Uniaxial stress (1D) and its Newton iterations", "",
               "Strain 11 is prescribed; the other five strains are solved for zero stress with the material tangent.",
               "", f"![uniaxial stress](figures/{c['name']}.png)"]
    tangent_figure(spec["cases"], data, figures / "tangent_error.png")
    md += ["", "### Tangent check", "", "![tangent error](figures/tangent_error.png)"]
    for c in uni:
        if history_figure(c, data[c["name"]], figures / f"history_{c['name']}.png"):
            md += ["", "### History variables", "", f"![history](figures/history_{c['name']}.png)"]

    readme = folder / "README.md"
    text = readme.read_text() if readme.exists() else f"# {spec['material']}\n\n{BEGIN}\n{END}\n"
    text = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: BEGIN + "\n" + "\n".join(md) + "\n" + END,
                  text, flags=re.S)
    readme.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tool", required=True, help="path to the material_report executable")
    p.add_argument("folders", nargs="+", type=pathlib.Path)
    a = p.parse_args()
    for folder in a.folders:
        report(a.tool, folder)
        print("updated", folder / "README.md")


if __name__ == "__main__":
    main()
