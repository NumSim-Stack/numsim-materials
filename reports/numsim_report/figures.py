"""Figures of a report: panels of registered kinds, rendered to PNG."""
from __future__ import annotations

import io
import math
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

from .data import COMPONENTS, pretty_label  # noqa: E402
from .expressions import Expressions  # noqa: E402

PanelRenderer = Callable[[Expressions, plt.Axes, dict, list[str]], None]
PANELS: dict[str, PanelRenderer] = {}


def panel_kind(name: str):
    """Registers a renderer for panels with "kind": name."""
    def register(fn: PanelRenderer) -> PanelRenderer:
        PANELS[name] = fn
        return fn
    return register


@panel_kind("line")
def line_panel(expr: Expressions, ax, panel: dict, cases: list[str]) -> None:
    """y expressions over x per case, with optional dashed reference curves."""
    ys = panel["y"] if isinstance(panel["y"], list) else [panel["y"]]
    plot = ax.semilogy if panel.get("logy") else ax.plot
    for n, case in enumerate(cases):
        x = expr(panel["x"], case)
        for y in ys:
            e, label = (y["expr"], y.get("label", y["expr"])) if isinstance(y, dict) else (y, pretty_label(y))
            plot(x, expr(e, case), label=label if len(cases) == 1 else f"{case}: {label}")
        for ref in panel.get("reference", []):
            ax.plot(expr(ref.get("x", panel["x"]), case), expr(ref["y"], case), "k--", lw=1,
                    label=ref.get("label", "reference") if n == 0 else None)
    single = ys[0] if len(ys) == 1 and isinstance(ys[0], str) else None
    ax.set(xlabel=panel.get("xlabel", pretty_label(panel["x"])),
           ylabel=panel.get("ylabel", pretty_label(single) if single else ""))
    if not panel.get("logy"):
        ax.ticklabel_format(style="sci", scilimits=(-2, 4), useMathText=True)


@panel_kind("residuals")
def residual_panel(expr: Expressions, ax, panel: dict, cases: list[str]) -> None:
    """Newton residual per iteration of every step, relative to the largest stress of the case."""
    for case in cases:
        data = expr.cases[case]
        scale = max(float(np.max(np.sqrt(sum(data.columns["stress_" + c] ** 2 for c in COMPONENTS)))), 1e-300)
        steps = sorted(set(int(s) for s in data.residuals["step"]) - {0})
        for s in steps:
            r = data.residuals["residual"][data.residuals["step"] == s] / scale
            if len(r) > 1:
                ax.semilogy(range(len(r)), np.maximum(r, 1e-18), color=plt.get_cmap("viridis")(s / len(steps)),
                            lw=0.8, alpha=0.6)
    ax.set(xlabel="Newton iteration", ylabel="|P:(σ − σ*)| / max|σ|")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


@panel_kind("iterations")
def iteration_panel(expr: Expressions, ax, panel: dict, cases: list[str]) -> None:
    for case in cases:
        c = expr.cases[case].columns
        ax.step(c["step"][1:], c["iterations"][1:], where="mid", label=case if len(cases) > 1 else None)
    ax.set(xlabel="step", ylabel="iterations")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)


def render_figure(expr: Expressions, figure: dict) -> bytes:
    """One figure as PNG; `columns` panels per row (default 2)."""
    panels = figure["panels"]
    ncols = min(figure.get("columns", 2), len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    try:
        for ax, panel in zip(axes.flat, panels):
            kind = panel.get("kind", "line")
            if kind not in PANELS:
                raise ValueError(f"unknown panel kind '{kind}' (one of {', '.join(PANELS)})")
            PANELS[kind](expr, ax, panel, expr.case_names(panel))
            ax.set_title(panel.get("title", ""))
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.tight_layout()
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=100)
        return png.getvalue()
    finally:
        plt.close(fig)


def render_figures(expr: Expressions) -> list[bytes]:
    return [render_figure(expr, figure) for figure in expr.spec.get("figures", [])]


def render_check_figure(expr: Expressions, check: dict) -> bytes:
    """Value and expected over x (default: step) per case, and the error of the check on a log scale."""
    plot = check["plot"] if isinstance(check["plot"], dict) else {}
    x_expr = plot.get("x", "step")
    relative = check.get("relative", False)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    try:
        for n, case in enumerate(expr.case_names(check)):
            x = np.broadcast_to(np.asarray(expr(x_expr, case), float), expr.cases[case].columns["step"].shape)
            value = np.broadcast_to(np.asarray(expr(check["value"], case), float), x.shape)
            expected = np.broadcast_to(np.asarray(expr(check.get("expected", 0), case), float), x.shape)
            prefix = f"{case}: " if len(expr.case_names(check)) > 1 else ""
            left.plot(x[1:], value[1:], label=prefix + "value")
            left.plot(x[1:], expected[1:], "k--", lw=1, label="expected" if n == 0 else None)
            error = np.abs(value - expected)[1:]
            if relative:
                error = error / max(float(np.max(np.abs(expected[1:]), initial=0.0)), 1e-300)
            right.semilogy(x[1:], np.maximum(error, 1e-18), label=case)
        left.set(xlabel=plot.get("xlabel", pretty_label(x_expr)), ylabel=plot.get("ylabel", ""), title=check["name"])
        left.ticklabel_format(style="sci", scilimits=(-2, 4), useMathText=True)
        right.axhline(float(check["tolerance"]) or 1e-18, color="#b00020", lw=1, ls=":", label="tolerance")
        right.set(xlabel=left.get_xlabel(), ylabel="|value − expected| / max|expected|" if relative else "|value − expected|",
                  title="relative error" if relative else "error")
        for ax in (left, right):
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
        fig.tight_layout()
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=100)
        return png.getvalue()
    finally:
        plt.close(fig)


def render_check_figures(expr: Expressions) -> list[tuple[str, bytes]]:
    """(check name, PNG) for every check with "plot" that applies to at least one case."""
    return [(c["name"], render_check_figure(expr, c)) for c in expr.spec.get("checks", [])
            if c.get("plot") and expr.case_names(c)]
