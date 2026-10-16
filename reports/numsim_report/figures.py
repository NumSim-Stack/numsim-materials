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
