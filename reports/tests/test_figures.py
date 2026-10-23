import unittest

import numpy as np

from numsim_report import figures
from numsim_report.data import COMPONENTS, CaseData, with_invariants
from numsim_report.expressions import Expressions


def case(name, n=5):
    cols = {"step": np.arange(n), "iterations": np.ones(n), **{"strain_" + c: np.zeros(n) for c in COMPONENTS},
            **{"stress_" + c: np.zeros(n) for c in COMPONENTS}}
    cols["strain_11"] = np.linspace(0, 1e-2, n)
    cols["stress_11"] = 200.0 * cols["strain_11"]
    residuals = {"step": np.array([1, 1, 2, 2]), "iteration": np.array([0, 1, 0, 1]), "residual": np.array([1, 1e-8, 1, 1e-9])}
    return CaseData(name, with_invariants(cols), residuals)


class PanelTest(unittest.TestCase):
    def test_builtin_kinds_are_registered(self):
        self.assertEqual(set(figures.PANELS), {"line", "residuals", "iterations"})

    def test_renders_every_kind_to_png(self):
        spec = {"cases": [{"name": "a"}], "parameters": {"E": 200.0}}
        expr = Expressions(spec, {"a": case("a")})
        png = figures.render_figure(expr, {"panels": [
            {"case": "a", "x": "strain_11", "y": ["stress_11"], "reference": [{"y": "E * strain_11"}]},
            {"case": "a", "kind": "residuals"},
            {"case": "a", "kind": "iterations"},
            {"case": "a", "x": "step", "y": [{"expr": "step + 1", "label": "s"}], "logy": True}]})
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_single_curve_labels_the_y_axis(self):
        expr = Expressions({"cases": [{"name": "a"}]}, {"a": case("a")})
        fig, ax = figures.plt.subplots()
        figures.line_panel(expr, ax, {"case": "a", "x": "strain_11", "y": ["stress_11"]}, ["a"])
        self.assertEqual(ax.get_ylabel(), "σ₁₁")
        figures.plt.close(fig)

    def test_check_figure_shows_value_expected_and_error(self):
        spec = {"cases": [{"name": "a"}], "parameters": {"E": 200.0},
                "checks": [{"name": "hooke", "case": "a", "value": "stress_11", "expected": "E * strain_11",
                            "tolerance": 1e-9, "relative": True, "plot": {"x": "strain_11"}}]}
        expr = Expressions(spec, {"a": case("a")})
        plots = figures.render_check_figures(expr)
        self.assertEqual([name for name, _ in plots], ["hooke"])
        self.assertTrue(plots[0][1].startswith(b"\x89PNG"))
        self.assertEqual(figures.render_check_figures(Expressions({"cases": [], "checks": [{"name": "x", "value": "0", "tolerance": 1}]}, {})), [])

    def test_unknown_kind_is_an_error(self):
        expr = Expressions({"cases": [{"name": "a"}]}, {"a": case("a")})
        with self.assertRaises(ValueError):
            figures.render_figure(expr, {"panels": [{"case": "a", "kind": "pie"}]})
