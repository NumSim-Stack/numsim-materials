import unittest

import numpy as np

from numsim_report.data import CaseData
from numsim_report.expressions import Expressions, evaluate_checks


def case(name, **columns):
    n = len(next(iter(columns.values())))
    return CaseData(name, {"step": np.arange(n), **{k: np.asarray(v, float) for k, v in columns.items()}}, {})


SPEC = {"parameters": {"E": 2.0}, "derived": {"E2": "2 * E"}, "cases": [{"name": "a"}, {"name": "b"}]}


class ExpressionTest(unittest.TestCase):
    def test_parameters_derived_and_columns(self):
        expr = Expressions(SPEC, {"a": case("a", x=[0, 1, 2])})
        self.assertEqual(expr("E2"), 4.0)
        np.testing.assert_array_equal(expr("E * x", "a"), [0, 2, 4])
        self.assertEqual(expr(3), 3)

    def test_derived_values_resolve_in_any_order(self):
        spec = {"parameters": {"K": 2.0}, "derived": {"A": "B + 1", "B": "C * 2", "C": "K"}}
        expr = Expressions(spec, {})
        self.assertEqual(expr("A"), 5.0)
        with self.assertRaises(NameError):
            Expressions({"derived": {"A": "B", "B": "A"}}, {})

    def test_no_builtins_beyond_the_whitelist(self):
        with self.assertRaises(NameError):
            Expressions(SPEC, {})("open('x')")

    def test_case_selection(self):
        expr = Expressions(SPEC, {"a": case("a", x=[0]), "b": case("b", x=[0])})
        self.assertEqual(expr.case_names({}), ["a", "b"])
        self.assertEqual(expr.case_names({"case": "b"}), ["b"])
        self.assertEqual(expr.case_names({"cases": ["a", "missing"]}), ["a"])


class CheckTest(unittest.TestCase):
    def run_check(self, check, **columns):
        spec = {**SPEC, "checks": [{"name": "c", "case": "a", **check}]}
        return evaluate_checks(Expressions(spec, {"a": case("a", **columns)}))[0]

    def test_absolute_error_skips_the_unloaded_step(self):
        r = self.run_check({"value": "x", "expected": "E * y", "tolerance": 0.1}, x=[99, 2, 4.05], y=[0, 1, 2])
        self.assertAlmostEqual(r.error, 0.05)
        self.assertTrue(r.passed)

    def test_relative_error(self):
        r = self.run_check({"value": "x", "expected": "y", "tolerance": 0.01, "relative": True}, x=[0, 10, 21], y=[0, 10, 20])
        self.assertAlmostEqual(r.error, 0.05)
        self.assertFalse(r.passed)

    def test_where_restricts_the_steps(self):
        r = self.run_check({"value": "x", "where": "y < 2", "tolerance": 0}, x=[0, 0, 5], y=[0, 1, 2])
        self.assertTrue(r.passed)

    def test_nan_fails(self):
        self.assertFalse(self.run_check({"value": "x", "tolerance": 1}, x=[0, float("nan")]).passed)


class KeyTest(unittest.TestCase):
    def test_unknown_keys_of_checks_and_figures_are_named(self):
        from numsim_report.expressions import check_keys
        good = {"checks": [{"name": "c", "cases": "*", "value": "x", "expected": 0, "tolerance": 1, "relative": True,
                            "where": "x > 0", "plot": {"x": "step", "y": "x", "xlabel": "s", "ylabel": "y"}}],
                "figures": [{"title": "f", "text": "t", "columns": 1, "panels": [
                    {"title": "p", "kind": "line", "case": "a", "x": "x", "y": [{"expr": "y", "label": "l"}], "logy": True,
                     "xlabel": "x", "ylabel": "y", "reference": [{"x": "x", "y": "y", "label": "r"}]}]}]}
        check_keys(good)
        for spec, where in (({"checks": [{"name": "c", "relatve": True}]}, "checks[0] 'c'"),
                            ({"checks": [{"name": "c", "plot": {"xlable": "s"}}]}, "checks[0] 'c'.plot"),
                            ({"figures": [{"title": "f", "pannels": []}]}, "figures[0]"),
                            ({"figures": [{"panels": [{"x": "x", "y": "y", "logx": True}]}]}, "figures[0].panels[0]"),
                            ({"figures": [{"panels": [{"x": "x", "y": [{"expr": "y", "lable": "l"}]}]}]}, "figures[0].panels[0].y[0]"),
                            ({"figures": [{"panels": [{"x": "x", "y": "y", "reference": [{"y": "y", "z": 1}]}]}]},
                             "figures[0].panels[0].reference[0]")):
            with self.subTest(where=where), self.assertRaises(ValueError) as e:
                check_keys(spec)
            self.assertIn(where, str(e.exception))
