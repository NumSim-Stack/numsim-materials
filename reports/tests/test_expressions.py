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
