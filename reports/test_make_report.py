"""Unit tests of make_report.py that need no material_report run."""
import unittest

import numpy as np

import make_report as mr


def case(name, **columns):
    cols = {"step": np.arange(len(next(iter(columns.values())))), **{k: np.asarray(v, float) for k, v in columns.items()}}
    return mr.CaseData(name, cols, {})


SPEC = {"parameters": {"E": 2.0}, "derived": {"E2": "2 * E"}, "cases": [{"name": "a"}, {"name": "b"}]}


class ExpressionTest(unittest.TestCase):
    def test_parameters_derived_and_columns(self):
        expr = mr.Expressions(SPEC, {"a": case("a", x=[0, 1, 2])})
        self.assertEqual(expr("E2"), 4.0)
        np.testing.assert_array_equal(expr("E * x", "a"), [0, 2, 4])
        self.assertEqual(expr(3), 3)

    def test_no_builtins_beyond_the_whitelist(self):
        expr = mr.Expressions(SPEC, {})
        with self.assertRaises(NameError):
            expr("open('x')")

    def test_case_selection(self):
        expr = mr.Expressions(SPEC, {"a": case("a", x=[0]), "b": case("b", x=[0])})
        self.assertEqual(expr.case_names({}, SPEC), ["a", "b"])
        self.assertEqual(expr.case_names({"case": "b"}, SPEC), ["b"])
        self.assertEqual(expr.case_names({"cases": ["a", "missing"]}, SPEC), ["a"])


class CheckTest(unittest.TestCase):
    def run_check(self, check, **columns):
        spec = {**SPEC, "checks": [{"name": "c", "case": "a", **check}]}
        return mr.evaluate_checks(spec, mr.Expressions(spec, {"a": case("a", **columns)}))[0]

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
        r = self.run_check({"value": "x", "tolerance": 1}, x=[0, np.nan])
        self.assertFalse(r.passed)


class MarkdownTest(unittest.TestCase):
    def test_replaces_only_the_marked_block(self):
        text = f"# t\n\nhand written\n\n{mr.BEGIN}\nold\n{mr.END}\n\ntail\n"
        out = mr.replace_results(text, "new")
        self.assertIn("hand written", out)
        self.assertIn(f"{mr.BEGIN}\nnew\n{mr.END}", out)
        self.assertNotIn("old", out)
        self.assertIn("tail", out)

    def test_appends_the_block_when_missing(self):
        self.assertTrue(mr.replace_results("# t\n", "new").endswith(f"{mr.BEGIN}\nnew\n{mr.END}\n"))

    def test_check_table_groups_by_name(self):
        rows = [mr.CheckResult("c", "a", 1e-9, 1e-6), mr.CheckResult("c", "b", 1e-3, 1e-6)]
        table = mr.check_table(rows)
        self.assertEqual(len([l for l in table if l.startswith("| c |")]), 1)
        self.assertIn("**FAIL** (b)", table[-1])


class PanelRegistryTest(unittest.TestCase):
    def test_builtin_kinds_are_registered(self):
        self.assertEqual(set(mr.PANELS), {"line", "residuals", "iterations"})


if __name__ == "__main__":
    unittest.main()
