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


class HtmlTest(unittest.TestCase):
    SPEC = {"parameters": {"K": 2.0, "steps": 5}, "derived": {"E": "3 * K"},
            "parameter_usage": {"K": ["model.materials[1].K", "cases[0].scale", "checks[0].tolerance"], "steps": []},
            "model": {"materials": [{"name": "strain"}, {"name": "elastic"}]},
            "cases": [{"name": "uniaxial_strain"}], "checks": [{"name": "tangent"}]}

    def test_parameter_rows_name_the_material_and_case(self):
        rows = mr.parameter_rows(self.SPEC)
        self.assertEqual(rows[0], ("K", "2.0", ["elastic.K", "uniaxial_strain.scale", "check 'tangent'.tolerance"]))
        self.assertEqual(rows[1], ("steps", "5", []))

    def test_derived_rows_show_expression_and_value(self):
        expr = mr.Expressions(self.SPEC, {})
        self.assertEqual(mr.derived_rows(self.SPEC, expr), [("E", "3 * K", "6")])

    def test_table_escapes_cells(self):
        html = mr.html_table(["a"], [["<b>&"]])
        self.assertIn("&lt;b&gt;&amp;", html)
        self.assertNotIn("<b>&", html)

    def test_check_table_groups_by_name(self):
        rows = [mr.CheckResult("c", "a", 1e-9, 1e-6), mr.CheckResult("c", "b", 1e-3, 1e-6)]
        html = mr.check_section(rows)
        self.assertEqual(html.count("<td>c</td>"), 1)
        self.assertIn("FAIL (b)", html)

    def test_index_links_every_report(self):
        page = mr.index_page([("a", 3, 3, True), ("b", 1, 2, False)])
        self.assertIn('href="a.html"', page)
        self.assertIn("1/2", page)

    def test_figures_are_embedded(self):
        uri = mr.data_uri(b"\x89PNG")
        self.assertTrue(uri.startswith("data:image/png;base64,"))


class PanelRegistryTest(unittest.TestCase):
    def test_builtin_kinds_are_registered(self):
        self.assertEqual(set(mr.PANELS), {"line", "residuals", "iterations"})


if __name__ == "__main__":
    unittest.main()
