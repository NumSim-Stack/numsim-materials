import unittest

from numsim_report import html as h
from numsim_report.expressions import CheckResult, Expressions

SPEC = {"parameters": {"K": 2.0, "steps": 5}, "derived": {"E": "3 * K"},
        "parameter_usage": {"K": ["model.materials[1].K", "cases[0].scale", "checks[0].tolerance"], "steps": []},
        "model": {"materials": [{"name": "strain"}, {"name": "elastic"}]},
        "cases": [{"name": "uniaxial_strain"}], "checks": [{"name": "tangent"}]}


class TableTest(unittest.TestCase):
    def test_escapes_text_but_not_raw(self):
        out = h.table(["a"], [["<b>&", h.Raw("<i>x</i>")]])
        self.assertIn("&lt;b&gt;&amp;", out)
        self.assertIn("<i>x</i>", out)


class ParameterTest(unittest.TestCase):
    def test_usage_names_material_case_and_check(self):
        rows = h.parameter_rows(SPEC)
        self.assertEqual(rows[0], ("K", "2.0", ["elastic.K", "uniaxial_strain.scale", "check 'tangent'.tolerance"]))
        self.assertEqual(rows[1], ("steps", "5", []))

    def test_derived_rows(self):
        self.assertEqual(h.derived_rows(SPEC, Expressions(SPEC, {})), [("E", "3 * K", "6")])


class SectionTest(unittest.TestCase):
    def test_check_table_groups_by_name(self):
        rows = [CheckResult("c", "a", 1e-9, 1e-6), CheckResult("c", "b", 1e-3, 1e-6)]
        out = h.check_section(rows)
        self.assertEqual(out.count("<td>c</td>"), 1)
        self.assertIn("FAIL (b)", out)

    def test_check_plots_follow_the_table(self):
        out = h.check_section([CheckResult("c", "a", 0.0, 1.0)], [("c", b"\x89PNG")])
        self.assertLess(out.index("<table>"), out.index("<img"))
        self.assertIn("<h3>c</h3>", out)

    def test_index_links_every_report(self):
        page = h.index_page([h.ReportSummary("a", 3, 3, True), h.ReportSummary("b", 1, 2, False)])
        self.assertIn('href="a.html"', page)
        self.assertIn("1/2", page)
        self.assertFalse(h.ReportSummary("b", 2, 2, False).ok)

    def test_figures_are_embedded(self):
        self.assertTrue(h.data_uri(b"\x89PNG").startswith("data:image/png;base64,"))
