import pathlib
import tempfile
import unittest

import numpy as np

from numsim_report import data


class CsvTest(unittest.TestCase):
    def test_reads_columns_as_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "a.csv"
            p.write_text("step,x\n0,1.5\n1,nan\n")
            cols = data.read_csv(p)
        np.testing.assert_array_equal(cols["step"], [0, 1])
        self.assertTrue(np.isnan(cols["x"][1]))

    def test_invariants(self):
        cols = {"stress_" + c: np.zeros(1) for c in data.COMPONENTS}
        cols["stress_11"] = np.array([3.0])
        data.with_invariants(cols)
        self.assertAlmostEqual(cols["p"][0], 1.0)
        self.assertAlmostEqual(cols["q"][0], 3.0)


class LabelTest(unittest.TestCase):
    def test_tensor_components_become_symbols(self):
        self.assertEqual(data.pretty_label("strain_11"), "ε₁₁")
        self.assertEqual(data.pretty_label("stress_12"), "σ₁₂")
        self.assertEqual(data.pretty_label("eps_p_22"), "eps_p_22")
        self.assertEqual(data.pretty_label("q"), "q")


FAKE_TOOL = """#!/bin/sh
# what material_report writes for cases c (stopped at step 1), ok and e (no step recorded), then exit {code}
echo '{{"cases": [{{"name": "c"}}, {{"name": "ok"}}, {{"name": "e"}}]}}' > "$2/resolved.json"
echo '[]' > "$2/graph.json"
head='step,converged,stress_11,stress_22,stress_33,stress_12,stress_13,stress_23'
printf "$head\\n0,1,0,0,0,0,0,0\\n1,0,nan,nan,nan,nan,nan,nan\\n" > "$2/c.csv"
printf "$head\\n0,1,0,0,0,0,0,0\\n1,1,3,0,0,0,0,0\\n" > "$2/ok.csv"
echo "$head" > "$2/e.csv"
for s in c ok e; do echo 'step,iteration,residual' > "$2/${{s}}_iterations.csv"; done
exit {code}
"""


class RunToolTest(unittest.TestCase):
    def run_fake(self, code: int) -> data.RunOutput:
        with tempfile.TemporaryDirectory() as tmp:
            tool = pathlib.Path(tmp) / "material_report"
            tool.write_text(FAKE_TOOL.format(code=code))
            tool.chmod(0o755)
            return data.run_tool(str(tool), pathlib.Path(tmp) / "report.json")

    def test_a_case_that_did_not_converge_loads_what_it_recorded(self):
        run = self.run_fake(1)
        self.assertFalse(run.converged)
        self.assertEqual(sorted(run.cases), ["c", "ok"], "e recorded no step")
        np.testing.assert_array_equal(run.cases["c"].columns["converged"], [1, 0])
        self.assertAlmostEqual(run.cases["ok"].columns["q"][1], 3.0)

    def test_input_errors_and_other_exit_codes_raise(self):
        for code in (2, 3):
            with self.subTest(code=code), self.assertRaisesRegex(data.ToolError, f"exit code {code}"):
                self.run_fake(code)
