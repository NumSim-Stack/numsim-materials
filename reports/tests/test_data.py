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
