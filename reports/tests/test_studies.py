import unittest

import numpy as np

from numsim_report.data import COMPONENTS, CaseData, with_invariants
from numsim_report.expressions import Expressions
from numsim_report.studies import evaluate_studies, fit_order


def run(n, final):
    cols = {"step": np.arange(n + 1), **{"strain_" + c: np.zeros(n + 1) for c in COMPONENTS},
            **{"stress_" + c: np.zeros(n + 1) for c in COMPONENTS}}
    cols["stress_11"] = np.linspace(0, final, n + 1)
    return CaseData("u", with_invariants(cols), {})


class StudyTest(unittest.TestCase):
    def test_fits_the_order_from_errors_against_the_finest(self):
        # first-order scheme: value = exact + 1 / n
        self.assertAlmostEqual(fit_order([10, 20, 40], [0.1, 0.05, 0.025]), 1.0, places=6)

    def test_evaluates_a_study(self):
        spec = {"cases": [{"name": "u", "path": [0, 1], "scale": 0.01}],  # final = 100 + 1 / n: first order
                "studies": [{"name": "s", "case": "u", "steps": [10, 20, 40, 80], "quantity": "stress_11", "min_order": 0.9}]}
        runs = {"s": {n: run(n, 100 + 1.0 / n) for n in (10, 20, 40, 80)}}
        [res] = evaluate_studies(Expressions(spec, {}), runs)
        self.assertEqual(res.name, "s")
        self.assertEqual([r.steps for r in res.rows], [10, 20, 40])  # the finest is the reference
        self.assertAlmostEqual(res.rows[0].error, 0.1 - 1 / 80)
        self.assertAlmostEqual(res.rows[0].increment, 0.1)
        self.assertGreater(res.order, 0.9)
        self.assertTrue(res.passed)

    def test_round_off_errors_count_as_exact(self):
        spec = {"cases": [{"name": "u", "path": [0, 1]}], "studies": [{"name": "s", "case": "u", "steps": [10, 20], "quantity": "stress_11"}]}
        [res] = evaluate_studies(Expressions(spec, {}), {"s": {10: run(10, 1.0), 20: run(20, 1.0)}})
        self.assertTrue(res.exact)
        self.assertTrue(res.passed)
