import unittest

from convert_uvwmaterial_card import convert

ELASTIC = {"name": "linear_elasticity", "parameter": {"E": 1800, "nu": 0.45}}
MORI_TANAKA = {"name": "small_strain_mean_field_composite", "kernal": "mori_tanaka",
               "inclusions": {"inclusion1": {"material": {"name": "linear_elasticity", "parameter": {"E": 2100, "nu": 0.3}},
                                             "name": "inclusion", "volume_fraction": 0.5,
                                             "eshelby_tensor": {"type": "sphere", "parameter": {"nu": 0.45}}}},
               "matrix": {"material": ELASTIC, "volume_fraction": 0.5}}
SHORT_FIBRE = {"name": "small_strain_short_fibre_composite", "integration_precision": 7,
               "fibre_orientation_tensor": [1 / 3, 0, 0, 0, 1 / 3, 0, 0, 0, 1 / 3],
               "composite": {**MORI_TANAKA, "inclusions": {"inclusion1": {**MORI_TANAKA["inclusions"]["inclusion1"],
                             "eshelby_tensor": {"type": "ellipsoid", "parameter": {"nu": 0.45, "aspec_ratio": 27.0}}}}}}
DAMAGE_ROM = {"name": "small_strain_rule_of_mixture_composite", "kernal": "vrh",
              "materials": {"material1": {"name": "small_strain_isotropic_damage", "base_material": ELASTIC,
                                          "state_function": {"name": "von_mises_strain"},
                                          "yield_function": {"name": "strain_based_damage", "parameter": {"crit_val": 0.01}},
                                          "propagation_law": {"name": "damage_exponential2", "parameter": {"eps0": 0.01, "epsf": 0.02}},
                                          "volume_fraction": 0.5},
                            "material2": {**ELASTIC, "volume_fraction": 0.5}}}


def by_name(doc, name):
    return next(m for m in doc["materials"] if m["name"] == name)


class ConvertTest(unittest.TestCase):
    def test_linear_elasticity_moduli(self):
        doc = convert(ELASTIC)
        m = by_name(doc, "material")
        self.assertEqual(m["type"], "linear_elasticity")
        self.assertAlmostEqual(m["K"], 1800 / (3 * (1 - 0.9)))
        self.assertAlmostEqual(m["G"], 1800 / (2 * 1.45))
        self.assertEqual(doc["point"], {"strain_source": "strain", "stress_source": "material"})

    def test_mori_tanaka_card(self):
        doc = convert(MORI_TANAKA, "mt")
        top = by_name(doc, "mt")
        self.assertEqual(top["type"], "mean_field_composite")
        self.assertEqual(top["kernel"], "mori_tanaka")
        inc = by_name(doc, top["inclusions"][0])
        self.assertEqual((inc["type"], inc["shape"], inc["volume_fraction"]), ("inclusion", "sphere", 0.5))
        self.assertEqual(by_name(doc, inc["material"])["type"], "linear_elasticity")
        self.assertEqual(by_name(doc, top["matrix"])["type"], "linear_elasticity")

    def test_short_fibre_card(self):
        top = by_name(convert(SHORT_FIBRE, "sf"), "sf")
        self.assertEqual(top["type"], "short_fibre_composite")
        self.assertEqual(top["aspect_ratio"], 27.0)
        self.assertEqual(top["precision"], 7)
        self.assertAlmostEqual(sum(top["orientation"][:3]), 1.0)

    def test_damage_in_a_rule_of_mixture(self):
        doc = convert(DAMAGE_ROM, "rom")
        top = by_name(doc, "rom")
        self.assertEqual((top["type"], top["kernel"], top["fractions"]), ("rule_of_mixture", "hill", [0.5, 0.5]))
        damaged = by_name(doc, top["phases"][0])
        self.assertEqual(damaged["type"], "isotropic_damage")
        law = by_name(doc, damaged["damage_source"])
        self.assertEqual(law["type"], "exponential_damage_law")
        self.assertAlmostEqual(law["beta"], 100.0)
        self.assertEqual(by_name(doc, damaged["yield_source"])["kappa_critical"], 0.01)

    def test_unported_types_name_their_issue(self):
        with self.assertRaisesRegex(ValueError, "#116"):
            convert({"name": "small_strain_mean_field_composite_history"})
        with self.assertRaisesRegex(ValueError, "unknown"):
            convert({"name": "no_such_thing"})
