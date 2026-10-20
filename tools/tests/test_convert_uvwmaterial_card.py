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
DAMAGE = {"name": "small_strain_isotropic_damage", "base_material": ELASTIC,
          "state_function": {"name": "von_mises_strain"},
          "yield_function": {"name": "strain_based_damage", "parameter": {"crit_val": 0.01}},
          "propagation_law": {"name": "damage_exponential2", "parameter": {"eps0": 0.01, "epsf": 0.02}}}
DAMAGE_ROM = {"name": "small_strain_rule_of_mixture_composite", "kernal": "voigt",
              "materials": {"material1": {**DAMAGE, "volume_fraction": 0.5}, "material2": {**ELASTIC, "volume_fraction": 0.5}}}
LINEAR_ROM = {"name": "small_strain_rule_of_mixture_composite", "kernal": "vrh",
              "materials": {"a": {**ELASTIC, "volume_fraction": 0.3}, "b": {**MORI_TANAKA["inclusions"]["inclusion1"]["material"], "volume_fraction": 0.7}}}


def by_name(doc, name):
    return next(m for m in doc["model"]["materials"] if m["name"] == name)


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
        top = by_name(doc, "mt")  # the composite is a tangent producer; the stress is a linear_stress on it
        self.assertEqual((top["type"], top["tangent_source"], top["strain_source"]), ("linear_stress", "mt_tangent", "strain"))
        self.assertEqual(doc["point"], {"strain_source": "strain", "stress_source": "mt", "tangent_source": "mt_tangent"})
        mf = by_name(doc, "mt_tangent")
        self.assertEqual((mf["type"], mf["kernel"]), ("mean_field_composite_global", {"type": "mori_tanaka"}))
        inc = by_name(doc, mf["phases"][1])  # the matrix first
        self.assertEqual(inc["type"], "phase")
        self.assertEqual(by_name(doc, inc["fraction_source"]), {"type": "constant_scalar", "name": inc["name"] + "_c", "value": 0.5})
        self.assertEqual(by_name(doc, inc["tangent_source"])["type"], "isotropic_tangent")
        shape = by_name(doc, inc["eshelby_source"])
        self.assertEqual((shape["type"], shape["nu_property"]), ("eshelby_sphere", "poisson_ratio"))
        matrix = by_name(doc, mf["phases"][0])
        self.assertEqual((matrix["type"], by_name(doc, matrix["fraction_source"])["value"]), ("phase", 0.5))
        self.assertEqual(by_name(doc, shape["nu_source"]), {"type": "isotropic_moduli", "name": matrix["tangent_source"] + "_moduli", "tangent_source": matrix["tangent_source"]})

    def test_self_consistent_card_carries_its_shapes(self):
        doc = convert({**MORI_TANAKA, "kernal": "scs"}, "sc")
        mf = by_name(doc, "sc_tangent")
        matrix, inclusion = mf["phases"]
        # every phase embedded in the effective medium: the shapes are kernel data, no Eshelby node, no residual
        self.assertEqual(mf["kernel"], {"type": "self_consistent", "shapes": {matrix: {"type": "sphere"}, inclusion: {"type": "sphere"}}})
        self.assertNotIn("eshelby_source", by_name(doc, inclusion))
        self.assertFalse(any(m["type"].startswith("eshelby_") for m in doc["model"]["materials"]))
        self.assertEqual(doc["point"], {"strain_source": "strain", "stress_source": "sc", "tangent_source": "sc_tangent"})

    def test_short_fibre_card(self):
        doc = convert(SHORT_FIBRE, "sf")
        self.assertEqual(by_name(doc, "sf")["type"], "linear_stress")
        top = by_name(doc, "sf_tangent")
        self.assertEqual(top["type"], "short_fibre_composite")
        fibres = by_name(doc, top["fibre"])
        self.assertEqual(fibres["type"], "phase")
        shape = by_name(doc, fibres["eshelby_source"])
        matrix = by_name(doc, top["matrix"])
        self.assertEqual((shape["type"], shape["aspect_ratio"], shape["direction"], shape["nu_source"]),
                         ("eshelby_spheroid", 27.0, [1, 0, 0], matrix["tangent_source"] + "_moduli"))
        self.assertEqual(top["precision"], 7)
        self.assertEqual(top["averaging"], "stiffness")
        self.assertAlmostEqual(sum(top["orientation"][:3]), 1.0)
        self.assertAlmostEqual(by_name(doc, matrix["fraction_source"])["value"], 0.5)

    def test_linear_rule_of_mixture_is_a_tangent_and_a_stress(self):
        doc = convert(LINEAR_ROM, "rom")
        self.assertEqual(by_name(doc, "rom")["type"], "linear_stress")
        rom = by_name(doc, "rom_tangent")
        self.assertEqual((rom["type"], rom["kernel"]), ("mean_field_composite_global", {"type": "hill"}))
        self.assertEqual([by_name(doc, by_name(doc, p)["fraction_source"])["value"] for p in rom["phases"]], [0.3, 0.7])
        self.assertEqual(doc["point"]["tangent_source"], "rom_tangent")

    def test_damage_in_a_rule_of_mixture_localises(self):
        for kernal, kernel in (("voigt", "voigt"), ("vrh", "hill")):
            doc = convert({**DAMAGE_ROM, "kernal": kernal}, "rom")
            top = by_name(doc, "rom")
            strain = by_name(doc, "rom_strain")
            self.assertEqual((strain["type"], strain["kernel"], strain["phases"]), ("mean_field_strain", {"type": kernel}, ["rom_material1", "rom_material2"]))
            damage_src, elastic_src = top["phases"]  # per phase the sources by role, the damage material first
            self.assertEqual(damage_src["concentration"], "rom_strain::concentration_rom_material1")
            damaged_name, stress_property = damage_src["stress"].split("::")
            damaged = by_name(doc, damaged_name)
            self.assertEqual((damaged["type"], stress_property, damage_src["tangent"]), ("isotropic_damage", "damaged_stress", f"{damaged_name}::damaged_tangent"))
            state = by_name(doc, damaged["state_source"])  # the chain reads the phase strain
            self.assertEqual((state["strain_source"], state["strain_property"]), ("rom_strain", "strain_rom_material1"))
            self.assertEqual(by_name(doc, damaged["elastic_source"])["strain_property"], "strain_rom_material1")
            elastic_phase = by_name(doc, "rom_material2")
            self.assertEqual(elastic_src["stress"], "rom_material2_stress::stress")
            self.assertEqual(by_name(doc, "rom_material2_stress"), {"type": "linear_stress", "name": "rom_material2_stress", "tangent_source": elastic_phase["tangent_source"],
                                                                    "strain_source": "rom_strain", "strain_property": "strain_rom_material2"})
            damage_phase = by_name(doc, "rom_material1")
            # the phase node reads its material's tangent over a local edge either way: a global one is a cycle
            self.assertEqual((damage_phase["tangent_edge"], damage_phase["initial_tangent_source"]), ("local", "rom_material1_initial_C"))
            self.assertEqual(by_name(doc, "rom_material1_initial_C")["type"], "isotropic_tangent")  # no strain: no cycle
            if kernel == "voigt":  # equal strain: the concentrations do not depend on the tangents
                self.assertEqual(top["type"], "mean_field_composite_local_linear")
                self.assertNotIn("residual", doc["point"])
            else:
                self.assertEqual(top["type"], "mean_field_composite_local_nonlinear")
                self.assertEqual(damage_src["used_tangent"], "rom_material1::tangent")
                self.assertNotIn("used_tangent", elastic_src)
                self.assertEqual(doc["point"]["residual"], "rom::residual")
            law = by_name(doc, damaged["damage_source"])
            self.assertAlmostEqual(law["beta"], 100.0)

    def test_linear_law_and_state_function_parameters(self):
        card = {**DAMAGE, "propagation_law": {"name": "linear", "parameter": {"K": 3.0}},
                "state_function": {"name": "strain_energy_state_function"}}
        doc = convert(card, "d")
        law = by_name(doc, by_name(doc, "d")["damage_source"])
        self.assertEqual((law["type"], law["K"], law["m"]), ("power_damage_law", 3.0, 1.0))
        state = by_name(doc, by_name(doc, "d")["state_source"])
        self.assertEqual((state["type"], state["tangent_source"], state["youngs_modulus"]), ("strain_energy_state_function", "d_elastic", 1800))
        card["state_function"] = {"name": "vector_strain_state_function", "parameter": {"m": [1, 0, 0], "n": [0, 0, 1]}}
        state = by_name(convert(card, "d"), "d_state")
        self.assertEqual((state["m_direction"], state["n_direction"]), ([0], [2]))
        card["state_function"]["parameter"]["m"] = [1, 1, 0]
        with self.assertRaisesRegex(ValueError, "unit axis"):
            convert(card, "d")

    def test_unported_types_name_their_issue(self):
        with self.assertRaisesRegex(ValueError, "#116"):
            convert({"name": "small_strain_mean_field_composite_history"})
        with self.assertRaisesRegex(ValueError, "#116"):
            convert({**MORI_TANAKA, "matrix": {"material": DAMAGE, "volume_fraction": 0.5}}, "mt")
        with self.assertRaisesRegex(ValueError, "unknown"):
            convert({"name": "no_such_thing"})
