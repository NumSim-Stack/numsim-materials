#!/usr/bin/env python3
"""Converts a uvwmaterial material card (nested JSON, positional "parameter" blocks) into a
numsim-materials json_model document ({"materials": [...]}, a flat graph of named materials).

usage: convert_uvwmaterial_card.py <card.json> [--strain-source strain] [--prefix name]

Supported: linear_elasticity, linear_elastic_incremental, saint_venant_kirchhoff,
small_strain_rule_of_mixture_composite, small_strain_mean_field_composite,
small_strain_short_fibre_composite, small_strain_isotropic_damage (with its state function, yield
function and propagation law). Anything else raises with the issue of the port that covers it.
See docs/uvwmaterial-mapping.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

UNSUPPORTED = {
    "small_strain_rule_of_mixture_history": "#117", "small_strain_mean_field_composite_history": "#116",
    "small_strain_mean_field_composite_plastic": "#116", "small_strain_mean_field_composite_damage_nonlocal": "#121",
    "small_strain_isotropic_nonlocal_damage": "#121", "small_strain_short_fibre_history_composite": "#118",
    "small_strain_short_fibre_damage_nonlocal": "#121", "small_strain_short_fibre_interface_damage_nonlocal": "#121",
    "small_strain_plasticity_single_function": "#120", "small_strain_incremental_plasticity_single_function": "#120",
    "incremental_saint_venant_kirchhoff": "#113", "hencky_strains_linear_elastic": "#113",
    "hencky_strains_linear_thermo_elastic": "#113", "hencky_strains_linear_thermo_chemo_elastic": "#113",
    "finite_strain_mean_field_composite": "#122", "finite_strain_mean_field_composite_explicit": "#122",
    "finite_strain_mean_field_composite_elastic": "#122", "linear_isotropic_conductivity": "#109",
    "conductivity_mean_field_composite": "#110", "gamm2022": "#123", "neo_hookean_incompressible": "#112",
}
KERNELS = {"voigt": "voigt", "reuss": "reuss", "vrh": "hill", "dilute": "dilute", "mori_tanaka": "mori_tanaka", "scs": "self_consistent"}
STATE_FUNCTIONS = {"von_mises_strain": "von_mises_state_function", "strain_energy_state_function": "strain_energy_state_function",
                   "vector_strain_state_function": "vector_strain_state_function"}


def moduli(parameter: dict) -> dict:
    """E, nu -> K, G (numsim-materials takes the bulk and shear modulus)."""
    if "K" in parameter and "G" in parameter:
        return {"K": parameter["K"], "G": parameter["G"]}
    E, nu = parameter["E"], parameter["nu"]
    return {"K": E / (3 * (1 - 2 * nu)), "G": E / (2 * (1 + nu))}


class Converter:
    def __init__(self, strain_source: str = "strain"):
        self.strain_source = strain_source
        self.materials: list[dict] = [{"type": "external_strain_source", "name": strain_source}]
        self.names: set[str] = {strain_source}

    def unique(self, name: str) -> str:
        candidate, n = name, 1
        while candidate in self.names:
            n += 1
            candidate = f"{name}_{n}"
        self.names.add(candidate)
        return candidate

    def add(self, material: dict) -> str:
        material["name"] = self.unique(material["name"])
        self.materials.append(material)
        return material["name"]

    def convert(self, card: dict, name: str) -> str:
        """Adds the card's graph and returns the name of the material producing its stress and tangent."""
        kind = card["name"]
        if kind in UNSUPPORTED:
            raise ValueError(f"{kind}: not ported yet (see {UNSUPPORTED[kind]} of the uvwmaterial port)")
        handler = getattr(self, "convert_" + kind, None)
        if handler is None:
            raise ValueError(f"{kind}: unknown uvwmaterial type")
        return handler(card, name)

    def convert_linear_elasticity(self, card, name):
        return self.add({"type": "linear_elasticity", "name": name, "strain_producer_name": self.strain_source, **moduli(card["parameter"])})

    def convert_linear_elastic_incremental(self, card, name):
        total = self.add({"type": "linear_elasticity", "name": name + "_tangent", "strain_producer_name": self.strain_source, **moduli(card["parameter"])})
        return self.add({"type": "incremental_linear_elasticity", "name": name, "strain_source": self.strain_source, "tangent_source": total})

    def convert_saint_venant_kirchhoff(self, card, name):
        raise ValueError("saint_venant_kirchhoff: a finite-strain card needs an external_deformation_gradient_source; "
                         "write {\"type\": \"saint_venant_kirchhoff\", \"deformation_gradient_source\": ..., \"K\": ..., \"G\": ...} by hand")

    def convert_small_strain_rule_of_mixture_composite(self, card, name):
        phases, fractions = [], []
        for key, phase in card["materials"].items():
            phases.append(self.convert(phase, f"{name}_{key}"))
            fractions.append(phase["volume_fraction"])
        return self.add({"type": "rule_of_mixture", "name": name, "phases": phases, "fractions": fractions,
                         "kernel": KERNELS[card["kernal"]], "strain_source": self.strain_source})

    def inclusion(self, key: str, spec: dict, name: str) -> str:
        material = self.convert(spec["material"], f"{name}_{key}_material")
        es = spec["eshelby_tensor"]
        kind, parameter = es["type"], es.get("parameter", {})
        inc = {"type": "inclusion", "name": f"{name}_{key}", "material": material, "volume_fraction": spec["volume_fraction"]}
        if kind == "sphere":
            inc["shape"] = "sphere"
        elif kind in ("long_fibre_x1", "long_fibre_x2", "long_fibre_x3", "long_fibre"):
            inc["shape"] = "cylinder"
            inc["direction"] = parameter.get("direction_vector", {"long_fibre_x1": [1, 0, 0], "long_fibre_x2": [0, 1, 0], "long_fibre_x3": [0, 0, 1]}.get(kind, [1, 0, 0]))
        elif kind in ("ellipsoid", "general"):
            inc["shape"] = "spheroid"
            inc["aspect_ratio"] = parameter["aspec_ratio"]
            inc["direction"] = parameter.get("direction_vector", [1, 0, 0])
        else:
            raise ValueError(f"eshelby_tensor type {kind}: not supported ({'#122' if 'finite' in kind else '#110'})")
        # the matrix Poisson ratio of the card's Eshelby tensor is taken from the matrix material itself
        return self.add(inc)

    def convert_small_strain_mean_field_composite(self, card, name):
        matrix = self.convert(card["matrix"]["material"], f"{name}_matrix")
        inclusions = [self.inclusion(key, spec, name) for key, spec in card["inclusions"].items()]
        return self.add({"type": "mean_field_composite", "name": name, "matrix": matrix, "inclusions": inclusions,
                         "kernel": KERNELS[card["kernal"]], "strain_source": self.strain_source})

    def convert_small_strain_short_fibre_composite(self, card, name):
        composite = card["composite"]
        if len(composite["inclusions"]) != 1:
            raise ValueError("short fibre composite: one fibre phase expected")
        [(key, spec)] = composite["inclusions"].items()
        matrix = self.convert(composite["matrix"]["material"], f"{name}_matrix")
        fibre = self.convert(spec["material"], f"{name}_fibre")
        a = card["fibre_orientation_tensor"]  # 9 components row-major
        orientation = [a[0], a[4], a[8], a[1], a[2], a[5]]
        return self.add({"type": "short_fibre_composite", "name": name, "matrix": matrix, "fibre": fibre,
                         "volume_fraction": spec["volume_fraction"], "aspect_ratio": spec["eshelby_tensor"]["parameter"]["aspec_ratio"],
                         "orientation": orientation, "precision": card.get("integration_precision", 15),
                         "strain_source": self.strain_source})

    def propagation_law(self, law: dict, yield_name: str, name: str) -> str:
        kind, p = law["name"], law.get("parameter", {})
        if kind == "linear":
            return self.add({"type": "linear_damage_law", "name": name, "yield_source": yield_name, "K": p["K"]})
        if kind == "power_law":
            return self.add({"type": "power_damage_law", "name": name, "yield_source": yield_name, "K": p["K"], "m": p["m"]})
        if kind == "exponential_law":
            return self.add({"type": "saturation_damage_law", "name": name, "yield_source": yield_name, "K": p["K"], "m": p["m"]})
        if kind == "damage_exponential2":
            if p.get("constant", 1.0) != 1.0:
                raise ValueError("damage_exponential2: the scale 'constant' is not supported")
            return self.add({"type": "exponential_damage_law", "name": name, "yield_source": yield_name,
                             "kappa_0": p["eps0"], "beta": 1.0 / (p["epsf"] - p["eps0"])})
        if kind == "damage_exponential3":
            return self.add({"type": "peerlings_damage_law", "name": name, "yield_source": yield_name,
                             "kappa_0": p["kappa0"], "alpha": p["alpha"], "beta": p["beta"]})
        raise ValueError(f"propagation law {kind}: not supported (damage_exponential1 was not carried over, #119)")

    def convert_small_strain_isotropic_damage(self, card, name):
        elastic = self.convert(card["base_material"], f"{name}_elastic")
        sf = card["state_function"]
        if sf["name"] not in STATE_FUNCTIONS:
            raise ValueError(f"state function {sf['name']}: not supported (#119)")
        state = self.add({"type": STATE_FUNCTIONS[sf["name"]], "name": f"{name}_state", "strain_source": self.strain_source,
                          **({"m": sf["parameter"]["m"], "n": sf["parameter"]["n"]} if sf["name"] == "vector_strain_state_function" else {})})
        yf = card["yield_function"]
        if yf["name"] != "strain_based_damage":
            raise ValueError(f"yield function {yf['name']}: not supported")
        yield_name = self.add({"type": "strain_threshold_yield", "name": f"{name}_yield", "state_source": state,
                               "kappa_critical": yf["parameter"]["crit_val"]})
        law = self.propagation_law(card["propagation_law"], yield_name, f"{name}_law")
        return self.add({"type": "isotropic_damage", "name": name, "elastic_source": elastic, "damage_source": law,
                         "state_source": state, "yield_source": yield_name})


def convert(card: dict, name: str = "material", strain_source: str = "strain") -> dict:
    """The json_model document of a card; its top material is named `name` and provides stress/tangent."""
    c = Converter(strain_source)
    top = c.convert(card, name)
    return {"materials": c.materials, "point": {"strain_source": strain_source, "stress_source": top}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("card", type=pathlib.Path)
    p.add_argument("--strain-source", default="strain")
    p.add_argument("--prefix", default="material", help="name of the top material")
    a = p.parse_args()
    try:
        print(json.dumps(convert(json.loads(a.card.read_text()), a.prefix, a.strain_source), indent=2))
    except (ValueError, KeyError) as e:
        print(f"convert_uvwmaterial_card: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
