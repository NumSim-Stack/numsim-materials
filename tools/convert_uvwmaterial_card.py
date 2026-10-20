#!/usr/bin/env python3
"""Converts a uvwmaterial material card (nested JSON, positional "parameter" blocks) into a
numsim-materials json_model document ({"materials": [...]}, a flat graph of named materials).

usage: convert_uvwmaterial_card.py <card.json> [--strain-source strain] [--prefix name]

Supported: linear_elasticity, linear_elastic_incremental, saint_venant_kirchhoff, hencky_strains_linear_elastic,
small_strain_rule_of_mixture_composite, small_strain_mean_field_composite,
small_strain_short_fibre_composite, small_strain_isotropic_damage (with its state function, yield
function and propagation law). Anything else raises with the issue of the port that covers it.
Composites come out as phase nodes (produced tangents, fractions, Eshelby tensors) feeding a tangent
producer and a linear_stress; the point names the tangent producer (and the residual of a self-consistent
fixed point).
See docs/uvwmaterial-mapping.md.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys

UNSUPPORTED = {
    "small_strain_rule_of_mixture_history": "#117", "small_strain_mean_field_composite_history": "#116",
    "small_strain_mean_field_composite_plastic": "#116", "small_strain_mean_field_composite_damage_nonlocal": "#121",
    "small_strain_isotropic_nonlocal_damage": "#121", "small_strain_short_fibre_history_composite": "#118",
    "small_strain_short_fibre_damage_nonlocal": "#121", "small_strain_short_fibre_interface_damage_nonlocal": "#121",
    "small_strain_plasticity_single_function": "#120", "small_strain_incremental_plasticity_single_function": "#120",
    "incremental_saint_venant_kirchhoff": "#113",
    "finite_strain_mean_field_composite": "#122", "finite_strain_mean_field_composite_explicit": "#122",
    "finite_strain_mean_field_composite_elastic": "#122", "linear_isotropic_conductivity": "#109",
    "conductivity_mean_field_composite": "#110", "gamm2022": "#123", "neo_hookean_incompressible": "#112",
}
KERNELS = {"voigt": "voigt", "reuss": "reuss", "vrh": "hill", "dilute": "dilute", "mori_tanaka": "mori_tanaka", "scs": "self_consistent"}
TANGENT_PROPERTY = {"small_strain_isotropic_damage": "damaged_tangent"}
STRESS_PROPERTY = {"small_strain_isotropic_damage": "damaged_stress"}
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
        self.tangent_sources: dict[str, str] = {}  # top material -> the node producing its tangent, when not itself
        self.residual: str | None = None           # "node::residual" of a fixed point the point has to close
        self.strain_property = "strain"            # the property of strain_source the materials read

    def strain_of(self, key: str = "strain_source") -> dict:
        """The strain wiring of a material: the composite strain, or a phase strain inside local_strain()."""
        return {key: self.strain_source, **({"strain_property": self.strain_property} if self.strain_property != "strain" else {})}

    @contextlib.contextmanager
    def local_strain(self, node: str, prop: str):
        """Materials converted inside read their strain from node::prop (a mean_field_strain output)."""
        saved = (self.strain_source, self.strain_property)
        self.strain_source, self.strain_property = node, prop
        try:
            yield
        finally:
            self.strain_source, self.strain_property = saved

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
        return self.add({"type": "linear_elasticity", "name": name, **self.strain_of("strain_producer_name"), **moduli(card["parameter"])})

    def convert_linear_elastic_incremental(self, card, name):
        total = self.add({"type": "linear_elasticity", "name": name + "_tangent", **self.strain_of("strain_producer_name"), **moduli(card["parameter"])})
        return self.add({"type": "incremental_linear_elasticity", "name": name, **self.strain_of(), "tangent_source": total})

    def convert_saint_venant_kirchhoff(self, card, name):
        raise ValueError("saint_venant_kirchhoff: a finite-strain card needs an external_deformation_gradient_source; "
                         "write {\"type\": \"saint_venant_kirchhoff\", \"deformation_gradient_source\": ..., \"K\": ..., \"G\": ...} by hand")

    def fraction(self, value, name: str) -> str:
        """A volume fraction as a produced property: constant_scalar <name>_c."""
        return self.add({"type": "constant_scalar", "name": f"{name}_c", "value": value})

    def phase_tangent(self, spec: dict, name: str) -> tuple[str, str, bool]:
        """A phase material card as a tangent producer: (node, tangent property, linear). A linear_elasticity
        card becomes constant K, G and an isotropic_tangent; any other card is the converted material at the
        composite strain, named with its tangent property."""
        if spec["name"] == "linear_elasticity":
            kg = moduli(spec["parameter"])
            K = self.add({"type": "constant_scalar", "name": f"{name}_K", "value": kg["K"]})
            G = self.add({"type": "constant_scalar", "name": f"{name}_G", "value": kg["G"]})
            node = self.add({"type": "isotropic_tangent", "name": f"{name}_C", "K_source": K, "G_source": G})
            return node, "tangent", True
        return self.convert(spec, name), TANGENT_PROPERTY.get(spec["name"], "tangent"), False

    def convert_hencky_strains_linear_elastic(self, card, name):
        raise ValueError("hencky_strains_linear_elastic: a finite-strain card needs an external_deformation_gradient_source; "
                         "write {\"type\": \"hencky_elasticity\", \"deformation_gradient_source\": ..., \"tangent_source\": an isotropic_tangent} by hand "
                         "(thermo variants: a thermal_eigenstrain as \"eigenstrain_source\")")

    convert_hencky_strains_linear_thermo_elastic = convert_hencky_strains_linear_elastic
    convert_hencky_strains_linear_thermo_chemo_elastic = convert_hencky_strains_linear_elastic

    def convert_small_strain_rule_of_mixture_composite(self, card, name):
        """Linear phases: mean_field_composite_global tangent + linear_stress. Nonlinear phases: mean_field_strain with the
        kernel localises, every material reads its phase strain, a mean_field_composite_local_* averages; a nonlinear phase
        reads its tangent over a local edge (not for Voigt, whose concentrations need no tangent) and the
        point closes the fixed point."""
        kernel = KERNELS[card["kernal"]]
        if all(spec["name"] == "linear_elasticity" for spec in card["materials"].values()):
            phases = []
            for key, spec in card["materials"].items():
                node, _, _ = self.phase_tangent(spec, f"{name}_{key}_material")
                phases.append(self.add({"type": "phase", "name": f"{name}_{key}", "tangent_source": node,
                                        "fraction_source": self.fraction(spec["volume_fraction"], f"{name}_{key}")}))
            rom = self.add({"type": "mean_field_composite_global", "name": f"{name}_tangent", "phases": phases, "kernel": {"type": kernel}})
            self.tangent_sources[name] = rom
            return self.add({"type": "linear_stress", "name": name, "tangent_source": rom, "strain_source": self.strain_source})
        strain_node = f"{name}_strain"
        phases, sources = [], []
        for key, spec in card["materials"].items():
            phase_name = f"{name}_{key}"
            phase = {"type": "phase", "name": phase_name, "fraction_source": self.fraction(spec["volume_fraction"], phase_name)}
            local = False
            if spec["name"] == "linear_elasticity":
                tangent, _, _ = self.phase_tangent(spec, f"{phase_name}_material")
                stress = self.add({"type": "linear_stress", "name": f"{phase_name}_stress", "tangent_source": tangent,
                                   "strain_source": strain_node, "strain_property": f"strain_{phase_name}"})
                phase["tangent_source"] = tangent
                stress_of, tangent_of = f"{stress}::stress", f"{tangent}::tangent"
            else:
                with self.local_strain(strain_node, f"strain_{phase_name}"):
                    material = self.convert(spec, f"{phase_name}_material")
                tangent_property = TANGENT_PROPERTY.get(spec["name"], "tangent")
                phase |= {"tangent_source": material, "tangent_property": tangent_property}
                # the phase node depends on the material, the material on the localisation: a local edge (previous
                # pass, from the elastic tangent); only a scheme whose concentrations read tangents needs the residual
                base = spec.get("base_material", {})
                if spec["name"] != "small_strain_isotropic_damage" or base.get("name") != "linear_elasticity":
                    raise ValueError(f"rule of mixture: no initial tangent known for {spec['name']} on a local edge; wire the phase by hand")
                initial, _, _ = self.phase_tangent(base, f"{phase_name}_initial")  # strain-independent: no cycle
                phase |= {"tangent_edge": "local", "initial_tangent_source": initial}
                local = kernel != "voigt"
                stress_of = f"{material}::{STRESS_PROPERTY.get(spec['name'], 'stress')}"
                tangent_of = f"{material}::{tangent_property}"
            phase_name = self.add(phase)
            phases.append(phase_name)
            source = {"concentration": f"{strain_node}::concentration_{phase_name}", "stress": stress_of, "tangent": tangent_of,
                      "fraction": f"{phase_name}::volume_fraction"}
            if local:
                source["used_tangent"] = f"{phase_name}::tangent"
            sources.append(source)
        self.add({"type": "mean_field_strain", "name": strain_node, "strain_source": self.strain_source,
                  "phases": phases, "kernel": {"type": kernel}})
        nonlinear = any("used_tangent" in src for src in sources)
        if nonlinear:
            self.residual = f"{name}::residual"
        return self.add({"type": "mean_field_composite_local_nonlinear" if nonlinear else "mean_field_composite_local_linear",
                         "name": name, "phases": sources})

    ESHELBY_DIRECTIONS = {"long_fibre_x1": [1, 0, 0], "long_fibre_x2": [0, 1, 0], "long_fibre_x3": [0, 0, 1]}

    def poisson_ratio(self, tangent: str) -> dict:
        """nu as a produced property (isotropic_moduli of a tangent node), made once."""
        name = f"{tangent}_moduli"
        if name not in self.names:
            self.add({"type": "isotropic_moduli", "name": name, "tangent_source": tangent})
        return {"nu_source": name, "nu_property": "poisson_ratio"}

    def shape(self, es: dict) -> dict:
        """A uvwmaterial eshelby_tensor as a shape object (the kernel data of the self-consistent scheme)."""
        kind, parameter = es["type"], es.get("parameter", {})
        if kind == "sphere":
            return {"type": "sphere"}
        if kind in ("long_fibre_x1", "long_fibre_x2", "long_fibre_x3", "long_fibre"):
            return {"type": "cylinder", "direction": parameter.get("direction_vector", self.ESHELBY_DIRECTIONS.get(kind, [1, 0, 0]))}
        if kind in ("ellipsoid", "general"):
            return {"type": "spheroid", "aspect_ratio": parameter["aspec_ratio"], "direction": parameter.get("direction_vector", [1, 0, 0])}
        raise ValueError(f"eshelby_tensor type {kind}: not supported ({'#122' if 'finite' in kind else '#110'})")

    def eshelby(self, es: dict, nu: dict, name: str) -> str:
        """One Eshelby node per uvwmaterial eshelby_tensor, consuming the produced Poisson ratio `nu`."""
        shape = self.shape(es)
        return self.add({"type": f"eshelby_{shape.pop('type')}", "name": name, **nu, **shape})

    def linear_phase(self, spec: dict, name: str, what: str) -> str:
        node, _, linear = self.phase_tangent(spec, name)
        if not linear:
            raise ValueError(f"{what}: a nonlinear phase needs the local strains of #116 (mean_field_strain, by hand)")
        return node

    def inclusion(self, key: str, spec: dict, name: str, nu: dict | None) -> tuple[str, float]:
        """A phase; with `nu` its Eshelby node (dilute, Mori-Tanaka), without it the shape is the kernel's."""
        tangent = self.linear_phase(spec["material"], f"{name}_{key}_material", "mean field composite")
        phase = {"type": "phase", "name": f"{name}_{key}", "tangent_source": tangent,
                 "fraction_source": self.fraction(spec["volume_fraction"], f"{name}_{key}")}
        if nu is not None:
            phase["eshelby_source"] = self.eshelby(spec["eshelby_tensor"], nu, f"{name}_{key}_shape")
        return self.add(phase), spec["volume_fraction"]

    def matrix_phase(self, tangent: str, fraction: float, name: str) -> str:
        return self.add({"type": "phase", "name": f"{name}_matrix", "tangent_source": tangent,
                         "fraction_source": self.fraction(fraction, f"{name}_matrix")})

    def stress_of(self, tangent: str, name: str) -> str:
        """The composite is a tangent producer: its stress is a linear_stress, the point reads the tangent there."""
        self.tangent_sources[name] = tangent
        return self.add({"type": "linear_stress", "name": name, "tangent_source": tangent, "strain_source": self.strain_source})

    def convert_small_strain_mean_field_composite(self, card, name):
        kernel = KERNELS[card["kernal"]]
        matrix = self.linear_phase(card["matrix"]["material"], f"{name}_matrix_material", "mean field composite")
        sc = kernel == "self_consistent"
        inclusions = [self.inclusion(key, spec, name, None if sc else self.poisson_ratio(matrix)) for key, spec in card["inclusions"].items()]
        matrix_phase = self.matrix_phase(matrix, 1 - sum(c for _, c in inclusions), name)
        kernel_doc = {"type": kernel}
        if sc:  # every phase embedded in the effective medium: the shapes are data of the kernel
            kernel_doc["shapes"] = {matrix_phase: {"type": "sphere"}} | {
                phase: self.shape(spec["eshelby_tensor"]) for (phase, _), spec in zip(inclusions, card["inclusions"].values())}
        tangent = self.add({"type": "mean_field_composite_global", "name": f"{name}_tangent",
                            "phases": [matrix_phase] + [p for p, _ in inclusions], "kernel": kernel_doc})
        return self.stress_of(tangent, name)

    def convert_small_strain_short_fibre_composite(self, card, name):
        composite = card["composite"]
        if len(composite["inclusions"]) != 1:
            raise ValueError("short fibre composite: one fibre phase expected")
        [(key, spec)] = composite["inclusions"].items()
        matrix = self.linear_phase(composite["matrix"]["material"], f"{name}_matrix_material", "short fibre composite")
        fibre = self.linear_phase(spec["material"], f"{name}_fibre_material", "short fibre composite")
        shape = self.add({"type": "eshelby_spheroid", "name": f"{name}_shape", **self.poisson_ratio(matrix),
                          "aspect_ratio": spec["eshelby_tensor"]["parameter"]["aspec_ratio"], "direction": [1, 0, 0]})
        phase = self.add({"type": "phase", "name": f"{name}_fibres", "tangent_source": fibre, "eshelby_source": shape,
                          "fraction_source": self.fraction(spec["volume_fraction"], f"{name}_fibres")})
        matrix_phase = self.matrix_phase(matrix, 1 - spec["volume_fraction"], name)
        a = card["fibre_orientation_tensor"]  # 9 components row-major
        orientation = [a[0], a[4], a[8], a[1], a[2], a[5]]
        tangent = self.add({"type": "short_fibre_composite", "name": f"{name}_tangent", "matrix": matrix_phase, "fibre": phase,
                            "orientation": orientation, "precision": card.get("integration_precision", 23),
                            "averaging": "stiffness"})  # uvwmaterial: Voigt over the unidirectional composites
        return self.stress_of(tangent, name)

    def propagation_law(self, law: dict, yield_name: str, name: str) -> str:
        kind, p = law["name"], law.get("parameter", {})
        if kind == "linear":  # D = K kappa: the power law with m = 1 (linear_damage_law here is the kappa_0 .. kappa_f ramp)
            return self.add({"type": "power_damage_law", "name": name, "yield_source": yield_name, "K": p["K"], "m": 1.0})
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

    @staticmethod
    def axis_index(v: list, what: str) -> list:
        """A unit axis vector as the index list numsim-materials takes."""
        nonzero = [i for i, x in enumerate(v) if x != 0]
        if len(nonzero) != 1 or abs(v[nonzero[0]]) != 1:
            raise ValueError(f"vector_strain_state_function: {what} must be a unit axis vector, got {v}")
        return [nonzero[0]]

    def state_function_parameters(self, sf: dict, elastic: str, base: dict) -> dict:
        if sf["name"] == "vector_strain_state_function":
            return {"m_direction": self.axis_index(sf["parameter"]["m"], "m"), "n_direction": self.axis_index(sf["parameter"]["n"], "n")}
        if sf["name"] == "strain_energy_state_function":
            kg = moduli(base)
            return {"tangent_source": elastic, "youngs_modulus": base.get("E", 9 * kg["K"] * kg["G"] / (3 * kg["K"] + kg["G"]))}
        return {}

    def convert_small_strain_isotropic_damage(self, card, name):
        elastic = self.convert(card["base_material"], f"{name}_elastic")
        sf = card["state_function"]
        if sf["name"] not in STATE_FUNCTIONS:
            raise ValueError(f"state function {sf['name']}: not supported (#119)")
        state = self.add({"type": STATE_FUNCTIONS[sf["name"]], "name": f"{name}_state", **self.strain_of(),
                          **self.state_function_parameters(sf, elastic, card["base_material"].get("parameter", {}))})
        yf = card["yield_function"]
        if yf["name"] != "strain_based_damage":
            raise ValueError(f"yield function {yf['name']}: not supported")
        yield_name = self.add({"type": "strain_threshold_yield", "name": f"{name}_yield", "state_source": state,
                               "kappa_critical": yf["parameter"]["crit_val"]})
        law = self.propagation_law(card["propagation_law"], yield_name, f"{name}_law")
        return self.add({"type": "isotropic_damage", "name": name, "elastic_source": elastic, "damage_source": law,
                         "state_source": state, "yield_source": yield_name})


def convert(card: dict, name: str = "material", strain_source: str = "strain") -> dict:
    """The report document of a card ("model" is the json_model document, "point" its evaluation point);
    the top material is named `name` and provides stress and tangent."""
    c = Converter(strain_source)
    top = c.convert(card, name)
    point = {"strain_source": strain_source, "stress_source": top}
    if top in c.tangent_sources:
        point["tangent_source"] = c.tangent_sources[top]
    if c.residual:
        point |= {"residual": c.residual, "tolerance": 1e-10, "max_passes": 200}
    return {"model": {"materials": c.materials}, "point": point}


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
