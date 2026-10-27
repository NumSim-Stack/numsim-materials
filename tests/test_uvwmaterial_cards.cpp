#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/isotropize.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using tensor4 = tmech::tensor<T, 3, 4>;

/// "model" of tools/convert_uvwmaterial_card.py for the MORI_TANAKA card of tools/tests (its own tests check the rest);
/// the composite is the tangent producer "mt_tangent", "mt" its linear_stress.
char const *const converted = R"({"materials": [{"type": "external_strain_source", "name": "strain"}, {"type": "constant_scalar", "name": "mt_matrix_material_K", "value": 6000.000000000001}, {"type": "constant_scalar", "name": "mt_matrix_material_G", "value": 620.6896551724138}, {"type": "isotropic_tangent", "name": "mt_matrix_material_C", "K_source": "mt_matrix_material_K", "G_source": "mt_matrix_material_G"}, {"type": "isotropic_moduli", "name": "mt_matrix_material_C_moduli", "tangent_source": "mt_matrix_material_C"}, {"type": "constant_scalar", "name": "mt_inclusion1_material_K", "value": 1749.9999999999998}, {"type": "constant_scalar", "name": "mt_inclusion1_material_G", "value": 807.6923076923076}, {"type": "isotropic_tangent", "name": "mt_inclusion1_material_C", "K_source": "mt_inclusion1_material_K", "G_source": "mt_inclusion1_material_G"}, {"type": "constant_scalar", "name": "mt_inclusion1_c", "value": 0.5}, {"type": "eshelby_sphere", "name": "mt_inclusion1_shape", "nu_source": "mt_matrix_material_C_moduli", "nu_property": "poisson_ratio"}, {"type": "phase", "name": "mt_inclusion1", "tangent_source": "mt_inclusion1_material_C", "fraction_source": "mt_inclusion1_c", "eshelby_source": "mt_inclusion1_shape"}, {"type": "constant_scalar", "name": "mt_matrix_c", "value": 0.5}, {"type": "phase", "name": "mt_matrix", "tangent_source": "mt_matrix_material_C", "fraction_source": "mt_matrix_c"}, {"type": "mean_field_composite_global", "name": "mt_tangent", "phases": ["mt_matrix", "mt_inclusion1"], "kernel": {"type": "mori_tanaka"}}, {"type": "linear_stress", "name": "mt", "tangent_source": "mt_tangent", "strain_source": "strain"}]})";

TEST(UvwmaterialCards, TheConvertedMoriTanakaCardLoadsAndMatchesTheAnalyticModuli) {
  nm::material_context<policy> ctx;
  nm::umat::make_json_builder<policy>(converted)(ctx, {});
  ctx.update();
  T const Km{nm::bulk_modulus(1800.0, 0.45)}, Gm{nm::shear_modulus(1800.0, 0.45)};
  T const Ki{nm::bulk_modulus(2100.0, 0.3)};
  T const c{0.5};
  T const K_mt{Km + c * (Ki - Km) / (1 + (1 - c) * (Ki - Km) / (Km + 4 * Gm / 3))};
  auto const [K, G]{nm::isotropize<T, 3>(ctx.get<tensor4>("mt_tangent", "tangent"))};
  EXPECT_NEAR(K, K_mt, 1e-9 * K_mt);
}

} // namespace
