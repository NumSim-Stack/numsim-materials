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

/// The Mori-Tanaka example card of uvwmaterial after tools/convert_uvwmaterial_card.py (checked by its own tests).
char const *const converted = R"({"materials": [{"type": "external_strain_source", "name": "strain"}, {"type": "linear_elasticity", "name": "mt_matrix", "strain_producer_name": "strain", "K": 6000.000000000001, "G": 620.6896551724138}, {"type": "linear_elasticity", "name": "mt_inclusion1_material", "strain_producer_name": "strain", "K": 1749.9999999999998, "G": 807.6923076923076}, {"type": "inclusion", "name": "mt_inclusion1", "material": "mt_inclusion1_material", "volume_fraction": 0.5, "shape": "sphere"}, {"type": "mean_field_composite", "name": "mt", "matrix": "mt_matrix", "inclusions": ["mt_inclusion1"], "kernel": "mori_tanaka", "strain_source": "strain"}]})";

TEST(UvwmaterialCards, TheConvertedMoriTanakaCardLoadsAndMatchesTheAnalyticModuli) {
  nm::material_context<policy> ctx;
  nm::umat::make_json_builder<policy>(converted)(ctx, {});
  ctx.update();
  T const Km{nm::bulk_modulus(1800.0, 0.45)}, Gm{nm::shear_modulus(1800.0, 0.45)};
  T const Ki{nm::bulk_modulus(2100.0, 0.3)};
  T const c{0.5};
  T const K_mt{Km + c * (Ki - Km) / (1 + (1 - c) * (Ki - Km) / (Km + 4 * Gm / 3))};
  auto const [K, G]{nm::isotropize<T, 3>(ctx.get<tensor4>("mt", "tangent"))};
  EXPECT_NEAR(K, K_mt, 1e-9 * K_mt);
}

} // namespace
