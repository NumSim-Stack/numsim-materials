#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <tmech/tmech.h>
#include "numsim-materials/core/graph_description.h"
#include "numsim-materials/core/property_value.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

tensor2 sample() {
  tensor2 A;
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 4;
  A(1, 2) = -3;
  A(2, 2) = 5;
  return A;
}

TEST(TensorIdentities, ProjectSymmetricSkewVolumetricDeviatoric) {
  auto const A{sample()};
  EXPECT_NEAR(tmech::norm(tensor2{tmech::dcontract(nm::sym_identity<T, 3>(), A) - tmech::sym(A)}), 0, 1e-15);
  EXPECT_NEAR(tmech::norm(tensor2{tmech::dcontract(nm::skew_identity<T, 3>(), A) - tmech::skew(A)}), 0, 1e-15);
  EXPECT_NEAR(tmech::norm(tensor2{tmech::dcontract(nm::vol_projector<T, 3>(), A) - tmech::trace(A) / 3 * tmech::eye<T, 3, 2>()}), 0, 1e-15);
  EXPECT_NEAR(tmech::norm(tensor4{nm::vol_projector<T, 3>() + nm::dev_projector<T, 3>() - nm::sym_identity<T, 3>()}), 0, 1e-15);
}

TEST(TensorIdentities, WorkInTwoDimensions) {
  auto const I{tmech::eye<T, 2, 2>()};
  tmech::tensor<T, 2, 2> const v{tmech::dcontract(nm::vol_projector<T, 2>(), I)};
  EXPECT_NEAR(tmech::norm(tmech::tensor<T, 2, 2>{v - I}), 0, 1e-15);
}

TEST(TensorIdentities, SymmetricBasisIsOrthonormalInAbaqusOrder) {
  auto const &E{nm::symmetric_basis<T>()};
  for (std::size_t a{0}; a < 6; ++a)
    for (std::size_t b{0}; b < 6; ++b)
      EXPECT_NEAR(tmech::dcontract(E[a], E[b]), a == b ? 1 : 0, 1e-15) << a << b;
  EXPECT_EQ(nm::component_index("12"), 3u);
  EXPECT_THROW(nm::component_index("21"), std::invalid_argument);
  std::array<T, 6> const v{1, 2, 3, 4, 5, 6};
  auto const t{nm::symmetric_tensor<T>(v)};
  EXPECT_EQ(t(1, 0), 4);
  EXPECT_EQ(t(2, 1), 6);
}

struct elastic_graph {
  nm::material_context<policy> ctx;
  elastic_graph() {
    nm::umat::make_json_builder<policy>(R"({"materials": [
      {"type": "external_strain_source", "name": "strain"},
      {"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "strain", "K": 100.0, "G": 50.0}]})")(ctx, {});
  }
};

TEST(PropertyValue, ReadsPlainAndHistoryProperties) {
  elastic_graph g;
  auto const *C{nm::property_value<tensor4>(g.ctx.find_property("elastic", "tangent"))};
  ASSERT_NE(C, nullptr);
  EXPECT_DOUBLE_EQ((*C)(0, 0, 0, 0), 100 + 4 * 50.0 / 3);
  EXPECT_NE(nm::property_value<tensor2>(g.ctx.find_property("strain", "strain")), nullptr); // history
  EXPECT_EQ(nm::property_value<T>(g.ctx.find_property("elastic", "tangent")), nullptr);      // wrong type
  EXPECT_EQ(nm::property_value<T>(nullptr), nullptr);
}

TEST(GraphDescription, ListsProducedAndConsumedProperties) {
  elastic_graph g;
  auto const graph{nm::describe_graph(g.ctx)};
  ASSERT_EQ(graph.size(), 2u);
  auto const &elastic{graph[1]};
  EXPECT_EQ(elastic.name, "elastic");
  ASSERT_EQ(elastic.produced.size(), 2u);
  EXPECT_EQ(elastic.produced[0].name, "stress"); // sorted by name
  EXPECT_EQ(elastic.produced[1].name, "tangent");
  ASSERT_EQ(elastic.inputs.size(), 1u);
  EXPECT_EQ(elastic.inputs[0].source, "strain::strain");
  EXPECT_FALSE(elastic.inputs[0].local);
  EXPECT_TRUE(graph[0].produced[0].history);
}

} // namespace
