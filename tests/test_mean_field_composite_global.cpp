#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/isotropize.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/eshelby_cylinder.h"
#include "numsim-materials/materials/eshelby_sphere.h"
#include "numsim-materials/materials/eshelby_spheroid.h"
#include "numsim-materials/materials/isotropic_moduli.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/mean_field_composite_global.h"
#include "numsim-materials/mean_field/kernel_registry.h"
#include "numsim-materials/materials/phase.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/json_model.h"
#include "numsim-materials/umat/material_point_evaluator.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

constexpr T Km{3.0}, Gm{1.0}, Ki{60.0}, Gi{30.0}; // soft matrix, stiff inclusions
constexpr T nu_m{(3 * Km - 2 * Gm) / (2 * (3 * Km + Gm))};

T mori_tanaka_K(T c) { return Km + c * (Ki - Km) / (1 + (1 - c) * (Ki - Km) / (Km + 4 * Gm / 3)); }
T mori_tanaka_G(T c) {
  T const Gs{Gm * (9 * Km + 8 * Gm) / (6 * (Km + 2 * Gm))};
  return Gm + c * (Gi - Gm) / (1 + (1 - c) * (Gi - Gm) / (Gm + Gs));
}
T dilute_K(T c) { return Km + c * (Ki - Km) / (1 + (Ki - Km) / (Km + 4 * Gm / 3)); }

struct phase_spec {
  std::string name, material, shape;
  T fraction;
  T aspect_ratio{1};
  std::vector<T> direction{1, 0, 0};
};

void constant(ctx_type &ctx, std::string const &name, T value) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<T>("value", value);
  ctx.create<nm::constant_scalar<policy>>(p);
}

nm::object_spec<param_type> kernel_of(std::string const &type, param_type p = {}) { return {type, std::move(p)}; }

nm::object_spec<param_type> shape_of(std::string const &shape, T aspect_ratio, std::vector<T> const &direction) {
  param_type p;
  if (shape == "spheroid")
    p.insert<T>("aspect_ratio", aspect_ratio);
  if (shape != "sphere")
    p.insert<std::vector<T>>("direction", direction);
  return {shape, std::move(p)};
}

/// K, G -> isotropic tangents; nu of the matrix -> Eshelby nodes -> phases (matrix first) ->
/// mean_field_composite_global `kernel` -> linear_stress. Self-consistent carries the shapes in its kernel.
std::unique_ptr<ctx_type> build(std::string const &kernel, std::vector<phase_spec> const &inclusions,
                                std::string const &matrix = "matrix") {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  for (auto const &[name, K, G] : {std::tuple{"matrix", Km, Gm}, std::tuple{"stiff", Ki, Gi}, std::tuple{"void", 0.0, 0.0}}) {
    constant(ctx, std::string{name} + "_K", K);
    constant(ctx, std::string{name} + "_G", G);
    p.clear();
    p.insert<std::string>("name", std::string{name} + "_C");
    p.insert<std::string>("K_source", std::string{name} + "_K");
    p.insert<std::string>("G_source", std::string{name} + "_G");
    ctx.create<nm::isotropic_tangent<policy>>(p);
  }
  p.clear();
  p.insert<std::string>("name", "moduli");
  p.insert<std::string>("tangent_source", matrix + "_C");
  ctx.create<nm::isotropic_moduli<policy>>(p);
  auto shape = [&](std::string const &name, std::string const &shape, T aspect_ratio, std::vector<T> const &direction) {
    p.clear();
    p.insert<std::string>("name", name);
    p.insert<std::string>("nu_source", "moduli");
    p.insert<std::string>("nu_property", "poisson_ratio");
    p.insert<T>("aspect_ratio", aspect_ratio);
    p.insert<std::vector<T>>("direction", direction);
    if (shape == "sphere")
      ctx.create<nm::eshelby_sphere<policy>>(p);
    else if (shape == "cylinder")
      ctx.create<nm::eshelby_cylinder<policy>>(p);
    else
      ctx.create<nm::eshelby_spheroid<policy>>(p);
  };
  auto phase = [&](std::string const &name, std::string const &material, T fraction, std::string const &eshelby) {
    constant(ctx, name + "_c", fraction);
    p.clear();
    p.insert<std::string>("name", name);
    p.insert<std::string>("tangent_source", material + "_C");
    p.insert<std::string>("fraction_source", name + "_c");
    if (!eshelby.empty())
      p.insert<std::string>("eshelby_source", eshelby);
    ctx.create<nm::phase<policy>>(p);
  };
  std::vector<std::string> names{"m"};
  std::map<std::string, nm::object_spec<param_type>> shapes{{"m", shape_of("sphere", 1, {})}};
  T cm{1};
  for (auto const &inc : inclusions) {
    shape(inc.name + "_shape", inc.shape, inc.aspect_ratio, inc.direction);
    phase(inc.name, inc.material, inc.fraction, inc.name + "_shape");
    names.push_back(inc.name);
    shapes.emplace(inc.name, shape_of(inc.shape, inc.aspect_ratio, inc.direction));
    cm -= inc.fraction;
  }
  phase("m", matrix, cm, "");
  param_type kp;
  if (kernel == "self_consistent") {
    kp.insert<std::map<std::string, nm::object_spec<param_type>>>("shapes", shapes);
    kp.insert<T>("tolerance", 1e-13);
  }
  p.clear();
  p.insert<std::string>("name", "mf");
  p.insert<std::vector<std::string>>("phases", names);
  p.insert<nm::object_spec<param_type>>("kernel", kernel_of(kernel, kp));
  ctx.create<nm::mean_field_composite_global<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "composite");
  p.insert<std::string>("tangent_source", "mf");
  p.insert<std::string>("strain_source", "strain");
  ctx.create<nm::linear_stress<policy>>(p);
  ctx.finalize();
  ctx.update();
  return owner;
}

TEST(Phase, EshelbyTensorFollowsTheShape) {
  auto ctx{build("dilute", {{"s", "stiff", "sphere", 0.1}, {"c", "stiff", "cylinder", 0.1, 1, {0, 1, 0}},
                            {"p", "stiff", "spheroid", 0.1, 3.0, {0, 0, 2}}})};
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("s", "eshelby") - nm::eshelby::sphere(nu_m)}), 0, 1e-14);
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("c", "eshelby") - nm::eshelby::cylinder(nu_m, std::size_t{1})}), 0, 1e-13);
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("p", "eshelby") - nm::eshelby::ellipsoid(nu_m, 1.0, 1.0, 3.0)}), 0, 1e-12); // direction is normalised
  EXPECT_DOUBLE_EQ(ctx->get<T>("s", "volume_fraction"), 0.1);
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("s", "tangent") - ctx->get<tensor4>("stiff_C", "tangent")}), 0, 1e-14);
}

TEST(Phase, APhaseWithoutAShapeIsRejectedByTheMeanField) {
  ctx_type ctx;
  for (auto const &[name, K, G] : {std::tuple{"matrix", Km, Gm}, std::tuple{"stiff", Ki, Gi}}) {
    constant(ctx, std::string{name} + "_K", K);
    constant(ctx, std::string{name} + "_G", G);
    param_type p;
    p.insert<std::string>("name", std::string{name} + "_C");
    p.insert<std::string>("K_source", std::string{name} + "_K");
    p.insert<std::string>("G_source", std::string{name} + "_G");
    ctx.create<nm::isotropic_tangent<policy>>(p);
  }
  constant(ctx, "c", 0.1);
  constant(ctx, "cm", 0.9);
  param_type p;
  p.insert<std::string>("name", "s"); // a plain phase, fine for a rule of mixtures
  p.insert<std::string>("tangent_source", "stiff_C");
  p.insert<std::string>("fraction_source", "c");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "m");
  p.insert<std::string>("tangent_source", "matrix_C");
  p.insert<std::string>("fraction_source", "cm");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "mf");
  p.insert<std::string>("matrix", "m");
  p.insert<std::vector<std::string>>("phases", {"m", "s"});
  p.insert<nm::object_spec<param_type>>("kernel", kernel_of("mori_tanaka"));
  ctx.create<nm::mean_field_composite_global<policy>>(p);
  EXPECT_THROW(ctx.finalize(), std::runtime_error); // s::eshelby does not exist
}

TEST(MeanField, MoriTanakaSpheresMatchTheAnalyticModuli) {
  for (T const c : {0.05, 0.3, 0.6}) {
    auto ctx{build("mori_tanaka", {{"s", "stiff", "sphere", c}})};
    auto const [K, G]{nm::isotropize<T, 3>(ctx->get<tensor4>("mf", "tangent"))};
    EXPECT_NEAR(K, mori_tanaka_K(c), 1e-10 * mori_tanaka_K(c)) << c;
    EXPECT_NEAR(G, mori_tanaka_G(c), 1e-10 * mori_tanaka_G(c)) << c;
    EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("mf", "tangent") - nm::isotropic_stiffness<T, 3>(K, G)}), 0, 1e-10);
  }
}

TEST(MeanField, DiluteSpheresMatchTheAnalyticBulkModulus) {
  auto ctx{build("dilute", {{"s", "stiff", "sphere", 0.1}})};
  auto const [K, G]{nm::isotropize<T, 3>(ctx->get<tensor4>("mf", "tangent"))};
  EXPECT_NEAR(K, dilute_K(0.1), 1e-10);
  auto mt{build("mori_tanaka", {{"s", "stiff", "sphere", 0.001}})}; // dilute = Mori-Tanaka as c -> 0
  auto const [Kmt, Gmt]{nm::isotropize<T, 3>(mt->get<tensor4>("mf", "tangent"))};
  auto dil{build("dilute", {{"s", "stiff", "sphere", 0.001}})};
  auto const [Kd, Gd]{nm::isotropize<T, 3>(dil->get<tensor4>("mf", "tangent"))};
  EXPECT_NEAR(Kmt, Kd, 1e-5 * Kd);
}

TEST(MeanField, TwoInclusionsOfOneMaterialEqualOne) {
  auto one{build("mori_tanaka", {{"s", "stiff", "sphere", 0.3}})};
  auto two{build("mori_tanaka", {{"a", "stiff", "sphere", 0.1}, {"b", "stiff", "sphere", 0.2}})};
  EXPECT_NEAR(tmech::norm(tensor4{one->get<tensor4>("mf", "tangent") - two->get<tensor4>("mf", "tangent")}), 0, 1e-12);
}

TEST(MeanField, VoidsSoften) {
  auto ctx{build("mori_tanaka", {{"v", "void", "sphere", 0.2}})};
  auto const [K, G]{nm::isotropize<T, 3>(ctx->get<tensor4>("mf", "tangent"))};
  T const Kv{Km + 0.2 * (0 - Km) / (1 + 0.8 * (0 - Km) / (Km + 4 * Gm / 3))};
  EXPECT_NEAR(K, Kv, 1e-12);
  EXPECT_LT(G, Gm);
}

TEST(MeanField, ConcentrationsAverageToIdentityAndStressIsOnTheTangent) {
  for (auto const *kernel : {"dilute", "mori_tanaka", "self_consistent"}) {
    auto ctx{build(kernel, {{"s", "stiff", "sphere", 0.3}})};
    tensor4 const A_sum{0.7 * ctx->get<tensor4>("mf", "concentration_m") + 0.3 * ctx->get<tensor4>("mf", "concentration_s")};
    EXPECT_NEAR(tmech::norm(tensor4{A_sum - nm::sym_identity<T, 3>()}), 0, 1e-12) << kernel;
    tensor2 eps;
    eps(0, 0) = 1e-3;
    eps(1, 2) = eps(2, 1) = 5e-4;
    dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))->bind(tensor2{}, eps);
    ctx->update();
    tensor2 const expected{tmech::dcontract(ctx->get<tensor4>("mf", "tangent"), eps)};
    EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("composite", "stress") - expected}), 0, 1e-12) << kernel;
  }
}

TEST(MeanField, SelfConsistentIsSymmetricInThePhasesAndStifferThanMoriTanaka) {
  auto scs{build("self_consistent", {{"s", "stiff", "sphere", 0.4}})};
  auto const [K, G]{nm::isotropize<T, 3>(scs->get<tensor4>("mf", "tangent"))};
  EXPECT_GT(K, mori_tanaka_K(0.4));
  EXPECT_GT(G, mori_tanaka_G(0.4));
  // the same composite described from the stiff phase: matrix "stiff", inclusions "matrix" with c = 0.6
  auto swapped{build("self_consistent", {{"m2", "matrix", "sphere", 0.6}}, "stiff")};
  EXPECT_NEAR(tmech::norm(tensor4{scs->get<tensor4>("mf", "tangent") - swapped->get<tensor4>("mf", "tangent")}), 0,
              1e-8 * tmech::norm(scs->get<tensor4>("mf", "tangent")));
}

// homogenisation once: the fixed point converges inside one update, so another changes nothing
TEST(MeanField, SelfConsistentConvergesInOneUpdate) {
  auto ctx{build("self_consistent", {{"s", "stiff", "sphere", 0.3}})};
  tensor4 const C{ctx->get<tensor4>("mf", "tangent")};
  ctx->update();
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("mf", "tangent") - C}), 0, 1e-12 * tmech::norm(C));
}

/// Mori-Tanaka counting its calls.
template <typename Traits>
class counting_mori_tanaka final : public nm::mean_field::kernel_base<counting_mori_tanaka<Traits>, Traits> {
public:
  using base = nm::mean_field::kernel_base<counting_mori_tanaka<Traits>, Traits>;
  using value_type = typename Traits::value_type;
  explicit counting_mori_tanaka(typename base::parameter_handler p) : base(p), m_kernel(std::move(p)) {}
  [[nodiscard]] nm::mean_field::kernel_needs needs() const noexcept override { return m_kernel.needs(); }
  void concentrations(std::span<nm::mean_field::phase_view<value_type> const> phases, std::span<tensor4> A) override {
    ++calls;
    m_kernel.concentrations(phases, A);
  }
  static inline int calls{0};

private:
  nm::mean_field::mori_tanaka<Traits> m_kernel;
};

// constant phases: the concentrations and the tangent are computed once, the stress on every update
TEST(MeanField, ConstantPhasesAreHomogenisedOnce) {
  static bool const registered{[] {
    nm::mean_field::kernel_registry<policy>::instance().template add<counting_mori_tanaka<policy>>("counting_mori_tanaka");
    return true;
  }()};
  ASSERT_TRUE(registered);
  counting_mori_tanaka<policy>::calls = 0;
  auto ctx{build("counting_mori_tanaka", {{"s", "stiff", "sphere", 0.3}})};
  ctx->verify_constants(false); // counts the kernel calls
  auto reference{build("mori_tanaka", {{"s", "stiff", "sphere", 0.3}})};
  for (int i{1}; i <= 4; ++i) {
    tensor2 eps;
    eps(0, 1) = eps(1, 0) = 1e-3 * i;
    dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))->bind(tensor2{}, eps);
    ctx->update();
    tensor2 const expected{tmech::dcontract(reference->get<tensor4>("mf", "tangent"), eps)};
    EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("composite", "stress") - expected}), 0, 1e-14) << i;
  }
  EXPECT_EQ(counting_mori_tanaka<policy>::calls, 1);
  EXPECT_EQ(tmech::norm(tensor4{ctx->get<tensor4>("mf", "tangent") - reference->get<tensor4>("mf", "tangent")}), 0);
}

TEST(MeanField, AlignedFibresAreTransverselyIsotropic) {
  auto ctx{build("mori_tanaka", {{"f", "stiff", "cylinder", 0.3, 1, {1, 0, 0}}})};
  auto const &C{ctx->get<tensor4>("mf", "tangent")};
  EXPECT_NEAR(C(1, 1, 1, 1), C(2, 2, 2, 2), 1e-12);
  EXPECT_GT(C(0, 0, 0, 0), C(1, 1, 1, 1));
  // along the fibres the Voigt estimate is close: E_L ~ c E_i + (1 - c) E_m
  T const E_L{C(0, 0, 0, 0) - 2 * C(0, 0, 1, 1) * C(0, 0, 1, 1) / (C(1, 1, 1, 1) + C(1, 1, 2, 2))};
  T const voigt{0.3 * nm::youngs_modulus(Ki, Gi) + 0.7 * nm::youngs_modulus(Km, Gm)};
  EXPECT_NEAR(E_L, voigt, 0.05 * voigt);
}

TEST(MeanField, RejectsBadFractionsAndKernels) {
  EXPECT_THROW(build("mori_tanaka", {{"a", "stiff", "sphere", 0.6}, {"b", "stiff", "sphere", 0.6}}), std::invalid_argument); // matrix fraction < 0
  EXPECT_THROW(build("hashin", {{"a", "stiff", "sphere", 0.1}}), std::invalid_argument);
}

TEST(MeanField, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "constant_scalar", "name": "Km", "value": 3.0}, {"type": "constant_scalar", "name": "Gm", "value": 1.0},
    {"type": "constant_scalar", "name": "Ki", "value": 60.0}, {"type": "constant_scalar", "name": "Gi", "value": 30.0},
    {"type": "isotropic_tangent", "name": "Cm", "K_source": "Km", "G_source": "Gm"},
    {"type": "isotropic_tangent", "name": "Ci", "K_source": "Ki", "G_source": "Gi"},
    {"type": "isotropic_moduli", "name": "moduli", "tangent_source": "Cm"},
    {"type": "eshelby_sphere", "name": "sphere", "nu_source": "moduli", "nu_property": "poisson_ratio"},
    {"type": "constant_scalar", "name": "c", "value": 0.3},
    {"type": "constant_scalar", "name": "cm", "value": 0.7},
    {"type": "phase", "name": "spheres", "tangent_source": "Ci", "fraction_source": "c", "eshelby_source": "sphere"},
    {"type": "phase", "name": "matrix", "tangent_source": "Cm", "fraction_source": "cm"},
    {"type": "mean_field_composite_global", "name": "mf", "phases": ["matrix", "spheres"], "kernel": {"type": "mori_tanaka"}},
    {"type": "linear_stress", "name": "composite", "tangent_source": "mf", "strain_source": "strain"}]})")(ctx, {});
  ctx.update();
  auto const [K, G]{nm::isotropize<T, 3>(ctx.get<tensor4>("mf", "tangent"))};
  EXPECT_NEAR(K, mori_tanaka_K(0.3), 1e-10);
}

// self-consistent from a deck: the shapes are data of the kernel, no Eshelby node, no local edge
TEST(MeanField, SelfConsistentFromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "constant_scalar", "name": "Km", "value": 3.0}, {"type": "constant_scalar", "name": "Gm", "value": 1.0},
    {"type": "constant_scalar", "name": "Ki", "value": 60.0}, {"type": "constant_scalar", "name": "Gi", "value": 30.0},
    {"type": "isotropic_tangent", "name": "Cm", "K_source": "Km", "G_source": "Gm"},
    {"type": "isotropic_tangent", "name": "Ci", "K_source": "Ki", "G_source": "Gi"},
    {"type": "constant_scalar", "name": "c", "value": 0.4},
    {"type": "constant_scalar", "name": "cm", "value": 0.6},
    {"type": "phase", "name": "spheres", "tangent_source": "Ci", "fraction_source": "c"},
    {"type": "phase", "name": "matrix", "tangent_source": "Cm", "fraction_source": "cm"},
    {"type": "mean_field_composite_global", "name": "mf", "phases": ["matrix", "spheres"],
     "kernel": {"type": "self_consistent", "tolerance": 1e-13, "shapes": {"matrix": {"type": "sphere"}, "spheres": {"type": "sphere"}}}},
    {"type": "linear_stress", "name": "composite", "tangent_source": "mf", "strain_source": "strain"}]})")(ctx, {});
  ctx.update();
  auto reference{build("self_consistent", {{"s", "stiff", "sphere", 0.4}})};
  EXPECT_NEAR(tmech::norm(tensor4{ctx.get<tensor4>("mf", "tangent") - reference->get<tensor4>("mf", "tangent")}), 0,
              1e-10 * tmech::norm(ctx.get<tensor4>("mf", "tangent")));
}

} // namespace
