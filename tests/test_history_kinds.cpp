#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/history_kind.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/umat/statev_map.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;
namespace kinds = numsim::materials::history_kinds;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor1 = tmech::tensor<T, 3, 1>;
using tensor2 = tmech::tensor<T, 3, 2>;
using map_type = u::statev_map<policy>;

/// One plain-tensor history output of every kind; names sort in declaration order.
template <typename Traits>
class frame_probe final : public nm::material_base<frame_probe<Traits>, Traits> {
public:
  using base = nm::material_base<frame_probe<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using vector_type = tmech::tensor<value_type, 3, 1>;
  using tensor_type = tmech::tensor<value_type, 3, 2>;

  template <typename... Args>
  explicit frame_probe(Args&&... args)
      : base(std::forward<Args>(args)...),
        a_ref(base::template add_history_output<tensor_type>("a_ref", kinds::reference_strain)),
        b_mvec(base::template add_history_output<vector_type>("b_mvec", kinds::material_vector)),
        c_strain(base::template add_history_output<tensor_type>("c_strain")),
        d_svec(base::template add_history_output<vector_type>("d_svec", kinds::spatial_vector)),
        e_fe(base::template add_history_output<tensor_type>("e_fe", kinds::two_point)),
        f_fp(base::template add_history_output<tensor_type>("f_fp", kinds::material_tensor)),
        g_moduli(base::template add_history_output<tensor_type>("g_moduli", kinds::spatial_moduli)) {}

  static input_parameter_controller parameters() { return base::parameters(); }

  nm::history_property<tensor_type>& a_ref;
  nm::history_property<vector_type>& b_mvec;
  nm::history_property<tensor_type>& c_strain;
  nm::history_property<vector_type>& d_svec;
  nm::history_property<tensor_type>& e_fe;
  nm::history_property<tensor_type>& f_fp;
  nm::history_property<tensor_type>& g_moduli;
};

/// Only material-leg and scalar history: nothing a rotation acts on.
template <typename Traits>
class material_frame_probe final : public nm::material_base<material_frame_probe<Traits>, Traits> {
public:
  using base = nm::material_base<material_frame_probe<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using vector_type = tmech::tensor<value_type, 3, 1>;
  using tensor_type = tmech::tensor<value_type, 3, 2>;

  template <typename... Args>
  explicit material_frame_probe(Args&&... args)
      : base(std::forward<Args>(args)...),
        damage(base::template add_history_output<value_type>("damage")),
        fp(base::template add_history_output<tensor_type>("fp", kinds::material_tensor)),
        moduli(base::template add_history_output<tensor_type>("moduli", kinds::material_moduli)),
        mvec(base::template add_history_output<vector_type>("mvec", kinds::material_vector)),
        ref(base::template add_history_output<tensor_type>("ref", kinds::reference_strain)) {}

  static input_parameter_controller parameters() { return base::parameters(); }

  nm::history_property<value_type>& damage;
  nm::history_property<tensor_type>& fp;
  nm::history_property<tensor_type>& moduli;
  nm::history_property<vector_type>& mvec;
  nm::history_property<tensor_type>& ref;
};

/// A rank-2 tensor and a vector history declared without a kind.
template <typename Traits>
class default_kind_probe final : public nm::material_base<default_kind_probe<Traits>, Traits> {
public:
  using base = nm::material_base<default_kind_probe<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit default_kind_probe(Args&&... args)
      : base(std::forward<Args>(args)...),
        strain(base::template add_history_output<tmech::tensor<value_type, 3, 2>>("strain")),
        vec(base::template add_history_output<tmech::tensor<value_type, 3, 1>>("vec")) {}

  static input_parameter_controller parameters() { return base::parameters(); }

  nm::history_property<tmech::tensor<value_type, 3, 2>>& strain;
  nm::history_property<tmech::tensor<value_type, 3, 1>>& vec;
};

/// Declares a history with a kind its value cannot take: 0 engineering vector, 1 plain two-point, 2 scalar.
template <typename Traits, int Case>
class invalid_kind_probe final : public nm::material_base<invalid_kind_probe<Traits, Case>, Traits> {
public:
  using base = nm::material_base<invalid_kind_probe<Traits, Case>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit invalid_kind_probe(Args&&... args) : base(std::forward<Args>(args)...) {
    if constexpr (Case == 0)
      base::template add_history_output<tmech::tensor<value_type, 3, 1>>("v", kinds::strain);
    else if constexpr (Case == 1)
      base::template add_history_output<tmech::tensor<value_type, 3, 2>>(
          "fe", nm::history_kind{nm::storage::plain, {nm::leg::spatial, nm::leg::material}});
    else
      base::template add_history_output<value_type>("d", kinds::stress_like);
  }

  static input_parameter_controller parameters() { return base::parameters(); }
};

/// Reads probe::a_ref, a reference_strain history, through a plain tensor input.
template <typename Traits>
class reference_reader final : public nm::material_base<reference_reader<Traits>, Traits> {
public:
  using base = nm::material_base<reference_reader<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor_type = tmech::tensor<value_type, 3, 2>;

  template <typename... Args>
  explicit reference_reader(Args&&... args)
      : base(std::forward<Args>(args)...),
        ref(base::template add_input<tensor_type>("probe", "a_ref")),
        copy(base::template add_output<tensor_type>("copy")) {
    base::add_callback(&reference_reader::compute, copy).reads(ref);
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() { copy = ref.get(); }

  nm::input_property<tensor_type, nm::property_traits> const& ref;
  tensor_type& copy;
};

tensor2 rotation_z(T degrees) {
  const T a = degrees * std::numbers::pi_v<T> / T{180};
  tensor2 R;
  R(0, 0) = std::cos(a);
  R(0, 1) = -std::sin(a);
  R(1, 0) = std::sin(a);
  R(1, 1) = std::cos(a);
  R(2, 2) = 1.0;
  return R;
}

tensor2 symmetric_sample() {
  tensor2 S;
  S(0, 0) = 1.0;  S(1, 1) = 2.0;  S(2, 2) = 3.0;
  S(0, 1) = S(1, 0) = 0.2;
  S(0, 2) = S(2, 0) = 0.3;
  S(1, 2) = S(2, 1) = 0.4;
  return S;
}

tensor2 non_symmetric_sample() {
  tensor2 F;
  F(0, 0) = 1.1;   F(0, 1) = 0.2;   F(0, 2) = 0.05;
  F(1, 0) = 0.03;  F(1, 1) = 0.95;  F(1, 2) = 0.1;
  F(2, 0) = 0.07;  F(2, 1) = 0.02;  F(2, 2) = 1.05;
  return F;
}

tensor1 vector_sample() {
  tensor1 v;
  v(0) = 1.0;  v(1) = 2.0;  v(2) = 3.0;
  return v;
}

template <std::size_t Rank, typename A, typename B>
T distance(A const& a, B const& b) {
  tmech::tensor<T, 3, Rank> d;
  d = a - b;
  return tmech::norm(d);
}

template <typename Probe>
struct probe_fixture {
  ctx_type ctx;
  Probe* probe{nullptr};
  probe_fixture() {
    param_type p;
    p.insert<std::string>("name", "probe");
    probe = &ctx.template create<Probe>(p);
    ctx.finalize();
  }
};

/// Sets both sides, as a converged step followed by unpack would.
template <typename H, typename V>
void set_history(H& h, V const& v) {
  h.old_value() = v;
  h.new_value() = v;
}

/// The logic_error creating @p Probe throws, or a note that it threw none.
template <typename Probe>
std::string declaration_error() {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "probe");
  try {
    ctx.template create<Probe>(p);
  } catch (std::logic_error const& e) {
    return e.what();
  }
  return "no logic_error";
}

}  // namespace

// ---------------------------------------------------------------------------
// The kind itself
// ---------------------------------------------------------------------------

TEST(HistoryKind, RotatedActsOnEachSpatialLegOnly) {
  using nm::leg;
  const auto full = [](leg first, leg second) { return nm::history_kind{nm::storage::full, {first, second}}; };
  const auto R = rotation_z(30.0);
  const auto F = non_symmetric_sample();
  const auto v = vector_sample();

  tensor1 Rv;
  Rv = R * v;
  tensor2 RFRt, RF, FRt;
  RFRt = R * F * tmech::trans(R);
  RF = R * F;
  FRt = F * tmech::trans(R);

  // A vector uses the first leg only.
  EXPECT_LT(distance<1>(full(leg::spatial, leg::material).rotated(v, R), Rv), 1e-14);
  EXPECT_EQ(distance<1>(full(leg::material, leg::spatial).rotated(v, R), v), 0.0);
  EXPECT_LT(distance<2>(full(leg::spatial, leg::spatial).rotated(F, R), RFRt), 1e-14);
  EXPECT_LT(distance<2>(full(leg::spatial, leg::material).rotated(F, R), RF), 1e-14);
  EXPECT_LT(distance<2>(full(leg::material, leg::spatial).rotated(F, R), FRt), 1e-14);
  EXPECT_EQ(distance<2>(full(leg::material, leg::material).rotated(F, R), F), 0.0);
  EXPECT_GT(distance<2>(RF, F), 0.1) << "the rotation must be observable";
}

TEST(HistoryKind, ValidForAndRotatesFollowTheLegsARankUses) {
  using nm::leg;
  using nm::storage;
  constexpr nm::history_kind full_ms{storage::full, {leg::material, leg::spatial}};
  constexpr nm::history_kind plain_sm{storage::plain, {leg::spatial, leg::material}};
  constexpr nm::history_kind engineering_ms{storage::engineering, {leg::material, leg::spatial}};
  static_assert(kinds::strain.valid_for(2) && !kinds::strain.valid_for(1));

  // A vector: all components only.
  EXPECT_TRUE(kinds::spatial_vector.valid_for(1));
  EXPECT_TRUE(kinds::material_vector.valid_for(1));
  EXPECT_FALSE(kinds::strain.valid_for(1));
  EXPECT_FALSE(kinds::stress_like.valid_for(1));
  // A rank-2 tensor: six slots need both legs alike.
  EXPECT_TRUE(kinds::strain.valid_for(2));
  EXPECT_TRUE(kinds::reference_strain.valid_for(2));
  EXPECT_TRUE(kinds::material_moduli.valid_for(2));
  EXPECT_TRUE(kinds::two_point.valid_for(2));
  EXPECT_TRUE(full_ms.valid_for(2));
  EXPECT_FALSE(plain_sm.valid_for(2));
  EXPECT_FALSE(engineering_ms.valid_for(2));
  // Nothing else takes a kind.
  EXPECT_FALSE(kinds::two_point.valid_for(0));
  EXPECT_FALSE(kinds::two_point.valid_for(4));

  EXPECT_TRUE(kinds::strain.rotates(2));
  EXPECT_TRUE(kinds::two_point.rotates(2));
  EXPECT_TRUE(full_ms.rotates(2));
  EXPECT_FALSE(kinds::reference_strain.rotates(2));
  EXPECT_FALSE(kinds::material_tensor.rotates(2));
  EXPECT_TRUE(kinds::spatial_vector.rotates(1));
  EXPECT_FALSE(kinds::material_vector.rotates(1));
  EXPECT_FALSE(full_ms.rotates(1)) << "a vector uses the first leg only";
}

TEST(HistoryKind, NamedKindsAndDefaults) {
  EXPECT_EQ(kinds::spatial_moduli, kinds::stress_like);
  EXPECT_EQ(kinds::stress_like.store, nm::storage::plain);
  EXPECT_EQ(kinds::reference_strain.store, nm::storage::engineering);
  EXPECT_EQ(kinds::two_point.legs[1], nm::leg::material);
  EXPECT_EQ(nm::default_history_kind<tensor2>, kinds::strain);
  EXPECT_EQ(nm::default_history_kind<tensor1>, kinds::spatial_vector);
  EXPECT_FALSE(nm::default_history_kind<T>.has_value());
  EXPECT_FALSE((nm::default_history_kind<tmech::tensor<T, 3, 4>>.has_value()));
}

TEST(HistoryKind, AnInvalidKindThrowsWhenTheHistoryIsDeclared) {
  const auto vector_error = declaration_error<invalid_kind_probe<policy, 0>>();
  EXPECT_NE(vector_error.find("probe::v"), std::string::npos) << vector_error;
  EXPECT_NE(vector_error.find("engineering"), std::string::npos) << vector_error;

  const auto two_point_error = declaration_error<invalid_kind_probe<policy, 1>>();
  EXPECT_NE(two_point_error.find("probe::fe"), std::string::npos) << two_point_error;
  EXPECT_NE(two_point_error.find("plain"), std::string::npos) << two_point_error;

  const auto scalar_error = declaration_error<invalid_kind_probe<policy, 2>>();
  EXPECT_NE(scalar_error.find("probe::d"), std::string::npos) << scalar_error;
}

/// The kind lives at the declaration, so a consumer reads the history as the plain tensor it is.
TEST(HistoryKind, AConsumerReadsAReferenceStrainThroughAPlainTensorInput) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "probe");
  auto& probe = ctx.create<frame_probe<policy>>(p);
  param_type q;
  q.insert<std::string>("name", "reader");
  auto& reader = ctx.create<reference_reader<policy>>(q);
  ASSERT_NO_THROW(ctx.finalize());

  probe.a_ref.new_value() = symmetric_sample();
  ctx.update();
  EXPECT_EQ(distance<2>(reader.copy, symmetric_sample()), 0.0);
}

// ---------------------------------------------------------------------------
// STATEV storage
// ---------------------------------------------------------------------------

TEST(StatevMapKinds, WidthsAndDescriptionFollowTheKind) {
  probe_fixture<frame_probe<policy>> f;
  const map_type map(f.ctx, {});
  EXPECT_EQ(map.nstatv(), 6u + 3u + 6u + 3u + 9u + 9u + 6u);

  const std::vector<std::string> expected{
      "0..5  probe::a_ref  [engineering shear] [reference]",
      "6..8  probe::b_mvec  [vector] [reference]",
      "9..14  probe::c_strain  [engineering shear]",
      "15..17  probe::d_svec  [vector]",
      "18..26  probe::e_fe  [full] [two-point]",
      "27..35  probe::f_fp  [full] [reference]",
      "36..41  probe::g_moduli  [stress]"};
  EXPECT_EQ(map.describe(), expected);
}

TEST(StatevMapKinds, DrotRotatesSpatialHistoryAndLeavesMaterialHistory) {
  probe_fixture<frame_probe<policy>> f;
  auto& p = *f.probe;
  const map_type map(f.ctx, {});
  EXPECT_TRUE(map.has_rotatable_history());

  const auto R = rotation_z(30.0);
  const auto S = symmetric_sample();
  const auto F = non_symmetric_sample();
  const auto v = vector_sample();
  set_history(p.a_ref, S);
  set_history(p.b_mvec, v);
  set_history(p.c_strain, S);
  set_history(p.d_svec, v);
  set_history(p.e_fe, F);
  set_history(p.f_fp, F);
  set_history(p.g_moduli, S);

  map.rotate_history(R);

  tensor1 Rv;
  Rv = R * v;
  tensor2 RSRt, RF;
  RSRt = R * S * tmech::trans(R);
  RF = R * F;
  for (auto side : {0, 1}) {
    auto get = [side](auto& h) -> auto const& { return side == 0 ? h.old_value() : h.new_value(); };
    EXPECT_EQ(distance<2>(get(p.a_ref), S), 0.0);
    EXPECT_EQ(distance<1>(get(p.b_mvec), v), 0.0);
    EXPECT_LT(distance<2>(get(p.c_strain), RSRt), 1e-14);
    EXPECT_LT(distance<1>(get(p.d_svec), Rv), 1e-14);
    EXPECT_LT(distance<2>(get(p.e_fe), RF), 1e-14);
    EXPECT_EQ(distance<2>(get(p.f_fp), F), 0.0);
    EXPECT_LT(distance<2>(get(p.g_moduli), RSRt), 1e-14);
  }
  EXPECT_GT(distance<2>(RSRt, S), 0.1) << "the rotation must be observable";
}

TEST(StatevMapKinds, NonSymmetricTensorsKeepAllNineComponents) {
  probe_fixture<frame_probe<policy>> f;
  auto& p = *f.probe;
  const map_type map(f.ctx, {});

  const auto F = non_symmetric_sample();
  tensor2 Ft;
  Ft = tmech::trans(F);
  p.e_fe.new_value() = F;
  p.f_fp.new_value() = Ft;
  std::vector<T> statev(map.nstatv(), 0.0);
  EXPECT_NO_THROW(map.pack(statev.data()));

  // Row-major {11 12 13 21 22 23 31 32 33}.
  EXPECT_DOUBLE_EQ(statev[18 + 1], F(0, 1));
  EXPECT_DOUBLE_EQ(statev[18 + 3], F(1, 0));
  EXPECT_DOUBLE_EQ(statev[27 + 1], F(1, 0));
  EXPECT_DOUBLE_EQ(statev[27 + 7], F(1, 2));

  set_history(p.e_fe, tensor2{});
  set_history(p.f_fp, tensor2{});
  map.unpack(statev.data());
  EXPECT_EQ(distance<2>(p.e_fe.old_value(), F), 0.0);
  EXPECT_EQ(distance<2>(p.e_fe.new_value(), F), 0.0);
  EXPECT_EQ(distance<2>(p.f_fp.old_value(), Ft), 0.0);
  EXPECT_EQ(distance<2>(p.f_fp.new_value(), Ft), 0.0);
}

TEST(StatevMapKinds, VectorsRoundTripThroughThreeSlots) {
  probe_fixture<frame_probe<policy>> f;
  auto& p = *f.probe;
  const map_type map(f.ctx, {});

  const auto v = vector_sample();
  tensor1 w;
  w = -2.0 * v;
  p.b_mvec.new_value() = w;
  p.d_svec.new_value() = v;
  std::vector<T> statev(map.nstatv(), 0.0);
  map.pack(statev.data());
  EXPECT_DOUBLE_EQ(statev[6 + 2], -6.0);
  EXPECT_DOUBLE_EQ(statev[15 + 0], 1.0);
  EXPECT_DOUBLE_EQ(statev[15 + 2], 3.0);

  set_history(p.b_mvec, tensor1{});
  set_history(p.d_svec, tensor1{});
  map.unpack(statev.data());
  EXPECT_EQ(distance<1>(p.b_mvec.old_value(), w), 0.0);
  EXPECT_EQ(distance<1>(p.d_svec.new_value(), v), 0.0);
}

TEST(StatevMapKinds, SpatialModuliStoreShearAsIsNextToEngineeringStrain) {
  probe_fixture<frame_probe<policy>> f;
  auto& p = *f.probe;
  const map_type map(f.ctx, {});

  tensor2 shear;
  shear(0, 1) = shear(1, 0) = 0.1;
  p.g_moduli.new_value() = shear;
  p.c_strain.new_value() = shear;
  p.a_ref.new_value() = shear;
  std::vector<T> statev(map.nstatv(), 0.0);
  map.pack(statev.data());
  // Abaqus order {11,22,33,12,13,23}: slot 3 is the 12 component.
  EXPECT_DOUBLE_EQ(statev[36 + 3], 0.1);
  EXPECT_DOUBLE_EQ(statev[9 + 3], 0.2);
  EXPECT_DOUBLE_EQ(statev[0 + 3], 0.2);

  set_history(p.g_moduli, tensor2{});
  map.unpack(statev.data());
  EXPECT_EQ(distance<2>(p.g_moduli.old_value(), shear), 0.0);
}

/// Six slots hold only a symmetric tensor, whichever leg it lives on.
TEST(StatevMapKinds, SymmetricStorageRejectsASkewPart) {
  probe_fixture<frame_probe<policy>> f;
  const map_type map(f.ctx, {});
  f.probe->a_ref.new_value() = non_symmetric_sample();
  std::vector<T> statev(map.nstatv(), 0.0);
  EXPECT_THROW(map.pack(statev.data()), u::fatal_error);
}

TEST(StatevMapKinds, MaterialAndScalarHistoryIsNotRotatable) {
  probe_fixture<material_frame_probe<policy>> f;
  auto& p = *f.probe;
  const map_type map(f.ctx, {});
  EXPECT_EQ(map.nstatv(), 1u + 9u + 6u + 3u + 6u);
  EXPECT_FALSE(map.has_rotatable_history());

  const auto F = non_symmetric_sample();
  set_history(p.fp, F);
  map.rotate_history(rotation_z(30.0));
  EXPECT_EQ(distance<2>(p.fp.new_value(), F), 0.0);
  EXPECT_NE(map.describe()[2].find("[stress] [reference]"), std::string::npos) << map.describe()[2];
}

TEST(StatevMapKinds, HistoryWithoutAKindIsASpatialStrainOrASpatialVector) {
  probe_fixture<default_kind_probe<policy>> f;
  auto& p = *f.probe;
  const map_type map(f.ctx, {});
  const std::vector<std::string> expected{"0..5  probe::strain  [engineering shear]", "6..8  probe::vec  [vector]"};
  EXPECT_EQ(map.describe(), expected);
  ASSERT_TRUE(map.has_rotatable_history());

  const auto R = rotation_z(30.0);
  const auto S = symmetric_sample();
  const auto v = vector_sample();
  set_history(p.strain, S);
  set_history(p.vec, v);
  map.rotate_history(R);
  tensor1 Rv;
  Rv = R * v;
  tensor2 RSRt;
  RSRt = R * S * tmech::trans(R);
  EXPECT_LT(distance<2>(p.strain.old_value(), RSRt), 1e-14);
  EXPECT_LT(distance<1>(p.vec.new_value(), Rv), 1e-14);

  std::vector<T> statev(map.nstatv(), 0.0);
  map.pack(statev.data());
  EXPECT_DOUBLE_EQ(statev[3], 2.0 * RSRt(0, 1));
  EXPECT_DOUBLE_EQ(statev[6 + 2], Rv(2));
}
