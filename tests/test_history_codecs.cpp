#include <gtest/gtest.h>
#include <concepts>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/history_codec.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/umat/statev_map.h"
#include "test_fixture_materials.h"

namespace {

namespace nm = numsim::materials;
namespace codecs = numsim::materials::codecs;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using tensor1 = tmech::tensor<T, 3, 1>;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using map_type = nm::umat::statev_map<policy>;

// A codec that does not fit the value is a compile error, not a runtime check.
static_assert(nm::history_codec_for<codecs::strain, tensor2, T>);
static_assert(!nm::history_codec_for<codecs::reference_strain, tensor1, T>);
static_assert(!nm::history_codec_for<codecs::scalar, int, T>);

// Declared without a codec: the default for the value type, none for other values.
static_assert(std::same_as<nm::default_codec<T>, codecs::scalar>);
static_assert(std::same_as<nm::default_codec<tensor1>, codecs::vector>);
static_assert(std::same_as<nm::default_codec<tensor2>, codecs::strain>);
static_assert(std::same_as<nm::default_codec<tensor4>, codecs::moduli>);
static_assert(std::is_void_v<nm::default_codec<std::string>>);

/// A user value with its own codec: the direction turns with R, the length does not.
struct fibre {
  tensor1 direction;
  T length{};
};

struct fibre_codec {
  static constexpr std::size_t width = 4;
  static constexpr std::string_view tag = "[fibre]";
  static void read(fibre& f, T const* b) { f = {nm::full_from_buffer<T, 1>(b), b[3]}; }
  static void write(fibre const& f, T* b, std::string_view) {
    nm::full_to_buffer<T, 1>(f.direction, b);
    b[3] = f.length;
  }
  static fibre rotate(fibre const& f, tensor2 const& R) { return {tensor1{R * f.direction}, f.length}; }
};

/// One history per codec, names in STATEV order; c, d, h and i take the default codec.
class codec_probe final : public nm::material_base<codec_probe, policy> {
public:
  using base = nm::material_base<codec_probe, policy>;
  template <typename... Args>
  explicit codec_probe(Args&&... args)
      : base(std::forward<Args>(args)...),
        a_ref(add_history_output<tensor2>("a_ref", codecs::reference_strain{})),
        b_rvec(add_history_output<tensor1>("b_rvec", codecs::reference_vector{})),
        c_strain(add_history_output<tensor2>("c_strain")),
        d_vec(add_history_output<tensor1>("d_vec")),
        e_fe(add_history_output<tensor2>("e_fe", codecs::two_point{})),
        f_fp(add_history_output<tensor2>("f_fp", codecs::reference_tensor{})),
        g_stress(add_history_output<tensor2>("g_stress", codecs::stress{})),
        h_scalar(add_history_output<T>("h_scalar")),
        i_moduli(add_history_output<tensor4>("i_moduli")) {}
  static input_parameter_controller parameters() { return base::parameters(); }

  nm::history_property<tensor2>& a_ref;
  nm::history_property<tensor1>& b_rvec;
  nm::history_property<tensor2>& c_strain;
  nm::history_property<tensor1>& d_vec;
  nm::history_property<tensor2>& e_fe;
  nm::history_property<tensor2>& f_fp;
  nm::history_property<tensor2>& g_stress;
  nm::history_property<T>& h_scalar;
  nm::history_property<tensor4>& i_moduli;
};

/// Reads probe::a_ref, a reference_strain history, through a plain tensor input.
class reference_reader final : public nm::material_base<reference_reader, policy> {
public:
  using base = nm::material_base<reference_reader, policy>;
  template <typename... Args>
  explicit reference_reader(Args&&... args)
      : base(std::forward<Args>(args)...), ref(add_input<tensor2>("probe", "a_ref")),
        copy(add_output<tensor2>("copy")) {
    add_callback(&reference_reader::compute, copy).reads(ref);
  }
  static input_parameter_controller parameters() { return base::parameters(); }
  void compute() { copy = ref.get(); }

  nm::input_property<tensor2, nm::property_traits> const& ref;
  tensor2& copy;
};

class fibre_probe final : public nm::material_base<fibre_probe, policy> {
public:
  using base = nm::material_base<fibre_probe, policy>;
  template <typename... Args>
  explicit fibre_probe(Args&&... args)
      : base(std::forward<Args>(args)...), f(add_history_output<fibre>("fibre", fibre_codec{})) {}
  static input_parameter_controller parameters() { return base::parameters(); }

  nm::history_property<fibre>& f;
};

template <typename Probe>
Probe& create(ctx_type& ctx, std::string const& name = "probe") {
  policy::ParameterHandler p;
  p.insert<std::string>("name", name);
  return ctx.create<Probe>(p);
}

ctx_type& finalized(ctx_type& ctx) {
  ctx.finalize();
  return ctx;
}

/// A finalized codec_probe, its STATEV map and a zeroed STATEV.
struct codec_fixture {
  ctx_type ctx;
  codec_probe& p{create<codec_probe>(ctx)};
  map_type map{finalized(ctx), {}};
  std::vector<T> statev = std::vector<T>(map.nstatv(), 0.0);
};

const tensor2 sym{1.0, 0.2, 0.3, 0.2, 2.0, 0.4, 0.3, 0.4, 3.0};
const tensor2 nonsym{1.1, 0.2, 0.05, 0.03, 0.95, 0.1, 0.07, 0.02, 1.05};
const tensor1 vec{1.0, 2.0, 3.0};

tensor2 rotation_z(T radians) {
  tensor1 axis;
  axis(2) = 1.0;
  return nm::rotation(axis, radians);
}

template <std::size_t Rank, typename A, typename B>
T distance(A const& a, B const& b) {
  return tmech::norm(tmech::tensor<T, 3, Rank>{a - b});
}

/// Sets both sides, as a converged step followed by unpack would.
template <typename H, typename V>
void set_history(H& h, V const& v) {
  h.old_value() = v;
  h.new_value() = v;
}

/// The fatal_error message `f` throws, or a note that it threw none.
template <typename F>
std::string fatal_message(F&& f) {
  try {
    f();
  } catch (nm::fatal_error const& e) {
    return e.what();
  }
  return "no fatal_error";
}

}  // namespace

TEST(HistoryCodecs, WidthsAndDescriptionFollowTheCodec) {
  codec_fixture f;
  EXPECT_EQ(f.map.nstatv(), 6u + 3u + 6u + 3u + 9u + 9u + 6u + 1u + 36u);
  const std::vector<std::string> expected{
      "0..5  probe::a_ref  [engineering shear] [reference]",
      "6..8  probe::b_rvec  [vector] [reference]",
      "9..14  probe::c_strain  [engineering shear]",
      "15..17  probe::d_vec  [vector]",
      "18..26  probe::e_fe  [full] [two-point]",
      "27..35  probe::f_fp  [full] [reference]",
      "36..41  probe::g_stress  [stress]",
      "42..42  probe::h_scalar",
      "43..78  probe::i_moduli"};
  EXPECT_EQ(f.map.describe(), expected);
}

TEST(HistoryCodecs, DrotTurnsSpatialHistoryAndLeavesReferenceHistory) {
  codec_fixture f;
  auto& p = f.p;
  EXPECT_TRUE(f.map.has_rotatable_history());
  for (auto* h : {&p.a_ref, &p.c_strain, &p.g_stress}) set_history(*h, sym);
  for (auto* h : {&p.e_fe, &p.f_fp}) set_history(*h, nonsym);
  for (auto* h : {&p.b_rvec, &p.d_vec}) set_history(*h, vec);
  set_history(p.h_scalar, 1.5);
  set_history(p.i_moduli, tensor4{tmech::otimes(sym, sym)});

  const auto R = rotation_z(0.5);
  f.map.rotate_history(R);

  const tensor2 RSRt{R * sym * tmech::trans(R)};
  const tensor4 RC{tmech::otimes(RSRt, RSRt)};  // R(S⊗S) = (RSRᵀ)⊗(RSRᵀ)
  for (auto side : {0, 1}) {
    auto get = [side](auto& h) -> auto const& { return side == 0 ? h.old_value() : h.new_value(); };
    EXPECT_EQ(distance<2>(get(p.a_ref), sym), 0.0);
    EXPECT_EQ(distance<1>(get(p.b_rvec), vec), 0.0);
    EXPECT_LT(distance<2>(get(p.c_strain), RSRt), 1e-14);
    EXPECT_LT(distance<1>(get(p.d_vec), tensor1{R * vec}), 1e-14);
    EXPECT_LT(distance<2>(get(p.e_fe), tensor2{R * nonsym}), 1e-14);
    EXPECT_EQ(distance<2>(get(p.f_fp), nonsym), 0.0);
    EXPECT_LT(distance<2>(get(p.g_stress), RSRt), 1e-14);
    EXPECT_EQ(get(p.h_scalar), 1.5);
    EXPECT_LT(distance<4>(get(p.i_moduli), RC), 1e-12);
  }
  EXPECT_GT(distance<2>(RSRt, sym), 0.1) << "the rotation must be observable";
}

TEST(HistoryCodecs, FullCodecsKeepEveryComponent) {
  codec_fixture f;
  auto& p = f.p;
  const tensor2 Ft{tmech::trans(nonsym)};
  p.e_fe.new_value() = nonsym;
  p.f_fp.new_value() = Ft;
  p.b_rvec.new_value() = vec;
  f.map.pack(f.statev.data());
  // Row-major {11 12 13 21 22 23 31 32 33}.
  EXPECT_DOUBLE_EQ(f.statev[18 + 1], nonsym(0, 1));
  EXPECT_DOUBLE_EQ(f.statev[18 + 3], nonsym(1, 0));
  EXPECT_DOUBLE_EQ(f.statev[27 + 1], nonsym(1, 0));
  EXPECT_DOUBLE_EQ(f.statev[27 + 7], nonsym(1, 2));
  EXPECT_DOUBLE_EQ(f.statev[6 + 2], vec(2));

  for (auto* h : {&p.e_fe, &p.f_fp}) set_history(*h, tensor2{});
  set_history(p.b_rvec, tensor1{});
  f.map.unpack(f.statev.data());
  for (auto side : {0, 1}) {
    auto get = [side](auto& h) -> auto const& { return side == 0 ? h.old_value() : h.new_value(); };
    EXPECT_EQ(distance<2>(get(p.e_fe), nonsym), 0.0);
    EXPECT_EQ(distance<2>(get(p.f_fp), Ft), 0.0);
    EXPECT_EQ(distance<1>(get(p.b_rvec), vec), 0.0);
  }
}

TEST(HistoryCodecs, StressStoresShearAsIsNextToEngineeringStrain) {
  codec_fixture f;
  const tensor2 shear{0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (auto* h : {&f.p.a_ref, &f.p.c_strain, &f.p.g_stress}) h->new_value() = shear;
  f.map.pack(f.statev.data());
  // Abaqus order {11,22,33,12,13,23}: slot 3 is the 12 component.
  EXPECT_DOUBLE_EQ(f.statev[36 + 3], 0.1);
  EXPECT_DOUBLE_EQ(f.statev[9 + 3], 0.2);
  EXPECT_DOUBLE_EQ(f.statev[0 + 3], 0.2);

  set_history(f.p.g_stress, tensor2{});
  f.map.unpack(f.statev.data());
  EXPECT_EQ(distance<2>(f.p.g_stress.old_value(), shear), 0.0);
}

TEST(HistoryCodecs, SymmetricCodecsRejectASkewPartNamingTheHistory) {
  codec_fixture f;
  for (auto* h : {&f.p.a_ref, &f.p.c_strain, &f.p.g_stress}) {
    h->new_value() = nonsym;
    const auto message = fatal_message([&] { f.map.pack(f.statev.data()); });
    EXPECT_NE(message.find("probe::" + h->traits().id.name), std::string::npos) << message;
    h->new_value() = tensor2{};
  }
}

TEST(HistoryCodecs, ReferenceAndScalarHistoryIsNotRotatable) {
  ctx_type ctx;
  create<codec_probe>(ctx);
  // Leaves a_ref, b_rvec, f_fp and h_scalar.
  const map_type map(finalized(ctx), {{"probe", "c_strain"}, {"probe", "d_vec"}, {"probe", "e_fe"},
                                      {"probe", "g_stress"}, {"probe", "i_moduli"}});
  EXPECT_EQ(map.nstatv(), 6u + 3u + 9u + 1u);
  EXPECT_FALSE(map.has_rotatable_history());
}

/// The codec lives at the declaration, so a consumer reads the history as the plain tensor it is.
TEST(HistoryCodecs, AConsumerReadsAReferenceStrainThroughAPlainTensorInput) {
  ctx_type ctx;
  auto& probe = create<codec_probe>(ctx);
  auto& reader = create<reference_reader>(ctx, "reader");
  ASSERT_NO_THROW(ctx.finalize());
  probe.a_ref.new_value() = sym;
  ctx.update();
  EXPECT_EQ(distance<2>(reader.copy, sym), 0.0);
}

TEST(HistoryCodecs, AUserCodecRoundTripsAUserType) {
  ctx_type ctx;
  auto& p = create<fibre_probe>(ctx);
  const map_type map(finalized(ctx), {});
  EXPECT_EQ(map.describe(), std::vector<std::string>{"0..3  probe::fibre  [fibre]"});
  EXPECT_TRUE(map.has_rotatable_history());

  p.f.new_value() = {vec, 2.5};
  std::vector<T> statev(map.nstatv(), 0.0);
  map.pack(statev.data());
  EXPECT_EQ(statev, (std::vector<T>{1.0, 2.0, 3.0, 2.5}));

  p.f.new_value() = {};
  map.unpack(statev.data());
  const auto R = rotation_z(0.5);
  map.rotate_history(R);
  EXPECT_LT(distance<1>(p.f.old_value().direction, tensor1{R * vec}), 1e-14);
  EXPECT_EQ(p.f.new_value().length, 2.5);
}

TEST(HistoryCodecs, AHistoryWithoutACodecIsRefusedByName) {
  ctx_type ctx;
  create<nm::test::int_counter<policy>>(ctx, "weird");
  finalized(ctx);
  const auto message = fatal_message([&] { const map_type map(ctx, {}); });
  EXPECT_NE(message.find("weird::count"), std::string::npos) << message;
  EXPECT_NE(message.find("codec"), std::string::npos) << message;
}
