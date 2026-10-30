// Residual systems set up at run time: unknowns and equations from any nodes (#218).
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/errors.h"
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/default_materials.h"
#include "numsim-materials/driver/material_point.h"
#include "numsim-materials/solvers/system.h"
#include "numsim-materials/solvers/vector_newton.h"
#include "numsim-materials/umat/external_state_source.h"

#if __has_include(<nlohmann/json.hpp>)
#define NUMSIM_HAVE_JSON 1
#include <nlohmann/json.hpp>
#include "numsim-materials/io/json_material_factory.h"
#endif

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using system_type = nm::residual_system<policy>;
using solver_type = nm::vector_newton<policy>;

enum class system_mode { linear, nonlinear, singular, decoupled };

/// The rows "x" ("rx"), "y" ("ry") or both of the two-scalar systems of test_vector_newton, with their partials.
/// The x row of the decoupled system does not read y at all.
class scalar_rows final : public nm::material_base<scalar_rows, policy> {
public:
  using base = nm::material_base<scalar_rows, policy>;

  template <typename... Args>
  explicit scalar_rows(Args &&...args)
      : base(std::forward<Args>(args)...), m_mode(static_cast<system_mode>(base::get_parameter<int>("mode"))),
        m_a(base::get_parameter<T>("a")), m_b(base::get_parameter<T>("b")), m_rows(base::get_parameter<std::string>("rows")),
        m_x(base::add_input<T>(base::get_parameter<std::string>("solver"), "x")),
        m_y(m_rows == "x" && m_mode == system_mode::decoupled ? nullptr
                                                               : &base::add_input<T>(base::get_parameter<std::string>("solver"), "y")) {
    if (m_rows != "y")
      m_rx = &base::add_output<T>("rx");
    if (m_rows != "x")
      m_ry = &base::add_output<T>("ry");
    auto handle{base::add_callback(&scalar_rows::compute, m_rx, m_ry)};
    handle.reads(m_x, m_y);
    if (m_rx)
      handle.partials(*m_rx, m_x, m_y);
    if (m_ry)
      handle.partials(*m_ry, m_x, m_y);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<std::string>(para, "solver", std::string{"solver"});
    nm::optional<std::string>(para, "rows", std::string{"xy"});
    nm::optional<int>(para, "mode", 0);
    nm::optional<T>(para, "a", T{0});
    nm::optional<T>(para, "b", T{0});
    return para;
  }

  void compute() {
    T const x{m_x.get()}, y{m_y ? m_y->get() : T{0}};
    T rx{}, ry{}, jxx{}, jxy{}, jyx{}, jyy{};
    switch (m_mode) {
    case system_mode::linear:
      rx = 2 * x + y - m_a, jxx = 2, jxy = 1;
      ry = x + 3 * y - m_b, jyx = 1, jyy = 3;
      break;
    case system_mode::nonlinear:
      rx = x * x + y - m_a, jxx = 2 * x, jxy = 1;
      ry = x + y * y - m_b, jyx = 1, jyy = 2 * y;
      break;
    case system_mode::singular:
      rx = x + y - m_a, jxx = 1, jxy = 1;
      ry = x + y - m_b, jyx = 1, jyy = 1;
      break;
    case system_mode::decoupled:
      rx = 2 * x - m_a, jxx = 2, jxy = 0;
      ry = x + 3 * y - m_b, jyx = 1, jyy = 3;
      break;
    }
    if (m_rx)
      *m_rx = rx;
    if (m_ry)
      *m_ry = ry;
    if (!base::want_partials())
      return;
    if (m_rx) {
      base::partial(*m_rx, m_x) = jxx;
      if (m_y)
        base::partial(*m_rx, *m_y) = jxy;
    }
    if (m_ry) {
      base::partial(*m_ry, m_x) = jyx;
      base::partial(*m_ry, *m_y) = jyy;
    }
  }

private:
  system_mode m_mode;
  T const &m_a;
  T const &m_b;
  std::string const &m_rows;
  nm::input_property<T, nm::property_traits> const &m_x;
  nm::input_property<T, nm::property_traits> const *m_y;
  T *m_rx{nullptr};
  T *m_ry{nullptr};
};

/// R_g = g + tr(B) - a,  R_B = B - g P + c g^2 I, as mixed_system_t of test_vector_newton, with its partials.
template <typename Policy>
class mixed_rows final : public nm::material_base<mixed_rows<Policy>, Policy> {
public:
  using base = nm::material_base<mixed_rows<Policy>, Policy>;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto D = base::Dim;
  using t2 = tmech::tensor<T, D, 2>;
  using t4 = tmech::tensor<T, D, 4>;

  template <typename... Args>
  explicit mixed_rows(Args &&...args)
      : base(std::forward<Args>(args)...), m_rg(base::template add_output<T>("rg")), m_rB(base::template add_output<t2>("rB")),
        m_a(base::template get_parameter<T>("a")), m_c(base::template get_parameter<T>("c")),
        m_g(base::template add_input<T>("solver", "g")), m_B(base::template add_input<t2>("solver", "B")) {
    if constexpr (D == 2)
      m_P = t2{0.5, 0.1, 0.1, 0.7};
    else
      m_P = t2{0.5, 0.1, 0.2, 0.1, 0.7, 0.3, 0.2, 0.3, 0.9};
    base::add_callback(&mixed_rows::compute, m_rg, m_rB).reads(m_g, m_B).partials(m_rg, m_g, m_B).partials(m_rB, m_g, m_B);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<T>(para, "a", T{1});
    nm::optional<T>(para, "c", T{0});
    return para;
  }

  t2 const &P() const noexcept { return m_P; }

  void compute() {
    auto const g{m_g.get()};
    auto const &B{m_B.get()};
    auto const I{tmech::eye<T, D, 2>()};
    m_rg = g + tmech::trace(B) - m_a;
    m_rB = B - g * m_P + (m_c * g * g) * I;
    if (!base::want_partials())
      return;
    base::partial(m_rg, m_g) = T{1};
    base::partial(m_rg, m_B) = t2{I};
    base::partial(m_rB, m_g) = t2{-m_P + (2 * m_c * g) * I};
    base::partial(m_rB, m_B) = nm::sym_identity<T, D>();
  }

private:
  T &m_rg;
  t2 &m_rB;
  T const &m_a;
  T const &m_c;
  nm::input_property<T, nm::property_traits> const &m_g;
  nm::input_property<t2, nm::property_traits> const &m_B;
  t2 m_P{};
};

/// R_g = g + tr(B) - a - tr(eps),  R_B = B - g P, with eps from strain::strain: strain_coupled_system of
/// test_vector_newton with the strain as a seed instead of a member.
class strain_rows final : public nm::material_base<strain_rows, policy> {
public:
  using base = nm::material_base<strain_rows, policy>;

  template <typename... Args>
  explicit strain_rows(Args &&...args)
      : base(std::forward<Args>(args)...), m_rg(base::add_output<T>("rg")), m_rB(base::add_output<tensor2>("rB")),
        m_a(base::get_parameter<T>("a")), m_g(base::add_input<T>("solver", "g")), m_B(base::add_input<tensor2>("solver", "B")),
        m_eps(base::add_input<tensor2>("strain", "strain")) {
    base::add_callback(&strain_rows::compute, m_rg, m_rB)
        .reads(m_g, m_B, m_eps)
        .partials(m_rg, m_g, m_B, m_eps)
        .partials(m_rB, m_g, m_B, m_eps);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<T>(para, "a", T{2});
    return para;
  }

  static tensor2 P() { return tensor2{0.5, 0.2, 0.1, 0.2, 0.7, 0.3, 0.1, 0.3, 0.9}; }

  void compute() {
    auto const I{tmech::eye<T, 3, 2>()};
    m_rg = m_g.get() + tmech::trace(m_B.get()) - m_a - tmech::trace(m_eps.get());
    m_rB = m_B.get() - m_g.get() * P();
    if (!base::want_partials())
      return;
    base::partial(m_rg, m_g) = T{1};
    base::partial(m_rg, m_B) = tensor2{I};
    base::partial(m_rg, m_eps) = tensor2{-I};
    base::partial(m_rB, m_g) = tensor2{-P()};
    base::partial(m_rB, m_B) = nm::sym_identity<T, 3>();
    base::partial(m_rB, m_eps) = tensor4{};
  }

private:
  T &m_rg;
  tensor2 &m_rB;
  T const &m_a;
  nm::input_property<T, nm::property_traits> const &m_g;
  nm::input_property<tensor2, nm::property_traits> const &m_B;
  nm::input_property<tensor2, nm::property_traits> const &m_eps;
};

/// r = x^3 + x - load::state for the unknown x of "solver"; with "picard", r = x - cos(load + x) / 2.
class cubic_row final : public nm::material_base<cubic_row, policy> {
public:
  using base = nm::material_base<cubic_row, policy>;

  template <typename... Args>
  explicit cubic_row(Args &&...args)
      : base(std::forward<Args>(args)...), m_r(base::add_output<T>("r")), m_picard(base::get_parameter<bool>("picard")),
        m_x(base::add_input<T>("solver", "x")), m_load(base::add_input<T>("load", "state")) {
    auto handle{base::add_callback(&cubic_row::compute, m_r).reads(m_x, m_load)};
    if (base::get_parameter<bool>("declares_partials"))
      handle.partials(m_r, m_x, m_load);
    if (base::get_parameter<bool>("counts"))
      handle.stateful();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<bool>(para, "picard", false);
    nm::optional<bool>(para, "declares_partials", true);
    nm::optional<bool>(para, "counts", false);
    return para;
  }

  void compute() {
    auto const x{m_x.get()}, load{m_load.get()};
    m_r = m_picard ? x - std::cos(load + x) / 2 : x * x * x + x - load;
    if (base::want_partials()) {
      base::partial(m_r, m_x) = m_picard ? 1 + std::sin(load + x) / 2 : 3 * x * x + 1;
      base::partial(m_r, m_load) = m_picard ? std::sin(load + x) / 2 : T{-1};
    }
  }

private:
  T &m_r;
  bool const &m_picard;
  nm::input_property<T, nm::property_traits> const &m_x;
  nm::input_property<T, nm::property_traits> const &m_load;
};

/// r = x^2 - load::state for the unknown x of "solver": a double root, and a singular Jacobian, at load 0.
class square_row final : public nm::material_base<square_row, policy> {
public:
  using base = nm::material_base<square_row, policy>;

  template <typename... Args>
  explicit square_row(Args &&...args)
      : base(std::forward<Args>(args)...), m_r(base::add_output<T>("r")), m_x(base::add_input<T>("solver", "x")),
        m_load(base::add_input<T>("load", "state")) {
    base::add_callback(&square_row::compute, m_r).reads(m_x, m_load).partials(m_r, m_x, m_load);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    m_r = m_x.get() * m_x.get() - m_load.get();
    if (base::want_partials()) {
      base::partial(m_r, m_x) = 2 * m_x.get();
      base::partial(m_r, m_load) = T{-1};
    }
  }

private:
  T &m_r;
  nm::input_property<T, nm::property_traits> const &m_x;
  nm::input_property<T, nm::property_traits> const &m_load;
};

/// r = t^3 + t - 2 with t = x / scale for the unknown x of "solver": the root x = scale, whatever the scale.
class scaled_row final : public nm::material_base<scaled_row, policy> {
public:
  using base = nm::material_base<scaled_row, policy>;

  template <typename... Args>
  explicit scaled_row(Args &&...args)
      : base(std::forward<Args>(args)...), m_r(base::add_output<T>("r")), m_scale(base::get_parameter<T>("scale")),
        m_x(base::add_input<T>("solver", "x")) {
    base::add_callback(&scaled_row::compute, m_r).reads(m_x).partials(m_r, m_x);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<T>(para, "scale", T{1});
    return para;
  }

  void compute() {
    T const t{m_x.get() / m_scale};
    m_r = t * t * t + t - 2;
    if (base::want_partials())
      base::partial(m_r, m_x) = (3 * t * t + 1) / m_scale;
  }

private:
  T &m_r;
  T const &m_scale;
  nm::input_property<T, nm::property_traits> const &m_x;
};

/// r = "value" whatever x is: a residual that is NaN or infinite.
class poison_row final : public nm::material_base<poison_row, policy> {
public:
  using base = nm::material_base<poison_row, policy>;

  template <typename... Args>
  explicit poison_row(Args &&...args)
      : base(std::forward<Args>(args)...), m_r(base::add_output<T>("r")), m_value(base::get_parameter<T>("value")),
        m_x(base::add_input<T>("solver", "x")) {
    base::add_callback(&poison_row::compute, m_r).reads(m_x);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<T>(para, "value", T{0});
    return para;
  }

  void compute() { m_r = m_value + (m_x.get() - m_x.get()); }

private:
  T &m_r;
  T const &m_value;
  nm::input_property<T, nm::property_traits> const &m_x;
};

/// r = y^3 + y - solver::x for the unknown y of "second": a system reading the unknown of another.
class chain_row final : public nm::material_base<chain_row, policy> {
public:
  using base = nm::material_base<chain_row, policy>;

  template <typename... Args>
  explicit chain_row(Args &&...args)
      : base(std::forward<Args>(args)...), m_r(base::add_output<T>("r")), m_y(base::add_input<T>("second", "y")),
        m_x(base::add_input<T>("solver", "x")) {
    base::add_callback(&chain_row::compute, m_r).reads(m_y, m_x).partials(m_r, m_y, m_x);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    T const y{m_y.get()};
    m_r = y * y * y + y - m_x.get();
    if (base::want_partials()) {
      base::partial(m_r, m_y) = 3 * y * y + 1;
      base::partial(m_r, m_x) = T{-1};
    }
  }

private:
  T &m_r;
  nm::input_property<T, nm::property_traits> const &m_y;
  nm::input_property<T, nm::property_traits> const &m_x;
};

/// A residual system "solver" over `unknowns` and `equations`.
system_type &make_system(ctx_type &ctx, std::vector<nm::system_unknown> unknowns, std::vector<nm::system_equation> equations,
                         std::string const &method = "newton", int max_iter = 50, T tolerance = 1e-10,
                         std::string const &name = "solver") {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<std::string>("method", method);
  p.insert<std::vector<nm::system_unknown>>("unknowns", std::move(unknowns));
  p.insert<std::vector<nm::system_equation>>("equations", std::move(equations));
  p.insert<int>("max_iter", max_iter);
  p.insert<T>("tolerance", tolerance);
  return ctx.create<system_type>(p);
}

std::vector<nm::system_unknown> xy() { return {{"x", nm::unknown_kind::scalar}, {"y", nm::unknown_kind::scalar}}; }

/// x, y of the scalar system in one node "sys" (or split into "fx" and "fy"), solved by a system.
struct scalar_rig {
  std::unique_ptr<ctx_type> owner{std::make_unique<ctx_type>()};
  ctx_type &ctx{*owner};
  system_type *system{nullptr};

  scalar_rig(system_mode mode, T a, T b, int max_iter = 50, bool split = false, std::string const &method = "newton") {
    std::vector<nm::system_equation> equations{split ? std::vector<nm::system_equation>{{"fx::rx", "x"}, {"fy::ry", "y"}}
                                                     : std::vector<nm::system_equation>{{"sys::ry", "y"}, {"sys::rx", "x"}}};
    system = &make_system(ctx, xy(), equations, method, max_iter);
    for (auto const &[name, rows] : split ? std::vector<std::pair<std::string, std::string>>{{"fx", "x"}, {"fy", "y"}}
                                          : std::vector<std::pair<std::string, std::string>>{{"sys", "xy"}}) {
      param_type p;
      p.insert<std::string>("name", name);
      p.insert<std::string>("rows", rows);
      p.insert<int>("mode", static_cast<int>(mode));
      p.insert<T>("a", a);
      p.insert<T>("b", b);
      ctx.create<scalar_rows>(p);
    }
    ctx.finalize();
  }

  T x() { return ctx.get<T>("solver", "x"); }
  T y() { return ctx.get<T>("solver", "y"); }
};

/// vector_newton on the same scalar system: the root and the fewest max_iter it converges with.
struct newton_reference {
  T x{}, y{};
  int iterations{-1};
};

class scalar_system_2 final : public nm::material_base<scalar_system_2, policy> {
public:
  using base = nm::material_base<scalar_system_2, policy>;

  template <typename... Args>
  explicit scalar_system_2(Args &&...args)
      : base(std::forward<Args>(args)...), m_rx(base::add_output<T>("residual_x")), m_ry(base::add_output<T>("residual_y")),
        m_jxx(base::add_output<T>("jacobian_x_x")), m_jxy(base::add_output<T>("jacobian_x_y")),
        m_jyx(base::add_output<T>("jacobian_y_x")), m_jyy(base::add_output<T>("jacobian_y_y")),
        m_rows(static_cast<system_mode>(base::get_parameter<int>("mode")), base::get_parameter<T>("a"), base::get_parameter<T>("b")),
        m_x(base::add_input<T>("solver", "x", nm::EdgeKind::Local)), m_y(base::add_input<T>("solver", "y", nm::EdgeKind::Local)) {
    base::add_callback(&scalar_system_2::compute, m_rx, m_ry, m_jxx, m_jxy, m_jyx, m_jyy).reads(m_x, m_y);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<int>(para, "mode", 0);
    nm::optional<T>(para, "a", T{0});
    nm::optional<T>(para, "b", T{0});
    return para;
  }

  void compute() {
    auto const [mode, a, b]{m_rows};
    T const x{m_x.get()}, y{m_y.get()};
    switch (mode) {
    case system_mode::linear:
      m_rx = 2 * x + y - a, m_jxx = 2, m_jxy = 1;
      m_ry = x + 3 * y - b, m_jyx = 1, m_jyy = 3;
      break;
    case system_mode::nonlinear:
      m_rx = x * x + y - a, m_jxx = 2 * x, m_jxy = 1;
      m_ry = x + y * y - b, m_jyx = 1, m_jyy = 2 * y;
      break;
    case system_mode::singular:
      m_rx = x + y - a, m_jxx = 1, m_jxy = 1;
      m_ry = x + y - b, m_jyx = 1, m_jyy = 1;
      break;
    case system_mode::decoupled:
      m_rx = 2 * x - a, m_jxx = 2, m_jxy = 0;
      m_ry = x + 3 * y - b, m_jyx = 1, m_jyy = 3;
      break;
    }
  }

private:
  T &m_rx, &m_ry, &m_jxx, &m_jxy, &m_jyx, &m_jyy;
  std::tuple<system_mode, T, T> m_rows;
  nm::input_property<T, nm::property_traits> const &m_x;
  nm::input_property<T, nm::property_traits> const &m_y;
};

newton_reference vector_newton_on(system_mode mode, T a, T b) {
  newton_reference ref;
  for (int max_iter{0}; max_iter <= 50 && ref.iterations < 0; ++max_iter) {
    ctx_type ctx;
    param_type p;
    p.insert<std::string>("name", "solver");
    p.insert<std::string>("function", "sys");
    p.insert<int>("max_iter", max_iter);
    p.insert<bool>("graph_driven", false);
    p.insert<std::vector<nm::unknown_spec>>("unknowns", {{"x", nm::unknown_kind::scalar}, {"y", nm::unknown_kind::scalar}});
    auto &solver{ctx.create<solver_type>(p)};
    p.clear();
    p.insert<std::string>("name", "sys");
    p.insert<int>("mode", static_cast<int>(mode));
    p.insert<T>("a", a);
    p.insert<T>("b", b);
    ctx.create<scalar_system_2>(p);
    ctx.finalize();
    solver.solve();
    if (solver.converged())
      ref = {ctx.get<T>("solver", "x"), ctx.get<T>("solver", "y"), max_iter};
  }
  return ref;
}

/// The message of the exception of type E that `f` throws; empty when it throws none.
template <typename E, typename F>
std::string thrown(F &&f) {
  try {
    f();
  } catch (E const &e) {
    return e.what();
  }
  return {};
}

// --- the cases of test_vector_newton, as systems ---

TEST(ResidualSystem, TakesTheIterationsOfVectorNewtonAndLandsOnItsRoot) {
  struct problem {
    system_mode mode;
    T a, b;
  };
  for (auto const &[mode, a, b] : {problem{system_mode::linear, 5, 10}, problem{system_mode::nonlinear, 3, 5},
                                   problem{system_mode::nonlinear, 5.248, 27.4}, problem{system_mode::decoupled, 5, 10}}) {
    auto const ref{vector_newton_on(mode, a, b)};
    ASSERT_GE(ref.iterations, 0);
    scalar_rig rig{mode, a, b};
    rig.ctx.update();
    EXPECT_EQ(rig.system->iterations(), ref.iterations) << static_cast<int>(mode);
    EXPECT_NEAR(rig.x(), ref.x, 1e-14);
    EXPECT_NEAR(rig.y(), ref.y, 1e-14);
    if (ref.iterations > 0) {
      scalar_rig short_of_it{mode, a, b, ref.iterations - 1};
      EXPECT_THROW(short_of_it.ctx.update(), nm::cutback_error) << "max_iter counts Newton updates, as in vector_newton";
    }
  }
}

TEST(ResidualSystem, LinearSystemIsExactAndConvergesOnExactlyMaxIter) {
  scalar_rig rig{system_mode::linear, 5, 10, 1};
  EXPECT_NO_THROW(rig.ctx.update());
  EXPECT_EQ(rig.system->iterations(), 1);
  EXPECT_NEAR(rig.x(), 1, 1e-12);
  EXPECT_NEAR(rig.y(), 3, 1e-12);
}

TEST(ResidualSystem, NonConvergenceThrowsACutbackNamingTheSystem) {
  scalar_rig rig{system_mode::nonlinear, 3, 5, 0};
  auto const message{thrown<nm::cutback_error>([&] { rig.ctx.update(); })};
  EXPECT_TRUE(message.contains("'solver'")) << message;
}

TEST(ResidualSystem, ASingularJacobianThrowsACutbackAndLeavesTheUnknownsFinite) {
  scalar_rig rig{system_mode::singular, 4, 7};
  EXPECT_THROW(rig.ctx.update(), nm::cutback_error);
  EXPECT_TRUE(std::isfinite(rig.x()) && std::isfinite(rig.y()));
}

TEST(ResidualSystem, SolvesArePointLocal) {
  scalar_rig fresh{system_mode::nonlinear, 3, 5};
  fresh.ctx.update();
  scalar_rig shared{system_mode::nonlinear, 5.248, 27.4};
  shared.ctx.update();
  auto *sys{shared.ctx.find("sys")};
  sys->set_parameter<T>("a", 3);
  sys->set_parameter<T>("b", 5);
  shared.ctx.update();
  EXPECT_DOUBLE_EQ(shared.x(), fresh.x()) << "the seed is 0, not the previous root";
  EXPECT_DOUBLE_EQ(shared.y(), fresh.y());
}

TEST(ResidualSystem, AFailedSolveDoesNotMoveTheNext) {
  scalar_rig reference{system_mode::nonlinear, 3, 5};
  reference.ctx.update();
  scalar_rig rig{system_mode::nonlinear, -10, -10};
  EXPECT_THROW(rig.ctx.update(), nm::cutback_error);
  rig.ctx.revert();
  rig.ctx.find("sys")->set_parameter<T>("a", 3);
  rig.ctx.find("sys")->set_parameter<T>("b", 5);
  rig.ctx.update();
  EXPECT_DOUBLE_EQ(rig.x(), reference.x());
}

template <typename Policy>
void mixed_case(T c, T tolerance) {
  using t2 = tmech::tensor<T, Policy::Dim, 2>;
  nm::material_context<Policy> ctx;
  param_type p;
  p.insert<std::string>("name", "solver");
  p.insert<std::vector<nm::system_unknown>>("unknowns", {{"g", nm::unknown_kind::scalar}, {"B", nm::unknown_kind::sym_tensor}});
  p.insert<std::vector<nm::system_equation>>("equations", {{"sys::rB", "B"}, {"sys::rg", "g"}});
  auto &system{ctx.template create<nm::residual_system<Policy>>(p)};
  p.clear();
  p.insert<std::string>("name", "sys");
  p.insert<T>("a", 2);
  p.insert<T>("c", c);
  auto &sys{ctx.template create<mixed_rows<Policy>>(p)};
  ctx.finalize();
  ctx.update();
  T const trP{tmech::trace(sys.P())}, s{1 + trP}, D{T(Policy::Dim)};
  T const g{c == 0 ? 2 / s : (s - std::sqrt(s * s - 4 * D * c * 2)) / (2 * D * c)};
  EXPECT_NEAR(ctx.template get<T>("solver", "g"), g, tolerance);
  t2 const B{g * sys.P() - (c * g * g) * tmech::eye<T, Policy::Dim, 2>()};
  EXPECT_TRUE(tmech::almost_equal(ctx.template get<t2>("solver", "B"), B, tolerance));
  if (c == 0)
    EXPECT_EQ(system.iterations(), 1) << "linear: one update";
  else
    EXPECT_EQ(system.iterations(), 4) << "2.0 -> 4.2e-2 -> 2.1e-4 -> 5.6e-9 -> 2.2e-16, as vector_newton measured";
}

TEST(ResidualSystem, MixedScalarAndSymmetricTensorSystem) { mixed_case<policy>(0, 1e-10); }
TEST(ResidualSystem, NonlinearMixedSystemIteratesQuadratically) { mixed_case<policy>(0.1, 1e-12); }
TEST(ResidualSystem, MixedSystemIn2D) { mixed_case<nm::material_policy_2d>(0, 1e-10); }

// --- equations from any nodes ---

TEST(ResidualSystem, SplitOverTwoResidualNodesConvergesIdentically) {
  for (auto const mode : {system_mode::linear, system_mode::nonlinear, system_mode::decoupled}) {
    T const a{mode == system_mode::nonlinear ? T{3} : T{5}}, b{mode == system_mode::nonlinear ? T{5} : T{10}};
    scalar_rig one{mode, a, b};
    scalar_rig split{mode, a, b, 50, true};
    one.ctx.update();
    split.ctx.update();
    EXPECT_EQ(split.system->iterations(), one.system->iterations());
    EXPECT_EQ(split.x(), one.x());
    EXPECT_EQ(split.y(), one.y());
  }
}

TEST(ResidualSystem, StructuralZeroBlocksAreTheBlocksWithoutAPath) {
  scalar_rig split{system_mode::decoupled, 5, 10, 50, true};
  EXPECT_EQ(split.system->zero_blocks(), (std::vector<nm::block_ref>{{"x", "y"}}))
      << "what vector_newton declared as zero_blocks: [[\"x\", \"y\"]]";
  EXPECT_TRUE(scalar_rig(system_mode::linear, 5, 10, 50, true).system->zero_blocks().empty());
  EXPECT_TRUE(scalar_rig(system_mode::decoupled, 5, 10).system->zero_blocks().empty()) << "one node reads both";
  split.ctx.update();
  EXPECT_NEAR(split.x(), 2.5, 1e-12);
  EXPECT_NEAR(split.y(), 2.5, 1e-12);
}

TEST(ResidualSystem, NewtonFdFindsTheSameRoot) {
  scalar_rig fd{system_mode::nonlinear, 3, 5, 50, false, "newton_fd"};
  fd.ctx.update();
  EXPECT_NEAR(fd.x(), 1, 1e-10);
  EXPECT_NEAR(fd.y(), 2, 1e-10);
}

// --- the implicit function theorem through the system ---

struct strain_rig {
  std::unique_ptr<ctx_type> owner{std::make_unique<ctx_type>()};
  ctx_type &ctx{*owner};
  std::unique_ptr<nm::material_point<policy>> point;

  explicit strain_rig(std::string const &method = "newton") {
    param_type p;
    p.insert<std::string>("name", "strain");
    ctx.create<nm::external_strain_source<policy>>(p);
    make_system(ctx, {{"g", nm::unknown_kind::scalar}, {"B", nm::unknown_kind::sym_tensor}}, {{"sys::rg", "g"}, {"sys::rB", "B"}},
                method);
    p.clear();
    p.insert<std::string>("name", "sys");
    ctx.create<strain_rows>(p);
    ctx.finalize();
    param_type q;
    q.insert<std::vector<std::string>>("inputs", {"strain"});
    point = std::make_unique<nm::material_point<policy>>(ctx, q);
  }

  void evaluate(tensor2 const &eps) {
    point->reset();
    point->input<nm::external_strain_source<policy>>("strain").bind(tensor2{}, eps);
    point->evaluate();
  }
};

tensor2 const eps0{0.03, 0.01, 0.00, 0.01, 0.05, 0.02, 0.00, 0.02, 0.07};

TEST(ResidualSystemTangent, MatchesTheAnalyticStrainDerivative) {
  strain_rig rig;
  auto const &dg{rig.point->derivative<tensor2>("solver::g", "strain")};
  auto const &dB{rig.point->derivative<tensor4>("solver::B", "strain")};
  rig.evaluate(eps0);
  T const denom{1 + tmech::trace(strain_rows::P())};
  EXPECT_NEAR(rig.ctx.get<T>("solver", "g"), (2 + tmech::trace(eps0)) / denom, 1e-12);
  EXPECT_TRUE(tmech::almost_equal(dg, tensor2{tmech::eye<T, 3, 2>() / denom}, 1e-12));
  tensor4 const dB_ref{tmech::otimes(strain_rows::P(), tmech::eye<T, 3, 2>()) / denom};
  EXPECT_TRUE(tmech::almost_equal(dB, dB_ref, 1e-12)) << "minor-symmetric, major-asymmetric";
  ASSERT_GT(std::abs(dB_ref(0, 0, 1, 1) - dB_ref(1, 1, 0, 0)), 1e-3);
}

TEST(ResidualSystemTangent, AMemberReadOutsideGetsTheTotalDerivative) {
  strain_rig rig;
  auto const &drg{rig.point->derivative<tensor2>("sys::rg", "strain")};
  rig.evaluate(eps0);
  EXPECT_LT(tmech::norm(drg), 1e-12) << "R stays zero along the solution: dR/deps|x + dR/dx dx/deps = 0";
}

TEST(ResidualSystemTangent, AMissingPartialOnTheNewtonPathThrowsAtFinalize) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "load");
  ctx.create<nm::external_scalar_source<policy>>(p);
  make_system(ctx, {{"x", nm::unknown_kind::scalar}}, {{"f::r", "x"}});
  p.clear();
  p.insert<std::string>("name", "f");
  p.insert<bool>("declares_partials", false);
  ctx.create<cubic_row>(p);
  auto const message{thrown<std::logic_error>([&] { ctx.finalize(); })};
  EXPECT_TRUE(message.contains("'f::r'")) << message;
  EXPECT_TRUE(message.contains("solver::x")) << message;
}

TEST(ResidualSystemTangent, NewtonFdNeedsNoPartials) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "load");
  auto &load{ctx.create<nm::external_scalar_source<policy>>(p)};
  make_system(ctx, {{"x", nm::unknown_kind::scalar}}, {{"f::r", "x"}}, "newton_fd");
  p.clear();
  p.insert<std::string>("name", "f");
  p.insert<bool>("declares_partials", false);
  ctx.create<cubic_row>(p);
  ctx.finalize();
  load.bind(0, 2);
  ctx.update();
  EXPECT_NEAR(ctx.get<T>("solver", "x"), 1, 1e-10);
}

// --- picard ---

TEST(ResidualSystem, PicardWithUnknownsIteratesXMinusR) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "load");
  auto &load{ctx.create<nm::external_scalar_source<policy>>(p)};
  auto &picard{make_system(ctx, {{"x", nm::unknown_kind::scalar}}, {{"f::r", "x"}}, "picard", 200, 1e-12)};
  p.clear();
  p.insert<std::string>("name", "f");
  p.insert<bool>("picard", true);
  ctx.create<cubic_row>(p);
  ctx.finalize();
  load.bind(0, 0.3);
  ctx.update();
  T const x{ctx.get<T>("solver", "x")};
  EXPECT_NEAR(x, std::cos(0.3 + x) / 2, 1e-12);
  EXPECT_GT(picard.iterations(), 5) << "linear convergence: a contraction, not a Newton step";
}

// --- validation ---

TEST(ResidualSystemValidation, RejectsInconsistentUnknownsAndEquations) {
  using unknowns = std::vector<nm::system_unknown>;
  using equations = std::vector<nm::system_equation>;
  struct bad {
    unknowns u;
    equations e;
    std::string method;
    char const *says;
  };
  for (auto const &[u, e, method, says] : {
           bad{{{"x", nm::unknown_kind::scalar}, {"x", nm::unknown_kind::scalar}}, {{"a::r", "x"}, {"a::s", "x"}}, "newton", "'x'"},
           bad{{{"", nm::unknown_kind::scalar}}, {{"a::r", ""}}, "newton", "empty"},
           bad{{}, {}, "newton", "unknowns"},
           bad{xy(), {{"a::r", "x"}}, "newton", "'y'"},
           bad{xy(), {{"a::r", "x"}, {"a::s", "z"}}, "newton", "'z'"},
           bad{xy(), {{"a::r", "x"}, {"a::s", "x"}}, "newton", "'x'"},
           bad{xy(), {{"a::r", "x"}, {"a::r", "y"}}, "newton", "'a::r'"},
           bad{xy(), {{"a::r", "x"}, {"a_s", "y"}}, "newton", "a_s"},
           bad{xy(), {{"a::r", "x"}, {"a::s", "y"}}, "gauss_seidel", "gauss_seidel"},
           bad{{}, {{"a::r", "x"}}, "picard", "'x'"},
           bad{{{"B", nm::unknown_kind::sym_tensor, 1.0}}, {{"a::r", "B"}}, "newton", "seed"},
       }) {
    ctx_type ctx;
    auto const message{thrown<std::invalid_argument>([&] { make_system(ctx, u, e, method); })};
    EXPECT_TRUE(message.contains(says)) << says << ": " << message;
    EXPECT_TRUE(message.contains("'solver'")) << message;
  }
}

TEST(ResidualSystemValidation, AResidualOfAnotherKindFailsToWire) {
  ctx_type ctx;
  make_system(ctx, {{"B", nm::unknown_kind::sym_tensor}}, {{"load::state", "B"}});
  param_type p;
  p.insert<std::string>("name", "load");
  ctx.create<nm::external_scalar_source<policy>>(p);
  EXPECT_THROW(ctx.finalize(), std::runtime_error) << "a scalar residual for a tensor unknown fails to wire";
}

TEST(ResidualSystemValidation, AStatefulMemberThrowsAtFinalize) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "load");
  ctx.create<nm::external_scalar_source<policy>>(p);
  make_system(ctx, {{"x", nm::unknown_kind::scalar}}, {{"f::r", "x"}});
  p.clear();
  p.insert<std::string>("name", "f");
  p.insert<bool>("counts", true);
  ctx.create<cubic_row>(p);
  auto const message{thrown<std::logic_error>([&] { ctx.finalize(); })};
  EXPECT_TRUE(message.contains("f::r")) << message;
  EXPECT_TRUE(message.contains("stateful")) << message;
}

TEST(ResidualSystemValidation, SystemsAreDisjoint) {
  ctx_type ctx; // f::r reads the unknowns of both
  make_system(ctx, {{"x", nm::unknown_kind::scalar}}, {{"f::r", "x"}});
  make_system(ctx, {{"state", nm::unknown_kind::scalar}}, {{"f::r", "state"}}, "newton", 50, 1e-10, "load");
  param_type p;
  p.insert<std::string>("name", "f");
  ctx.create<cubic_row>(p);
  auto const message{thrown<std::logic_error>([&] { ctx.finalize(); })};
  EXPECT_TRUE(message.contains("'solver'") && message.contains("'load'")) << message;
}

TEST(ResidualSystemValidation, NoPathFromTheUnknownsToTheResidualsThrowsAtFinalize) {
  ctx_type ctx;
  make_system(ctx, {{"x", nm::unknown_kind::scalar}}, {{"load::state", "x"}});
  param_type p;
  p.insert<std::string>("name", "load");
  ctx.create<nm::external_scalar_source<policy>>(p);
  auto const message{thrown<std::logic_error>([&] { ctx.finalize(); })};
  EXPECT_TRUE(message.contains("'solver'")) << message;
}

// --- "seed": "history" ---

/// x of every step of `loads` at a host-owned point, its history in `buffer`, and the iterations each took.
struct history_rig {
  std::unique_ptr<ctx_type> owner{std::make_unique<ctx_type>()};
  ctx_type &ctx{*owner};
  system_type *system{nullptr};
  std::unique_ptr<nm::material_point<policy>> point;

  explicit history_rig(bool history) {
    param_type p;
    p.insert<std::string>("name", "load");
    ctx.create<nm::external_scalar_source<policy>>(p);
    system = &make_system(ctx, {{"x", nm::unknown_kind::scalar, 0, history}}, {{"f::r", "x"}}, "newton", 50, 1e-12);
    p.clear();
    p.insert<std::string>("name", "f");
    ctx.create<cubic_row>(p);
    ctx.finalize();
    param_type q;
    q.insert<std::vector<std::string>>("inputs", {"load"});
    point = std::make_unique<nm::material_point<policy>>(ctx, q);
  }

  std::pair<T, int> step(std::vector<T> &buffer, T load) {
    buffer.resize(point->history_width());
    point->reset();
    point->load_history(buffer);
    point->input<nm::external_scalar_source<policy>>("load").bind(0, load);
    point->evaluate();
    point->store_history(buffer);
    return {ctx.get<T>("solver", "x"), system->iterations()};
  }
};

TEST(ResidualSystemHistorySeed, TwoPointsAlternatelyTakeTheIterationsOfEachAlone) {
  std::vector<T> const loads_a{1, 2, 3, 3.5}, loads_b{-2, -1, 0.5, 4};
  auto const alone{[](std::vector<T> const &loads, bool history) {
    history_rig rig{history};
    std::vector<T> buffer;
    std::vector<std::pair<T, int>> out;
    for (auto const load : loads)
      out.push_back(rig.step(buffer, load));
    return out;
  }};
  auto const a{alone(loads_a, true)}, b{alone(loads_b, true)};

  history_rig shared{true};
  EXPECT_EQ(shared.point->history_width(), 1u) << "the unknown is kept as history";
  std::vector<T> buffer_a, buffer_b;
  for (std::size_t i{0}; i < loads_a.size(); ++i) {
    EXPECT_EQ(shared.step(buffer_a, loads_a[i]), a[i]) << "point a, step " << i;
    EXPECT_EQ(shared.step(buffer_b, loads_b[i]), b[i]) << "point b, step " << i;
  }

  int warm{0}, cold{0};
  for (auto const &[x, iterations] : alone(loads_a, true))
    warm += iterations;
  for (auto const &[x, iterations] : alone(loads_a, false))
    cold += iterations;
  EXPECT_LT(warm, cold) << "each step starts from the last converged x";
}

// --- review of the residual systems ---

/// A "solver" over the scalar x (seed `seed`) whose residual "f::r" is the row `Row`, and a "load" source.
template <typename Row>
struct row_rig {
  std::unique_ptr<ctx_type> owner{std::make_unique<ctx_type>()};
  ctx_type &ctx{*owner};
  system_type *system{nullptr};
  nm::external_scalar_source<policy> *load{nullptr};

  row_rig(std::string const &method, int max_iter, T tolerance, T seed, param_type row = {}) {
    param_type p;
    p.insert<std::string>("name", "load");
    load = &ctx.create<nm::external_scalar_source<policy>>(p);
    system = &make_system(ctx, {{"x", nm::unknown_kind::scalar, static_cast<double>(seed)}}, {{"f::r", "x"}}, method, max_iter, tolerance);
    row.insert<std::string>("name", "f");
    ctx.create<Row>(row);
    ctx.finalize();
  }

  T x() { return ctx.get<T>("solver", "x"); }
};

TEST(ResidualSystemReview, TheIterationsAreReportedWhenTheSolveThrows) {
  auto const ref{vector_newton_on(system_mode::nonlinear, 3, 5)};
  ASSERT_GT(ref.iterations, 1);
  scalar_rig rig{system_mode::nonlinear, 3, 5, ref.iterations - 1};
  EXPECT_THROW(rig.ctx.update(), nm::cutback_error);
  EXPECT_EQ(rig.system->iterations(), ref.iterations - 1) << "the updates taken before the throw";
}

TEST(ResidualSystemReview, ZeroBlocksBeforeFinalizeThrows) {
  ctx_type ctx;
  auto &system{make_system(ctx, xy(), {{"fx::rx", "x"}, {"fy::ry", "y"}})};
  auto const message{thrown<std::logic_error>([&] { (void)system.zero_blocks(); })};
  EXPECT_TRUE(message.contains("'solver'") && message.contains("finalize")) << message;
}

TEST(ResidualSystemReview, NewtonFdLeavesTheUnknownsAtTheIterateWhenTheJacobianIsSingular) {
  scalar_rig rig{system_mode::singular, 4, 7, 50, false, "newton_fd"};
  EXPECT_THROW(rig.ctx.update(), nm::cutback_error);
  EXPECT_EQ(rig.x(), 0) << "not x + h";
  EXPECT_EQ(rig.y(), 0);
}

TEST(ResidualSystemReview, ANonFiniteResidualNeverConverges) {
  T const inf{std::numeric_limits<T>::infinity()};
  for (T const value : {std::numeric_limits<T>::quiet_NaN(), inf, -inf})
    for (T const tolerance : {T{1e-10}, inf})
      for (std::string const method : {"picard", "newton_fd"}) {
        param_type row;
        row.insert<T>("value", value);
        row_rig<poison_row> rig{method, 5, tolerance, 0, row};
        auto const message{thrown<nm::cutback_error>([&] { rig.ctx.update(); })};
        EXPECT_TRUE(message.contains("'solver'")) << value << ", tolerance " << tolerance << ", " << method << ": " << message;
      }
}

TEST(ResidualSystemReview, AConvergedSeedNeedsNoUpdateWithToleranceAndMaxIterZero) {
  for (std::string const method : {"newton", "newton_fd", "picard"}) {
    row_rig<square_row> rig{method, 0, 0, 1};
    rig.load->bind(1, 1); // x = 1 solves x^2 = 1 exactly
    EXPECT_NO_THROW(rig.ctx.update()) << method;
    EXPECT_EQ(rig.system->iterations(), 0) << method;
    EXPECT_EQ(rig.x(), 1);
    row_rig<square_row> off{method, 0, 0, 1};
    off.load->bind(1, 2);
    EXPECT_THROW(off.ctx.update(), nm::cutback_error) << method << ": not converged, and no update allowed";
  }
}

TEST(ResidualSystemReview, NewtonFdConvergesAsFastAsNewtonOnSmallUnknowns) {
  for (T const scale : {T{1e-9}, T{1}, T{1e6}}) {
    param_type row;
    row.insert<T>("scale", scale);
    row_rig<scaled_row> newton{"newton", 50, 1e-10, 0.5 * scale, row};
    newton.ctx.update();
    EXPECT_NEAR(newton.x() / scale, 1, 1e-9);
    row_rig<scaled_row> fd{"newton_fd", 50, 1e-10, 0.5 * scale, row};
    EXPECT_NO_THROW(fd.ctx.update()) << "scale " << scale;
    EXPECT_NEAR(fd.x() / scale, 1, 1e-9) << "scale " << scale;
    EXPECT_LE(fd.system->iterations(), newton.system->iterations() + 1) << "scale " << scale;
  }
}

TEST(ResidualSystemTangent, NewtonFdAndPicardGiveTheImplicitFunctionTangent) {
  row_rig<cubic_row> fd{"newton_fd", 50, 1e-12, 0};
  auto const &dx_fd{fd.ctx.derivative<T>("solver", "x", "load", "state")};
  fd.load->bind(0, 2);
  fd.ctx.update();
  EXPECT_NEAR(fd.x(), 1, 1e-10);
  EXPECT_NEAR(dx_fd, 1 / (3 * fd.x() * fd.x() + 1), 1e-10);

  param_type row;
  row.insert<bool>("picard", true);
  row_rig<cubic_row> picard{"picard", 500, 1e-13, 0, row};
  auto const &dx{picard.ctx.derivative<T>("solver", "x", "load", "state")};
  auto const at{[&](T load) {
    picard.load->bind(0, load);
    picard.ctx.update();
    return picard.x();
  }};
  T const load{0.3}, h{1e-5};
  T const plus{at(load + h)}, minus{at(load - h)};
  T const x{at(load)};
  T const s{std::sin(load + x)};
  EXPECT_NEAR(dx, -(s / 2) / (1 + s / 2), 1e-10);
  EXPECT_NEAR(dx, (plus - minus) / (2 * h), 1e-7);
}

TEST(ResidualSystemTangent, TwoCoupledSystemsMatchCentralDifferences) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "load");
  auto &load{ctx.create<nm::external_scalar_source<policy>>(p)};
  make_system(ctx, {{"x", nm::unknown_kind::scalar}}, {{"f::r", "x"}}, "newton", 50, 1e-13);
  make_system(ctx, {{"y", nm::unknown_kind::scalar}}, {{"g::r", "y"}}, "newton", 50, 1e-13, "second");
  p.clear();
  p.insert<std::string>("name", "f");
  ctx.create<cubic_row>(p);
  p.clear();
  p.insert<std::string>("name", "g");
  ctx.create<chain_row>(p);
  ctx.finalize();
  auto const &dx{ctx.derivative<T>("solver", "x", "load", "state")};
  auto const &dy{ctx.derivative<T>("second", "y", "load", "state")};
  auto const at{[&](T value) {
    load.bind(0, value);
    ctx.update();
    return std::pair{ctx.get<T>("solver", "x"), ctx.get<T>("second", "y")};
  }};
  T const value{3}, h{1e-5};
  auto const [xp, yp]{at(value + h)};
  auto const [xm, ym]{at(value - h)};
  auto const [x, y]{at(value)};
  EXPECT_NEAR(dx, (xp - xm) / (2 * h), 1e-7);
  EXPECT_NEAR(dy, (yp - ym) / (2 * h), 1e-7) << "the second system reads the unknown of the first";
  EXPECT_NEAR(dy, 1 / (3 * y * y + 1) / (3 * x * x + 1), 1e-10);
}

TEST(ResidualSystemTangent, ASingularJacobianAtTheConvergedPointThrowsACutbackNamingTheSystem) {
  row_rig<square_row> rig{"newton", 50, 1e-10, 0};
  rig.ctx.derivative<T>("solver", "x", "load", "state");
  rig.load->bind(0, 0); // x = 0 is a double root, and the seed already solves it
  auto const message{thrown<nm::cutback_error>([&] { rig.ctx.update(); })};
  EXPECT_TRUE(message.contains("'solver'")) << message;
}

TEST(ResidualSystemTangent, ASeedThatIsAMemberOutputOfAPassedSystemThrows) {
  strain_rig rig;
  auto const message{thrown<std::logic_error>([&] { rig.ctx.derivative<T>("solver", "g", "sys", "rg"); })};
  EXPECT_TRUE(message.contains("sys::rg") && message.contains("'solver'")) << message;
}

#ifdef NUMSIM_HAVE_JSON
TEST(ResidualSystemJson, ADocumentBuildsTheSystem) {
  nm::register_default_materials<policy>();
  nm::material_factory<policy>::instance().template register_type<scalar_rows>("scalar_rows");
  auto const doc = nlohmann::json::parse(R"({"materials": [
    {"type": "system", "name": "solver", "method": "newton", "tolerance": 1e-12, "max_iter": 25,
     "unknowns": [{"name": "x", "kind": "scalar", "seed": 0}, {"name": "y", "kind": "scalar", "seed": "history"}],
     "equations": [{"residual": "sys::rx", "for": "x"}, {"residual": "sys::ry", "for": "y"}]},
    {"type": "scalar_rows", "name": "sys", "mode": 0, "a": 5.0, "b": 10.0}]})");
  ctx_type ctx;
  for (auto const &m : doc["materials"])
    nm::create_from_json(ctx, m);
  ctx.finalize();
  ctx.update();
  EXPECT_NEAR(ctx.get<T>("solver", "x"), 1, 1e-12);
  EXPECT_NEAR(ctx.get<T>("solver", "y"), 3, 1e-12);
  EXPECT_TRUE(ctx.find_property("solver", "y")->is_history());
  EXPECT_FALSE(ctx.find_property("solver", "x")->is_history());
}

TEST(ResidualSystemJson, RejectsAnUnknownSeedOrKind) {
  nm::register_default_materials<policy>();
  for (auto const *unknown : {R"({"name": "x", "kind": "scalar", "seed": "warm"})", R"({"name": "x", "kind": "vector"})"}) {
    auto const doc = nlohmann::json::parse(std::string{R"({"type": "system", "name": "solver", "unknowns": [)"} + unknown +
                                         R"(], "equations": [{"residual": "sys::rx", "for": "x"}]})");
    ctx_type ctx;
    auto const message{thrown<std::exception>([&] { nm::create_from_json(ctx, doc); })};
    EXPECT_TRUE(message.contains("unknowns") && message.contains("'x'")) << unknown << ": " << message;
  }
}

TEST(ResidualSystemJson, TheReadersRequireListsAndRejectUnknownKeys) {
  nm::register_default_materials<policy>();
  struct bad {
    char const *unknowns, *equations, *says;
  };
  for (auto const &[unknowns, equations, says] : {
           bad{R"({"name": "x", "kind": "scalar"})", R"([{"residual": "sys::rx", "for": "x"}])", "unknowns"},
           bad{R"(["x"])", R"([{"residual": "sys::rx", "for": "x"}])", "unknowns"},
           bad{R"([{"name": "x", "kind": "scalar", "seeed": 1}])", R"([{"residual": "sys::rx", "for": "x"}])", "seeed"},
           bad{R"([{"name": "x", "kind": "scalar"}])", R"({"residual": "sys::rx", "for": "x"})", "equations"},
           bad{R"([{"name": "x", "kind": "scalar"}])", R"([{"residual": "sys::rx", "forr": "x"}])", "forr"},
       }) {
    auto const doc = nlohmann::json::parse(std::string{R"({"type": "system", "name": "solver", "unknowns": )"} + unknowns +
                                           R"(, "equations": )" + equations + "}");
    ctx_type ctx;
    auto const message{thrown<std::exception>([&] { nm::create_from_json(ctx, doc); })};
    EXPECT_TRUE(message.contains(says)) << unknowns << " / " << equations << ": " << message;
  }
}
#endif

} // namespace
