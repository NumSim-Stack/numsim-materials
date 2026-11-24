// update() through a residual system with a requested derivative allocates nothing (#218): a host calls it per
// material point and increment, from every thread. Counts operator new and Eigen's heap allocations.
#include <cstddef>

namespace numsim_test {
std::size_t new_calls();
inline std::size_t eigen_allocations{0};
} // namespace numsim_test

#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert(x)                                                                                                \
  do {                                                                                                                 \
    if (!(x))                                                                                                          \
      ++numsim_test::eigen_allocations;                                                                                \
  } while (false)

#include <cmath>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/plasticity_utils.h"
#include "numsim-materials/solvers/system.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using t2 = tmech::tensor<T, 3, 2>;
using t4 = tmech::tensor<T, 3, 4>;

constexpr T K{166.67}, G{76.92}, sigma_0{50.0}, H{1000.0};

/// J2 yield residual in dlambda with linear hardening, and the plastic strain it implies.
class j2_flow final : public nm::material_base<j2_flow, policy> {
public:
  using base = nm::material_base<j2_flow, policy>;

  template <typename... Args>
  explicit j2_flow(Args &&...args)
      : base(std::forward<Args>(args)...), m_r(base::add_output<T>("residual")),
        m_eps_p(base::add_history_output<t2>("plastic_strain")), m_kappa(base::add_history_output<T>("kappa")),
        m_strain(base::add_input<t2>("stepper", "strain")), m_dl(base::add_input<T>("plastic", "dlambda")) {
    base::add_callback(&j2_flow::compute, m_r, m_eps_p, m_kappa)
        .reads(m_strain, m_dl)
        .partials(m_r, m_strain, m_dl)
        .partials(m_eps_p, m_strain, m_dl);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    T const dl{m_dl.get()};
    t2 const s{T(2) * G * tmech::dev(t2{m_strain.get() - m_eps_p.old_value()})};
    T const q{std::sqrt(T(1.5) * tmech::dcontract(s, s))};
    bool const plastic{q - sigma_0 - H * m_kappa.old_value() > T(0)};
    t2 const N{plastic ? t2{T(1.5) * s / q} : t2{}};
    m_r = plastic ? q - T(3) * G * dl - sigma_0 - H * (m_kappa.old_value() + dl) : dl;
    m_eps_p.new_value() = t2{m_eps_p.old_value() + dl * N};
    m_kappa.new_value() = m_kappa.old_value() + dl;
    if (!base::want_partials())
      return;
    t4 const A{plastic ? t4{(T(1.5) * nm::dev_projector<T, 3>() - tmech::otimes(N, N)) / q} : t4{}};
    base::partial(m_r, m_strain) = plastic ? t2{T(2) * G * N} : t2{};
    base::partial(m_r, m_dl) = plastic ? -T(3) * G - H : T(1);
    base::partial(m_eps_p, m_strain) = t4{(T(2) * G * dl) * A};
    base::partial(m_eps_p, m_dl) = N;
  }

private:
  T &m_r;
  nm::history_property<t2> &m_eps_p;
  nm::history_property<T> &m_kappa;
  nm::input_property<t2, nm::property_traits> const &m_strain;
  nm::input_property<T, nm::property_traits> const &m_dl;
};

/// sigma = C_e : (eps - plastic strain).
class j2_stress final : public nm::material_base<j2_stress, policy> {
public:
  using base = nm::material_base<j2_stress, policy>;

  template <typename... Args>
  explicit j2_stress(Args &&...args)
      : base(std::forward<Args>(args)...), m_stress(base::add_output<t2>("stress")),
        m_strain(base::add_input<t2>("stepper", "strain")), m_eps_p(base::add_input<t2>("flow", "plastic_strain")),
        m_C(nm::plasticity_detail::make_isotropic_tangent<T, 3>(K, G)) {
    base::add_callback(&j2_stress::compute, m_stress).reads(m_strain, m_eps_p).partials(m_stress, m_strain, m_eps_p);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    m_stress = tmech::dcontract(m_C, t2{m_strain.get() - m_eps_p.get()});
    if (base::want_partials()) {
      base::partial(m_stress, m_strain) = m_C;
      base::partial(m_stress, m_eps_p) = t4{-m_C};
    }
  }

private:
  t2 &m_stress;
  nm::input_property<t2, nm::property_traits> const &m_strain;
  nm::input_property<t2, nm::property_traits> const &m_eps_p;
  t4 const m_C;
};

struct j2_system {
  ctx_type ctx;
  nm::external_strain_source<policy> *src{};

  j2_system() {
    param_type p;
    p.insert<std::string>("name", "stepper");
    src = &ctx.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "plastic");
    p.insert<T>("tolerance", T{1e-10});
    p.insert<std::vector<nm::system_unknown>>("unknowns", {{"dlambda", nm::unknown_kind::scalar}});
    p.insert<std::vector<nm::system_equation>>("equations", {{"flow::residual", "dlambda"}});
    ctx.create<nm::residual_system<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "flow");
    ctx.create<j2_flow>(p);
    p.clear();
    p.insert<std::string>("name", "j2");
    ctx.create<j2_stress>(p);
    ctx.finalize();
    (void)ctx.derivative<t4>("j2", "stress", "stepper", "strain");
  }

  /// Heap allocations of one update() at `eps`, after a first update() there.
  std::size_t allocations(t2 const &eps) {
    src->bind(t2{}, eps);
    ctx.update();
    auto const new_before{numsim_test::new_calls()};
    numsim_test::eigen_allocations = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    ctx.update();
    Eigen::internal::set_is_malloc_allowed(true);
    return numsim_test::new_calls() - new_before + numsim_test::eigen_allocations;
  }
};

t2 shear(T gamma) {
  t2 e;
  e(0, 1) = e(1, 0) = gamma;
  return e;
}

} // namespace

TEST(SystemAllocations, AnElasticUpdateWithADerivativeAllocatesNothing) {
  j2_system sys;
  EXPECT_EQ(sys.allocations(shear(0.1)), 0u);
}

TEST(SystemAllocations, APlasticUpdateWithADerivativeAllocatesNothing) {
  j2_system sys;
  EXPECT_EQ(sys.allocations(shear(0.4)), 0u);
}
