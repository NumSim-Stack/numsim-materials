#include <gtest/gtest.h>
#include <cstddef>
#include <span>
#include <string>
#include <thread>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/input_types.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/umat_interface.h"

// The Fortran-callable symbol. Exactly one TU may define it.
NUMSIM_MATERIALS_DEFINE_UMAT(numsim::materials::material_policy_default)

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using registry = u::umat_registry<policy>;

constexpr T K = 166.67;
constexpr T G = 76.92;
constexpr T sigma_0 = 50.0;
constexpr T H_mod = 1000.0;

void build_j2(ctx_type& ctx, std::span<const double> /*props*/) {
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  ctx.create<nm::external_strain_source<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<nm::backward_euler<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<T>("K", H_mod);
  ctx.create<nm::linear_isotropic_hardening<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("elastic_source", "elastic");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T>("G", G);
  p.insert<T>("sigma_0", sigma_0);
  ctx.create<nm::j2_plasticity<policy>>(p);

  ctx.finalize();
}

/// Reports the time it was bound with, so TIME/DTIME plumbing is observable
/// from outside. Without something like this, nothing that reaches a material
/// through the shim depends on time at all, and TIME(1)-vs-TIME(2) or a dropped
/// DTIME is undetectable at the ABI.
template <typename Traits>
class time_probe_material final
    : public nm::material_base<time_probe_material<Traits>, Traits> {
public:
  using base = nm::material_base<time_probe_material<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  template <typename... Args>
  explicit time_probe_material(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>(
            "stress", &time_probe_material::compute)),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_time(base::template add_input_history<value_type>(
            nm::connection_source{"clock", "state"}, nm::EdgeKind::Global)) {}

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() {
    m_stress.fill(0.0);
    m_stress(0, 0) = m_time.new_value();
    m_stress(1, 1) = m_time.new_value() - m_time.old_value();
    m_stress(2, 2) = m_time.old_value();
  }

private:
  tensor2& m_stress;
  tensor4& m_tangent;
  const nm::input_history<value_type, nm::property_traits>& m_time;
};

void build_time_probe(ctx_type& ctx, std::span<const double> /*props*/) {
  param_type p;
  p.insert<std::string>("name", "stepper");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "clock");
  ctx.create<nm::external_scalar_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "probe");
  ctx.create<time_probe_material<policy>>(p);
  ctx.finalize();
}

/// Constants come entirely from *USER MATERIAL, CONSTANTS= — nothing is baked
/// into the builder. The slot mapping is this builder's contract with the deck:
///
///   props[0] = K   props[1] = G
void build_deck_elastic(ctx_type& ctx, std::span<const double> props) {
  u::require_props(props, 2, "deck_elastic");
  param_type p;
  p.insert<std::string>("name", "stepper");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", props[0]);
  p.insert<T>("G", props[1]);
  ctx.create<nm::linear_elasticity<policy>>(p);
  ctx.finalize();
}

/// Register once for the whole binary, as a real UMAT library would from a
/// static initialiser.
struct Registration {
  Registration() {
    registry::config cfg;
    cfg.strain_source = "stepper";
    cfg.stress_source = "j2";
    registry::instance().register_model("J2STEEL", build_j2, cfg);

    registry::config tp;
    tp.strain_source = "stepper";
    tp.stress_source = "probe";
    tp.time_source = "clock";
    registry::instance().register_model("TIMEPROBE", build_time_probe, tp);

    // Two deck materials, ONE builder: the constants differ only via PROPS.
    // Before PROPS was plumbed, this needed two builders with captured values.
    registry::config de;
    de.strain_source = "stepper";
    de.stress_source = "elastic";
    registry::instance().register_model("SOFT",  build_deck_elastic, de);
    registry::instance().register_model("STIFF", build_deck_elastic, de);
    // Used only by the too-few-constants test. It needs a name no other test
    // has warmed: the per-thread cache persists for the life of the binary, so
    // against an already-built name the NPROPS-consistency check fires first
    // and require_props is never reached.
    registry::instance().register_model("COLDNAME", build_deck_elastic, de);

    // Deliberately lower-case, to prove the registry folds case on both sides.
    registry::config lc;
    lc.strain_source = "stepper";
    lc.stress_source = "j2";
    registry::instance().register_model("mixedCase", build_j2, lc);
  }
};
const Registration registration_{};

/// CMNAME as Fortran passes it: character*80, blank padded, no NUL.
struct fortran_name {
  char buf[80];
  explicit fortran_name(const std::string& s) {
    for (std::size_t i = 0; i < 80; ++i) buf[i] = ' ';
    for (std::size_t i = 0; i < s.size() && i < 80; ++i) buf[i] = s[i];
  }
};

/// Call umat_ with the full Abaqus argument list.
void call_umat(const std::string& name, T* stress, T* statev, T* ddsdde,
               const T* stran, const T* dstran, T total_time, T dtime, int ndi,
               int nshr, int ntens, int nstatv, T* pnewdt,
               const T* drot_in = nullptr, T* sse_io = nullptr,
               T* spd_io = nullptr, T* scd_io = nullptr,
               const T* props_in = nullptr, int nprops_in = 0) {
  const fortran_name cm(name);
  T sse_local = 0, spd_local = 0, scd_local = 0;
  T* sse = sse_io ? sse_io : &sse_local;
  T* spd = spd_io ? spd_io : &spd_local;
  T* scd = scd_io ? scd_io : &scd_local;
  T rpl = 0;
  std::vector<T> ddsddt(ntens, 0.0), drplde(ntens, 0.0);
  T drpldt = 0;
  const T time[2] = {total_time, total_time};
  const T temp = 0, dtemp = 0, predef = 0, dpred = 0;
  const T no_props = 0;
  const T* props = props_in ? props_in : &no_props;
  const int nprops = nprops_in;
  const T coords[3] = {0, 0, 0};
  const T identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const T* drot = drot_in ? drot_in : identity;
  const T celent = 1.0;
  const T dfgrd0[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const T dfgrd1[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const int noel = 1, npt = 1, layer = 1, kspt = 1, jstep = 1, kinc = 1;

  umat_(stress, statev, ddsdde, sse, spd, scd, &rpl, ddsddt.data(),
        drplde.data(), &drpldt, stran, dstran, time, &dtime, &temp, &dtemp,
        &predef, &dpred, cm.buf, &ndi, &nshr, &ntens, &nstatv, props, &nprops,
        coords, drot, pnewdt, &celent, dfgrd0, dfgrd1, &noel, &npt, &layer,
        &kspt, &jstep, &kinc, 80);
}

// ---------------------------------------------------------------------------
// ABI plumbing
// ---------------------------------------------------------------------------

TEST(UmatInterface, ElementCaseFromNdiNshr) {
  EXPECT_EQ(u::case_from_element(3, 3), u::element_case::solid3d);
  // Plane strain and axisymmetric share NDI/NSHR and share the code path.
  EXPECT_EQ(u::case_from_element(3, 1), u::element_case::plane_strain);
  EXPECT_EQ(u::case_from_element(2, 1), u::element_case::plane_stress);
  EXPECT_THROW(u::case_from_element(1, 0), u::fatal_error);
}

TEST(UmatInterface, TrimsBlankPaddedFortranName) {
  const fortran_name cm("J2STEEL");
  EXPECT_EQ(u::trim_fortran_name(cm.buf, 80), "J2STEEL");
  const fortran_name empty("");
  EXPECT_EQ(u::trim_fortran_name(empty.buf, 80), "");
}

TEST(UmatInterface, ReportsStatevWidth) {
  EXPECT_EQ(registry::instance().nstatv("J2STEEL"), 7u);
  EXPECT_EQ(registry::instance().nstatv("J2STEEL", {},
                                       u::element_case::plane_stress),
            8u);
}

// ---------------------------------------------------------------------------
// Driving through the C ABI
// ---------------------------------------------------------------------------

/// The same trajectory as the direct-C++ evaluator, but reached through umat_.
TEST(UmatInterface, DrivesJ2ThroughTheFortranEntryPoint) {
  std::vector<T> statev(7, 0.0);
  T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.01, -0.0025, 0.0, 0.005, 0.0, 0.0};

  T stress[6], ddsdde[36], pnewdt = 1.0;
  for (int step = 0; step < 40; ++step) {
    call_umat("J2STEEL", stress, statev.data(), ddsdde, stran, dstran,
              0.1 * step, 0.1, 3, 3, 6, static_cast<int>(statev.size()),
              &pnewdt);
    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }

  EXPECT_DOUBLE_EQ(pnewdt, 1.0) << "no cutback should have been requested";
  EXPECT_GT(statev[0], 1e-8) << "the path must yield";

  // Cross-check against the evaluator driven directly in C++.
  ctx_type ctx;
  build_j2(ctx, {});
  u::material_point_evaluator<policy>::config cfg;
  cfg.strain_source = "stepper";
  cfg.stress_source = "j2";
  u::material_point_evaluator<policy> eval(ctx, cfg);

  std::vector<T> ref_statev(eval.nstatv(), 0.0);
  T ref_stran[6] = {0, 0, 0, 0, 0, 0};
  T ref_stress[6], ref_ddsdde[36];
  for (int step = 0; step < 40; ++step) {
    eval.evaluate({.stran = ref_stran, .dstran = dstran, .stress = ref_stress,
                   .ddsdde = ref_ddsdde, .statev = ref_statev});
    for (std::size_t i = 0; i < 6; ++i) ref_stran[i] += dstran[i];
  }

  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_DOUBLE_EQ(stress[i], ref_stress[i]) << "stress slot " << i;
  for (std::size_t i = 0; i < 36; ++i)
    EXPECT_DOUBLE_EQ(ddsdde[i], ref_ddsdde[i]) << "ddsdde slot " << i;
  for (std::size_t i = 0; i < statev.size(); ++i)
    EXPECT_DOUBLE_EQ(statev[i], ref_statev[i]) << "statev slot " << i;
}

TEST(UmatInterface, PlaneStressThroughTheFortranEntryPoint) {
  std::vector<T> statev(8, 0.0);  // 7 + the out-of-plane slot
  T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.01, -0.002, 0.004};

  T stress[3], ddsdde[9], pnewdt = 1.0;
  for (int step = 0; step < 30; ++step) {
    call_umat("J2STEEL", stress, statev.data(), ddsdde, stran, dstran,
              0.1 * step, 0.1, 2, 1, 3, static_cast<int>(statev.size()),
              &pnewdt);
    for (std::size_t i = 0; i < 3; ++i) stran[i] += dstran[i];
  }

  EXPECT_DOUBLE_EQ(pnewdt, 1.0);
  EXPECT_GT(statev[0], 1e-8) << "the path must yield";
}

// ---------------------------------------------------------------------------
// Error handling at the ABI boundary
// ---------------------------------------------------------------------------

/// A fatal fault must terminate the analysis. The handler is replaced so the
/// test can observe it instead of the test runner being killed by XIT/abort.
struct FatalProbe {
  static inline std::string last;
  static inline int count = 0;
  static void handler(const char* msg) {
    last = msg;
    ++count;
  }
  FatalProbe() {
    last.clear();
    count = 0;
    u::set_fatal_handler(&handler);
  }
  ~FatalProbe() { u::set_fatal_handler(nullptr); }
};

/// An unknown CMNAME is unrecoverable. Previously this returned normally with
/// PNEWDT untouched, leaving DDSDDE at whatever the caller's buffer happened to
/// contain — Abaqus would then solve on with a garbage tangent. It must now
/// terminate, and the outputs must be zeroed so that even a handler which
/// wrongly returns cannot pass uninitialised memory off as a tangent.
TEST(UmatInterface, UnknownMaterialIsFatal) {
  FatalProbe probe;
  std::vector<T> statev(7, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};

  // Poison the outputs: a correct implementation must not leave these behind.
  T stress[6], ddsdde[36], pnewdt = 1.0;
  for (auto& v : stress) v = -999.0;
  for (auto& v : ddsdde) v = -999.0;

  call_umat("NOSUCHMATERIAL", stress, statev.data(), ddsdde, stran, dstran, 0.0,
            0.1, 3, 3, 6, 7, &pnewdt);

  EXPECT_EQ(FatalProbe::count, 1) << "the analysis must be terminated";
  EXPECT_NE(FatalProbe::last.find("NOSUCHMATERIAL"), std::string::npos)
      << FatalProbe::last;
  EXPECT_DOUBLE_EQ(pnewdt, 1.0) << "a cutback cannot fix a setup fault";
  for (std::size_t i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(stress[i], 0.0) << i;
  for (std::size_t i = 0; i < 36; ++i) EXPECT_DOUBLE_EQ(ddsdde[i], 0.0) << i;
}

/// An undersized *DEPVAR is equally unfixable by a cutback. Previously it set
/// PNEWDT = 0.25, so Abaqus halved the increment until it died on the minimum
/// time step with the real cause buried far above.
TEST(UmatInterface, UndersizedStatevIsFatalNotACutback) {
  FatalProbe probe;
  std::vector<T> statev(3, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;

  call_umat("J2STEEL", stress, statev.data(), ddsdde, stran, dstran, 0.0, 0.1,
            3, 3, 6, 3, &pnewdt);

  EXPECT_EQ(FatalProbe::count, 1);
  EXPECT_NE(FatalProbe::last.find("NSTATV"), std::string::npos)
      << FatalProbe::last;
  EXPECT_DOUBLE_EQ(pnewdt, 1.0) << "no cutback for a setup fault";
}

/// A misconfigured model — here a stress_source naming a material that does not
/// exist — is a setup fault too, even though it surfaces from a layer below the
/// ABI shim. This is the case that proved the original classification was
/// decorative: it reached the generic handler and requested a cutback.
TEST(UmatInterface, MisconfiguredModelIsFatalNotACutback) {
  FatalProbe probe;
  {
    registry::config bad;
    bad.strain_source = "stepper";
    bad.stress_source = "typo";
    registry::instance().register_model("BADMODEL", build_j2, bad);
  }

  std::vector<T> statev(7, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;

  call_umat("BADMODEL", stress, statev.data(), ddsdde, stran, dstran, 0.0, 0.1,
            3, 3, 6, 7, &pnewdt);

  EXPECT_EQ(FatalProbe::count, 1);
  EXPECT_NE(FatalProbe::last.find("typo"), std::string::npos)
      << FatalProbe::last;
  EXPECT_DOUBLE_EQ(pnewdt, 1.0);
}

/// A genuine numerical failure keeps the cutback behaviour: it IS retry-worthy.
TEST(UmatInterface, PlaneStressNonConvergenceRequestsCutback) {
  FatalProbe probe;
  {
    registry::config cfg;
    cfg.strain_source = "stepper";
    cfg.stress_source = "j2";
    u::plane_stress_evaluator<policy>::options opts;
    opts.max_iter = 1;  // too few to converge
    registry::instance().register_model("TIGHTPS", build_j2, cfg, opts);
  }

  std::vector<T> statev(8, 0.0);
  const T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.02, -0.005, 0.01};
  T stress[3] = {0}, ddsdde[9] = {0}, pnewdt = 1.0;

  testing::internal::CaptureStderr();
  call_umat("TIGHTPS", stress, statev.data(), ddsdde, stran, dstran, 0.0, 0.1,
            2, 1, 3, 8, &pnewdt);
  const std::string err = testing::internal::GetCapturedStderr();

  EXPECT_EQ(FatalProbe::count, 0) << "a convergence failure is not fatal";
  EXPECT_DOUBLE_EQ(pnewdt, 0.25) << "it should ask for a smaller increment";
  EXPECT_NE(err.find("smaller increment"), std::string::npos) << err;
}

/// No C++ exception may unwind into Fortran, whatever goes wrong.
TEST(UmatInterface, NeverPropagatesExceptions) {
  FatalProbe probe;
  std::vector<T> statev(7, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;

  EXPECT_NO_THROW(call_umat("NOSUCHMATERIAL", stress, statev.data(), ddsdde,
                            stran, dstran, 0.0, 0.1, 3, 3, 6, 7, &pnewdt));
  EXPECT_NO_THROW(call_umat("J2STEEL", stress, statev.data(), ddsdde, stran,
                            dstran, 0.0, 0.1, 9, 9, 6, 7, &pnewdt));
}

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------

/// Abaqus calls UMAT from several threads at once. Each builds its own context
/// on first use; every thread must reproduce the single-threaded answer.
TEST(UmatInterface, ConcurrentCallsAgreeWithSingleThreadedResult) {
  const T dstran[6] = {0.01, -0.0025, 0.0, 0.005, 0.0, 0.0};

  auto drive = [&dstran](int steps) {
    std::vector<T> statev(7, 0.0);
    T stran[6] = {0, 0, 0, 0, 0, 0};
    T stress[6], ddsdde[36], pnewdt = 1.0;
    for (int step = 0; step < steps; ++step) {
      call_umat("J2STEEL", stress, statev.data(), ddsdde, stran, dstran,
                0.1 * step, 0.1, 3, 3, 6, static_cast<int>(statev.size()),
                &pnewdt);
      for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
    }
    return statev;
  };

  const std::vector<T> expected = drive(40);
  ASSERT_GT(expected[0], 1e-8);

  constexpr int n_threads = 8;
  std::vector<std::vector<T>> results(n_threads);
  std::vector<std::thread> threads;
  threads.reserve(n_threads);

  for (int t = 0; t < n_threads; ++t)
    threads.emplace_back([&results, &drive, t] { results[t] = drive(40); });
  for (auto& th : threads) th.join();

  for (int t = 0; t < n_threads; ++t) {
    ASSERT_EQ(results[t].size(), expected.size()) << "thread " << t;
    for (std::size_t i = 0; i < expected.size(); ++i)
      EXPECT_DOUBLE_EQ(results[t][i], expected[i])
          << "thread " << t << " statev " << i;
  }
}


// ---------------------------------------------------------------------------
// Arguments the shim forwards
// ---------------------------------------------------------------------------

/// DROT must reach the material through the Fortran entry point, not just the
/// C++ evaluator: the macro is where an argument is easiest to drop silently.
TEST(UmatInterface, ForwardsDrotThroughTheEntryPoint) {
  // 90 degrees about z, column-major.
  T drot[9];
  for (auto& v : drot) v = 0.0;
  drot[0 + 3 * 1] = -1.0;
  drot[1 + 3 * 0] = 1.0;
  drot[2 + 3 * 2] = 1.0;

  const T dstran[6] = {0.01, -0.0025, 0.0, 0.005, 0.0, 0.0};
  auto drive = [&dstran](const T* rot_last_step) {
    std::vector<T> statev(7, 0.0);
    T stran[6] = {0, 0, 0, 0, 0, 0};
    T stress[6], ddsdde[36], pnewdt = 1.0;
    for (int step = 0; step < 41; ++step) {
      const bool last = (step == 40);
      call_umat("J2STEEL", stress, statev.data(), ddsdde, stran, dstran,
                0.1 * step, 0.1, 3, 3, 6, static_cast<int>(statev.size()),
                &pnewdt, last ? rot_last_step : nullptr);
      for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
    }
    return statev;
  };

  const auto without = drive(nullptr);
  const auto with = drive(drot);
  ASSERT_GT(without[0], 1e-8) << "the path must be plastic to be directional";

  // Note this is NOT an objectivity check — the stored state is rotated but the
  // strain increment is not, so the physical problem genuinely differs and even
  // the scalar equivalent plastic strain is expected to move. Objectivity is
  // covered at the evaluator level, where the strains are rotated too. All this
  // test establishes is that DROT reaches the material at all, which is exactly
  // the property the macro could silently drop.
  T diff = 0;
  for (std::size_t i = 0; i < without.size(); ++i)
    diff = std::max(diff, std::abs(with[i] - without[i]));
  EXPECT_GT(diff, 1e-4) << "DROT was not forwarded: the state is unchanged";
}

/// SSE and SPD are in/out arguments; the shim must pass the host's storage
/// through so the values accumulate across increments.
TEST(UmatInterface, ForwardsEnergiesThroughTheEntryPoint) {
  registry::config cfg;
  cfg.strain_source = "stepper";
  cfg.stress_source = "j2";
  cfg.plastic_strain_property = "j2::plastic_strain";
  registry::instance().register_model("J2ENERGY", build_j2, cfg);

  std::vector<T> statev(7, 0.0);
  T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.01, -0.0025, 0.0, 0.005, 0.0, 0.0};
  T stress[6] = {0, 0, 0, 0, 0, 0}, ddsdde[36], pnewdt = 1.0;
  T sse = 0, spd = 0, scd = 5.0;  // SCD deliberately nonzero on entry

  for (int step = 0; step < 40; ++step) {
    call_umat("J2ENERGY", stress, statev.data(), ddsdde, stran, dstran,
              0.1 * step, 0.1, 3, 3, 6, static_cast<int>(statev.size()),
              &pnewdt, nullptr, &sse, &spd, &scd);
    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }

  EXPECT_GT(sse, 0.0) << "elastic energy should be reported";
  EXPECT_GT(spd, 0.0) << "plastic dissipation should accumulate";
  // Not a physical claim about creep: SCD is documented as never written by
  // this layer, so it must come back exactly as it was passed in. Asserting
  // against a NONZERO input makes that a real check rather than one satisfied
  // by a do-nothing implementation and a zero-initialised variable.
  EXPECT_DOUBLE_EQ(scd, 5.0) << "SCD is never written; it must pass through";
}


// ---------------------------------------------------------------------------
// TIME plumbing
// ---------------------------------------------------------------------------

/// TIME(2) is the TOTAL time at the start of the increment; TIME(1) is the step
/// time. They differ in any analysis past the first step, and a material
/// forming dt from the old/new pair depends on both the right slot and DTIME
/// actually arriving.
TEST(UmatInterface, PassesTotalTimeAndIncrementToTheMaterial) {
  std::vector<T> statev(1, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0, 0, 0, 0, 0, 0};
  T stress[6], ddsdde[36], pnewdt = 1.0;

  // Step time 2.5, total time 11.0 — deliberately different, which is what the
  // old fixture (both slots equal) could not distinguish.
  const T step_time = 2.5, total_time = 11.0, dtime = 0.25;
  const fortran_name cm("TIMEPROBE");
  T sse = 0, spd = 0, scd = 0, rpl = 0;
  std::vector<T> ddsddt(6, 0.0), drplde(6, 0.0);
  T drpldt = 0;
  const T time[2] = {step_time, total_time};
  const T temp = 0, dtemp = 0, predef = 0, dpred = 0, props = 0;
  const int nprops = 0;  // TIMEPROBE reads no constants
  const T coords[3] = {0, 0, 0};
  const T drot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const T celent = 1.0;
  const T dfgrd[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const int noel = 1, npt = 1, layer = 1, kspt = 1, jstep = 1, kinc = 1;
  int ndi = 3, nshr = 3, ntens = 6, nstatv = 1;

  umat_(stress, statev.data(), ddsdde, &sse, &spd, &scd, &rpl, ddsddt.data(),
        drplde.data(), &drpldt, stran, dstran, time, &dtime, &temp, &dtemp,
        &predef, &dpred, cm.buf, &ndi, &nshr, &ntens, &nstatv, &props, &nprops,
        coords, drot, &pnewdt, &celent, dfgrd, dfgrd, &noel, &npt, &layer,
        &kspt, &jstep, &kinc, 80);

  EXPECT_DOUBLE_EQ(pnewdt, 1.0);
  // stress_11 = t_new, stress_22 = dt, stress_33 = t_old
  EXPECT_DOUBLE_EQ(stress[2], total_time) << "TIME(2), not TIME(1)";
  EXPECT_DOUBLE_EQ(stress[0], total_time + dtime);
  EXPECT_DOUBLE_EQ(stress[1], dtime) << "DTIME must reach the material";
}

// ---------------------------------------------------------------------------
// Guards at the ABI
// ---------------------------------------------------------------------------

/// NTENS inconsistent with NDI/NSHR must be refused. Without the guard the
/// dispatch would build 6-element spans over a 4-element host array.
TEST(UmatInterface, RejectsNtensInconsistentWithNdiNshr) {
  FatalProbe probe;
  std::vector<T> statev(7, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;

  // NDI=3/NSHR=3 implies NTENS=6; claiming 4 is a contradiction.
  call_umat("J2STEEL", stress, statev.data(), ddsdde, stran, dstran, 0.0, 0.1,
            3, 3, 4, 7, &pnewdt);

  EXPECT_EQ(FatalProbe::count, 1);
  EXPECT_NE(FatalProbe::last.find("NTENS"), std::string::npos)
      << FatalProbe::last;
  EXPECT_DOUBLE_EQ(pnewdt, 1.0);
}

/// A fault raised INSIDE the registered builder is still a setup fault, even
/// though it surfaces from the core library as a plain std::runtime_error.
/// Classifying only at this layer's own throw sites missed it: the builder runs
/// arbitrary graph construction that this layer does not own.
TEST(UmatInterface, BuilderFailureIsFatalNotACutback) {
  FatalProbe probe;
  registry::instance().register_model(
      "BROKENBUILD",
      [](ctx_type& ctx, std::span<const double>) {
        param_type p;
        p.insert<std::string>("name", "stepper");
        ctx.create<nm::external_strain_source<policy>>(p);
        p.clear();
        p.insert<std::string>("name", "elastic");
        // Typo: no material called "steper" exists, so wire_inputs() throws a
        // plain std::runtime_error from the core library.
        p.insert<std::string>("strain_producer_name", "steper");
        p.insert<T>("K", K);
        p.insert<T>("G", G);
        ctx.create<nm::linear_elasticity<policy>>(p);
        ctx.finalize();
      },
      [] {
        registry::config c;
        c.strain_source = "stepper";
        c.stress_source = "elastic";
        return c;
      }());

  std::vector<T> statev(1, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;

  call_umat("BROKENBUILD", stress, statev.data(), ddsdde, stran, dstran, 0.0,
            0.1, 3, 3, 6, 1, &pnewdt);

  EXPECT_EQ(FatalProbe::count, 1) << "a broken graph cannot be fixed by a cutback";
  EXPECT_NE(FatalProbe::last.find("steper"), std::string::npos)
      << FatalProbe::last;
  EXPECT_DOUBLE_EQ(pnewdt, 1.0);
}

/// Abaqus input is case-insensitive and delivers CMNAME upper-cased, so the
/// registry must fold case on both sides.
TEST(UmatInterface, MaterialNameLookupIsCaseInsensitive) {
  EXPECT_TRUE(registry::instance().contains("MIXEDCASE"));
  EXPECT_TRUE(registry::instance().contains("mixedcase"));
  EXPECT_TRUE(registry::instance().contains("MixedCase"));

  std::vector<T> statev(7, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;

  // Registered as "mixedCase"; Abaqus will deliver "MIXEDCASE".
  call_umat("MIXEDCASE", stress, statev.data(), ddsdde, stran, dstran, 0.0, 0.1,
            3, 3, 6, 7, &pnewdt);

  EXPECT_DOUBLE_EQ(pnewdt, 1.0);
  EXPECT_GT(std::abs(stress[0]), 0.0) << "the model should have evaluated";
}


// ---------------------------------------------------------------------------
// Deck-driven material constants (PROPS)
// ---------------------------------------------------------------------------

/// The whole point: one compiled builder, two materials, constants supplied by
/// *USER MATERIAL, CONSTANTS= rather than baked into C++.
TEST(UmatInterface, MaterialConstantsComeFromPropsNotTheBuilder) {
  auto uniaxial_tangent = [](const char* name, const T* props, int nprops) {
    std::vector<T> statev(1, 0.0);
    const T stran[6] = {0, 0, 0, 0, 0, 0};
    const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
    T stress[6], ddsdde[36], pnewdt = 1.0;
    call_umat(name, stress, statev.data(), ddsdde, stran, dstran, 0.0, 0.1, 3,
              3, 6, 0, &pnewdt, nullptr, nullptr, nullptr, nullptr, props,
              nprops);
    EXPECT_DOUBLE_EQ(pnewdt, 1.0);
    return ddsdde[0];  // DDSDDE(1,1) = K + 4G/3
  };

  const T soft[2]  = {100.0,  40.0};
  const T stiff[2] = {300.0, 140.0};

  EXPECT_NEAR(uniaxial_tangent("SOFT", soft, 2), 100.0 + 4.0 * 40.0 / 3.0, 1e-9);
  EXPECT_NEAR(uniaxial_tangent("STIFF", stiff, 2), 300.0 + 4.0 * 140.0 / 3.0,
              1e-9);
}

/// A CONSTANTS= count smaller than the model reads is a setup fault. Without
/// require_props this would index past the end of the array and produce a
/// plausible-looking parameter that the deck never supplied.
TEST(UmatInterface, TooFewMaterialConstantsIsFatal) {
  FatalProbe probe;
  std::vector<T> statev(1, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;
  const T only_one[1] = {100.0};

  call_umat("COLDNAME", stress, statev.data(), ddsdde, stran, dstran, 0.0, 0.1,
            3, 3, 6, 0, &pnewdt, nullptr, nullptr, nullptr, nullptr, only_one,
            1);

  EXPECT_EQ(FatalProbe::count, 1);
  EXPECT_NE(FatalProbe::last.find("CONSTANTS"), std::string::npos)
      << FatalProbe::last;
  EXPECT_DOUBLE_EQ(pnewdt, 1.0) << "a wrong CONSTANTS= count is not a cutback";
}

/// The graph is built once and reused, so its parameters are fixed after the
/// first call. A later call arriving with a different NPROPS means the deck
/// contradicts the cached context — which cannot happen for a single material
/// name, and so indicates a dispatch mistake rather than a modelling one.
TEST(UmatInterface, ChangingNpropsForTheSameNameIsFatal) {
  FatalProbe probe;
  std::vector<T> statev(1, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;
  const T two[2] = {100.0, 40.0};
  const T three[3] = {100.0, 40.0, 7.0};

  // Builds the context with 2 constants.
  call_umat("STIFF", stress, statev.data(), ddsdde, stran, dstran, 0.0, 0.1, 3,
            3, 6, 0, &pnewdt, nullptr, nullptr, nullptr, nullptr, two, 2);
  ASSERT_EQ(FatalProbe::count, 0);

  // Same name, different count.
  call_umat("STIFF", stress, statev.data(), ddsdde, stran, dstran, 0.0, 0.1, 3,
            3, 6, 0, &pnewdt, nullptr, nullptr, nullptr, nullptr, three, 3);
  EXPECT_EQ(FatalProbe::count, 1);
  EXPECT_NE(FatalProbe::last.find("constant"), std::string::npos)
      << FatalProbe::last;
}

}  // namespace
