#include <gtest/gtest.h>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
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

void build_j2(ctx_type& ctx) {
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

/// Register once for the whole binary, as a real UMAT library would from a
/// static initialiser.
struct Registration {
  Registration() {
    registry::config cfg;
    cfg.strain_source = "stepper";
    cfg.stress_source = "j2";
    registry::instance().register_model("J2STEEL", build_j2, cfg);
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
               T* spd_io = nullptr, T* scd_io = nullptr) {
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
  const T props = 0;
  const int nprops = 0;
  const T coords[3] = {0, 0, 0};
  const T identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const T* drot = drot_in ? drot_in : identity;
  const T celent = 1.0;
  const T dfgrd0[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const T dfgrd1[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const int noel = 1, npt = 1, layer = 1, kspt = 1, jstep = 1, kinc = 1;

  umat_(stress, statev, ddsdde, sse, spd, scd, &rpl, ddsddt.data(),
        drplde.data(), &drpldt, stran, dstran, time, &dtime, &temp, &dtemp,
        &predef, &dpred, cm.buf, &ndi, &nshr, &ntens, &nstatv, &props, &nprops,
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
  EXPECT_EQ(registry::instance().nstatv("J2STEEL", u::element_case::plane_stress),
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
  build_j2(ctx);
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
  T sse = 0, spd = 0, scd = 0;

  for (int step = 0; step < 40; ++step) {
    call_umat("J2ENERGY", stress, statev.data(), ddsdde, stran, dstran,
              0.1 * step, 0.1, 3, 3, 6, static_cast<int>(statev.size()),
              &pnewdt, nullptr, &sse, &spd, &scd);
    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }

  EXPECT_GT(sse, 0.0) << "elastic energy should be reported";
  EXPECT_GT(spd, 0.0) << "plastic dissipation should accumulate";
  EXPECT_DOUBLE_EQ(scd, 0.0) << "no creep in this model";
}

}  // namespace
