#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <tmech/tmech.h>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/umat/calculix_interface.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/umat_interface.h"

// umat_ is the reference path; the clx_* symbols are the adapter under test.
NUMSIM_MATERIALS_DEFINE_UMAT(numsim::materials::material_policy_default)
NUMSIM_MATERIALS_DEFINE_CALCULIX_BEHAVIOUR(
    numsim::materials::material_policy_default, clx_linear_elastic_, "LINELAS")
NUMSIM_MATERIALS_DEFINE_CALCULIX_BEHAVIOUR(
    numsim::materials::material_policy_default, clx_j2_, "J2CLX")
NUMSIM_MATERIALS_DEFINE_CALCULIX_BEHAVIOUR(
    numsim::materials::material_policy_default, clx_asym_, "ASYMTANGENT")
NUMSIM_MATERIALS_DEFINE_CALCULIX_BEHAVIOUR(
    numsim::materials::material_policy_default, clx_missing_, "NOSUCHMODEL")
NUMSIM_MATERIALS_DEFINE_CALCULIX_BEHAVIOUR(
    numsim::materials::material_policy_default, clx_time_, "TIMEPROBE")

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using registry = u::umat_registry<policy>;

// Deliberately asymmetric moduli so K and G cannot alias each other.
constexpr T K = 166.67;
constexpr T G = 76.92;
constexpr T sigma_0 = 50.0;
constexpr T H_mod = 1000.0;

/// props[0] = K, props[1] = G. One builder, constants entirely from the deck.
void build_deck_elastic(ctx_type& ctx, std::span<const double> props) {
  u::require_props(props, 2, "linelas");
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

/// A stateful model, so the xstateini -> xstate plumbing has real state.
void build_j2(ctx_type& ctx, std::span<const double> /*props*/) {
  param_type p;
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
  ctx.create<nm::local_newton<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<T>("K", H_mod);
  ctx.create<nm::linear_isotropic_hardening<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<T>("K", K);
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T>("G", G);
  p.insert<T>("sigma_0", sigma_0);
  ctx.create<nm::j2_plasticity<policy>>(p);
  ctx.finalize();
}

// The six canonical slots {11,22,33,12,13,23} as (i,j) index pairs.
constexpr int slot_i[6] = {0, 1, 2, 0, 0, 1};
constexpr int slot_j[6] = {0, 1, 2, 1, 2, 2};

/// Deliberately NOT symmetric, so a transposed or one-sided read is visible.
constexpr T asym_value(int a, int b) {
  return static_cast<T>(100 * (a + 1) + (b + 1));
}

/// Minor-symmetric (which tangent_to_buffer asserts) but major-ASYMMETRIC, as a
/// non-associative model is. Stress stays zero; this pins the tangent packing.
template <typename Traits>
class asym_tangent_material final
    : public nm::material_base<asym_tangent_material<Traits>, Traits> {
public:
  using base = nm::material_base<asym_tangent_material<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  template <typename... Args>
  explicit asym_tangent_material(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>(
            "stress", &asym_tangent_material::compute)),
        m_tangent(base::template add_output<tensor4>("tangent")) {
    fill_tangent();
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() {
    m_stress.fill(0.0);
    fill_tangent();
  }

private:
  void fill_tangent() {
    m_tangent.fill(0.0);
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) {
        const value_type v = asym_value(a, b);
        const int i = slot_i[a], j = slot_j[a];
        const int k = slot_i[b], l = slot_j[b];
        // Every minor-symmetric permutation gets the same value.
        m_tangent(i, j, k, l) = v;
        m_tangent(j, i, k, l) = v;
        m_tangent(i, j, l, k) = v;
        m_tangent(j, i, l, k) = v;
      }
    }
  }

  tensor2& m_stress;
  tensor4& m_tangent;
};

/// Reports the time it was bound with; nothing else through the shim depends on
/// time, so without this a wrong TIME(2) is undetectable at the ABI.
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
    m_stress(0, 0) = m_time.old_value();                        // start of incr
    m_stress(1, 1) = m_time.new_value();                        // end of incr
    m_stress(2, 2) = m_time.new_value() - m_time.old_value();   // == dtime
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

void build_asym(ctx_type& ctx, std::span<const double> /*props*/) {
  param_type p;
  p.insert<std::string>("name", "stepper");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "probe");
  ctx.create<asym_tangent_material<policy>>(p);
  ctx.finalize();
}

struct Registration {
  Registration() {
    registry::config el;
    el.strain_source = "stepper";
    el.stress_source = "elastic";
    registry::instance().register_model("LINELAS", build_deck_elastic, el);

    registry::config j2;
    j2.strain_source = "stepper";
    j2.stress_source = "j2";
    registry::instance().register_model("J2CLX", build_j2, j2);
    // Same graph, reached through the Abaqus entry for the cross-check.
    registry::instance().register_model("J2REF", build_j2, j2);

    registry::config as;
    as.strain_source = "stepper";
    as.stress_source = "probe";
    registry::instance().register_model("ASYMTANGENT", build_asym, as);

    registry::config tp;
    tp.strain_source = "stepper";
    tp.stress_source = "probe";
    tp.time_source = "clock";
    registry::instance().register_model("TIMEPROBE", build_time_probe, tp);
  }
};
const Registration registration;

/// CMNAME as Fortran passes it: character*80, blank padded, no NUL.
struct fortran_name {
  char buf[80];
  explicit fortran_name(const std::string& s) {
    for (std::size_t i = 0; i < 80; ++i) buf[i] = ' ';
    for (std::size_t i = 0; i < s.size() && i < 80; ++i) buf[i] = s[i];
  }
};

/// A fatal fault must terminate the analysis. The handler is replaced so the
/// test can observe it instead of the runner being killed by XIT/abort.
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

using clx_fn = void (*)(const char*, const int*, const int*, const int*,
                        const double*, const double*, const double*,
                        const double*, const double*, const double*,
                        const double*, const double*, const int*, const double*,
                        const double*, const double*, const double*, const int*,
                        const int*, const int*, const int*, const double*,
                        double*, double*, double*, const int*, const double*,
                        const double*, double*, const int*, const int);

/// Defaults for a well-formed solid-3D call; tests set only what they vary.
struct clx_call {
  int iel = 1;
  int iint = 1;
  int mi1 = 1;
  int nstatv = 0;
  int icmd = 0;
  int ielas = 0;
  int iorien = 0;
  T dtime = 1.0;
  T time = 0.0;   // step time at END of increment
  T ttime = 0.0;  // total time at START of step
  const T* beta = nullptr;
  const T* mprops = nullptr;
  int nconst = 0;
  T pnewdt = -1.0;

  void run(clx_fn fn, const std::string& amat, const T* emec, const T* emec0,
           const T* statev_old, T* statev_new, T* stress, T* stiff) {
    const fortran_name am(amat);
    const int kode = -100 - nconst;
    const int ipkon = 0;
    const T zero6[6] = {0, 0, 0, 0, 0, 0};
    const T ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const T voj = 1, vj = 1, t1l = 0;
    const T pgauss[3] = {0, 0, 0};
    const T orab[7] = {0, 0, 0, 0, 0, 0, 0};
    const int ithermal = 0;
    const T* beta_p = beta ? beta : zero6;
    const T no_props = 0;
    const T* props_p = mprops ? mprops : &no_props;

    fn(am.buf, &iel, &iint, &kode, props_p, emec, emec0, beta_p, ident, &voj,
       ident, &vj, &ithermal, &t1l, &dtime, &time, &ttime, &icmd, &ielas, &mi1,
       &nstatv, statev_old, statev_new, stress, stiff, &iorien, pgauss, orab,
       &pnewdt, &ipkon, 80);
  }
};

/// Drive the Abaqus umat_ reference for a solid-3D point. STRAN/DSTRAN are the
/// ENGINEERING-shear strain at the start and its increment.
void call_umat_reference(const std::string& name, T* stress, T* statev,
                         T* ddsdde, const T* stran, const T* dstran,
                         int nstatv, const T* props, int nprops,
                         T total_time = 0.0, T dtime_in = 1.0) {
  const fortran_name cm(name);
  T sse = 0, spd = 0, scd = 0, rpl = 0, drpldt = 0, pnewdt = -1;
  T ddsddt[6] = {0}, drplde[6] = {0};
  const T time[2] = {total_time, total_time};
  const T dtime = dtime_in;
  const T temp = 0, dtemp = 0, predef = 0, dpred = 0;
  const T coords[3] = {0, 0, 0};
  const T identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const T celent = 1.0;
  int ndi = 3, nshr = 3, ntens = 6, nstatv_v = nstatv, nprops_v = nprops;
  const int noel = 1, npt = 1, layer = 1, kspt = 1, jstep = 1, kinc = 1;
  const T no_props = 0;
  const T* props_p = props ? props : &no_props;

  umat_(stress, statev, ddsdde, &sse, &spd, &scd, &rpl, ddsddt, drplde, &drpldt,
        stran, dstran, time, &dtime, &temp, &dtemp, &predef, &dpred, cm.buf,
        &ndi, &nshr, &ntens, &nstatv_v, props_p, &nprops_v, coords, identity,
        &pnewdt, &celent, identity, identity, &noel, &npt, &layer, &kspt,
        &jstep, &kinc, 80);
}

// ---------------------------------------------------------------------------
// Convention translation
// ---------------------------------------------------------------------------

/// The same physical strain, in each hook's own convention, must give the same
/// stress and tangent. Nonzero shear exposes the engineering doubling; nonzero
/// emec0 exposes the stran/dstran split (at emec0 = 0 a swap is invisible).
TEST(CalculiXInterface, MatchesTheAbaqusEntryForTheSamePhysicalStrain) {
  const T props[2] = {K, G};
  const int nconst = 2;
  const int nstatv = static_cast<int>(
      registry::instance().nstatv("LINELAS", std::span<const T>(props, 2)));

  // Both start and end strain are nonzero in all six slots.
  const T emec0[6] = {2.0e-4, 1.0e-4, -3.0e-4, 1.5e-4, 2.5e-4, -2.0e-4};
  const T emec[6] = {1.0e-3, -4.0e-4, 2.0e-4, 3.0e-4, -1.5e-4, 5.0e-4};

  // --- Abaqus reference: engineering shear, STRAN = start, DSTRAN = increment.
  T stran[6], dstran[6];
  for (int i = 0; i < 6; ++i) {
    const T s = (i < 3) ? 1.0 : 2.0;
    stran[i] = s * emec0[i];
    dstran[i] = s * (emec[i] - emec0[i]);
  }
  std::vector<T> ref_statev(nstatv > 0 ? nstatv : 1, 0.0);
  T ref_stress[6] = {0, 0, 0, 0, 0, 0};
  T ref_ddsdde[36] = {0};
  call_umat_reference("LINELAS", ref_stress, ref_statev.data(), ref_ddsdde,
                      stran, dstran, nstatv, props, nconst);

  // --- CalculiX adapter: native tensorial strain, split STATEV, packed stiff.
  std::vector<T> clx_old(nstatv > 0 ? nstatv : 1, 0.0);
  std::vector<T> clx_new(nstatv > 0 ? nstatv : 1, 0.0);
  T clx_stress[6] = {0, 0, 0, 0, 0, 0};
  T clx_stiff[21] = {0};
  clx_call c;
  c.nstatv = nstatv;
  c.mprops = props;
  c.nconst = nconst;
  c.run(&clx_linear_elastic_, "LINELAS", emec, emec0, clx_old.data(),
        clx_new.data(), clx_stress, clx_stiff);

  for (int i = 0; i < 6; ++i)
    EXPECT_NEAR(clx_stress[i], ref_stress[i], 1e-12) << "stress " << i;

  // ddsdde is COLUMN-major, so the reference entry (i, j) is ref[i + j*6].
  for (int j = 0; j < 6; ++j)
    for (int i = 0; i <= j; ++i)
      EXPECT_NEAR(clx_stiff[i + j * (j + 1) / 2], ref_ddsdde[i + j * 6], 1e-12)
          << "stiff(" << i << "," << j << ")";
}

/// Feeding emec straight through would halve every shear-driven stress.
TEST(CalculiXInterface, TensorialShearIsConvertedToEngineering) {
  const T props[2] = {K, G};
  const T e12 = 2.5e-3;
  const T emec0[6] = {0, 0, 0, 0, 0, 0};
  const T emec[6] = {0, 0, 0, e12, 0, 0};

  T statev_old[1] = {0}, statev_new[1] = {0};
  T stress[6] = {0, 0, 0, 0, 0, 0};
  T stiff[21] = {0};
  clx_call c;
  c.mprops = props;
  c.nconst = 2;
  c.run(&clx_linear_elastic_, "LINELAS", emec, emec0, statev_old, statev_new,
        stress, stiff);

  // sigma_12 = 2 * G * eps_12 (tensorial) = G * gamma_12. Dropping the
  // conversion would give exactly half of this.
  EXPECT_NEAR(stress[3], 2.0 * G * e12, 1e-12);
  EXPECT_NEAR(stress[0], 0.0, 1e-12);
  EXPECT_NEAR(stress[1], 0.0, 1e-12);
  EXPECT_NEAR(stress[2], 0.0, 1e-12);
  // The shear-shear tangent is G, with no engineering factor on the tangent.
  EXPECT_NEAR(stiff[3 + 3 * (3 + 1) / 2], G, 1e-9);
}

/// stiff(21) carries the SYMMETRIZED tangent (umat_abaqus.f:335-355). The buffer
/// is column-major, so a row-major read would hand ccx C(j,i) — invisible for
/// any symmetric material, hence the asymmetric probe.
TEST(CalculiXInterface, PacksTheSymmetrizedTangentIntoStiff21) {
  const T emec0[6] = {0, 0, 0, 0, 0, 0};
  const T emec[6] = {1.0e-3, 0, 0, 0, 0, 0};
  T statev_old[1] = {0}, statev_new[1] = {0};
  T stress[6] = {0, 0, 0, 0, 0, 0};
  T stiff[21] = {0};

  clx_call c;
  c.run(&clx_asym_, "ASYMTANGENT", emec, emec0, statev_old, statev_new, stress,
        stiff);

  for (int j = 0; j < 6; ++j) {
    for (int i = 0; i <= j; ++i) {
      const T expected = 0.5 * (asym_value(i, j) + asym_value(j, i));
      EXPECT_NEAR(stiff[i + j * (j + 1) / 2], expected, 1e-9)
          << "stiff(" << i << "," << j << ")";
      if (i != j) {
        // A one-sided read lands on one of these; the average equals neither.
        EXPECT_NE(expected, asym_value(i, j));
        EXPECT_NE(expected, asym_value(j, i));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// State variables: the full-array indexing contract
// ---------------------------------------------------------------------------

/// ccx passes the WHOLE state arrays' base (umat_main.f:40,233), not a per-point
/// slice as umat_abaqus.f:295 does, so the adapter must index by (iint, iel).
/// Driven at element 2, point 3, this fails outright at offset 0. Neighbouring
/// blocks are checked too: right values in the wrong place and dirty neighbours
/// are different bugs.
TEST(CalculiXInterface, IndexesStateByElementAndIntegrationPoint) {
  const int nstatv = static_cast<int>(registry::instance().nstatv("J2CLX"));
  ASSERT_GT(nstatv, 0) << "this test is meaningless without real state";

  constexpr int mi1 = 4;      // integration points per element
  constexpr int nelem = 3;    // elements
  constexpr int iel = 2;      // 1-based
  constexpr int iint = 3;     // 1-based
  const std::size_t total = static_cast<std::size_t>(nstatv) * mi1 * nelem;
  const std::size_t offset =
      static_cast<std::size_t>(nstatv) * ((iint - 1) + mi1 * (iel - 1));

  // Past yield, so state accumulates. TENSORIAL: shear is half the engineering.
  const T de[6] = {0.01, -0.0025, 0.0, 0.0025, 0.0, 0.0};
  constexpr int steps = 40;

  // --- CalculiX path, full arrays, driven at (iel=2, iint=3). ---
  std::vector<T> st_old(total, 0.0), st_new(total, 0.0);
  T clx_stress[6] = {0, 0, 0, 0, 0, 0};
  T clx_stiff[21] = {0};
  T emec0[6] = {0, 0, 0, 0, 0, 0};
  T emec[6] = {0, 0, 0, 0, 0, 0};

  for (int s = 0; s < steps; ++s) {
    for (int i = 0; i < 6; ++i) {
      emec0[i] = emec[i];
      emec[i] = emec0[i] + de[i];
    }
    clx_call c;
    c.iel = iel;
    c.iint = iint;
    c.mi1 = mi1;
    c.nstatv = nstatv;
    c.run(&clx_j2_, "J2CLX", emec, emec0, st_old.data(), st_new.data(),
          clx_stress, clx_stiff);
    // ccx commits xstate -> xstateini between increments.
    st_old = st_new;
  }

  // --- Abaqus reference, one point, its own array. ---
  std::vector<T> ref_statev(nstatv, 0.0);
  T ref_stress[6] = {0, 0, 0, 0, 0, 0};
  T ref_ddsdde[36] = {0};
  T stran[6] = {0, 0, 0, 0, 0, 0};
  T dstran[6];
  for (int i = 0; i < 6; ++i) dstran[i] = (i < 3 ? 1.0 : 2.0) * de[i];
  for (int s = 0; s < steps; ++s) {
    call_umat_reference("J2REF", ref_stress, ref_statev.data(), ref_ddsdde,
                        stran, dstran, nstatv, nullptr, 0);
    for (int i = 0; i < 6; ++i) stran[i] += dstran[i];
  }

  for (int i = 0; i < 6; ++i)
    EXPECT_NEAR(clx_stress[i], ref_stress[i], 1e-9) << "stress " << i;

  // The state landed in THIS point's block...
  for (int i = 0; i < nstatv; ++i)
    EXPECT_NEAR(st_new[offset + i], ref_statev[i], 1e-9)
        << "statev " << i << " at (iel=2, iint=3)";

  // ...and plasticity actually happened, so the block is not trivially zero.
  T norm = 0;
  for (int i = 0; i < nstatv; ++i) norm += std::abs(st_new[offset + i]);
  EXPECT_GT(norm, 1e-8) << "no state accumulated; the test proves nothing";

  // ...and nowhere else was touched.
  for (std::size_t k = 0; k < total; ++k) {
    if (k >= offset && k < offset + static_cast<std::size_t>(nstatv)) continue;
    EXPECT_EQ(st_new[k], 0.0) << "state written outside this point's block at "
                              << k;
  }
}

// ---------------------------------------------------------------------------
// TIME
// ---------------------------------------------------------------------------

/// CalculiX hands over the step time at the END of the increment and the total
/// time at the START of the step; the Abaqus convention wants both rebased onto
/// the START of the increment (umat_abaqus.f:187-188). Passing {time, ttime}
/// through is right only on the first increment of the first step.
///
/// The probe reports the bound times, so the rebasing is asserted exactly: the
/// material sees [ttime + time - dtime, ttime + time].
TEST(CalculiXInterface, RebasesTimeOntoTheStartOfTheIncrement) {
  const T emec0[6] = {0, 0, 0, 0, 0, 0};
  const T emec[6] = {1.0e-4, 0, 0, 0, 0, 0};

  auto times_seen = [&](T step_time_end, T total_time_step_start, T dt) {
    T so[1] = {0}, sn[1] = {0};
    T stress[6] = {0, 0, 0, 0, 0, 0};
    T stiff[21] = {0};
    clx_call c;
    c.dtime = dt;
    c.time = step_time_end;
    c.ttime = total_time_step_start;
    c.run(&clx_time_, "TIMEPROBE", emec, emec0, so, sn, stress, stiff);
    // stress = {t_start, t_end, dt} in slots 0, 1, 2.
    return std::array<T, 3>{stress[0], stress[1], stress[2]};
  };

  // First increment of the first step — the one case the naive mapping gets right.
  {
    const auto t = times_seen(/*time=*/0.1, /*ttime=*/0.0, /*dt=*/0.1);
    EXPECT_NEAR(t[0], 0.0, 1e-12) << "total time at the START of the increment";
    EXPECT_NEAR(t[1], 0.1, 1e-12) << "total time at the END of the increment";
    EXPECT_NEAR(t[2], 0.1, 1e-12) << "dtime";
  }

  // A later increment of a later step: spans [5.0+0.9-0.1, 5.0+0.9] = [5.8, 5.9].
  {
    const auto t = times_seen(/*time=*/0.9, /*ttime=*/5.0, /*dt=*/0.1);
    EXPECT_NEAR(t[0], 5.8, 1e-12)
        << "passing ttime straight through would give 5.0 here";
    EXPECT_NEAR(t[1], 5.9, 1e-12);
    EXPECT_NEAR(t[2], 0.1, 1e-12);
  }
}

// ---------------------------------------------------------------------------
// Guards and error paths
// ---------------------------------------------------------------------------

/// Nothing rotates around the native hook, so ignoring iorien is wrong-frame.
TEST(CalculiXInterface, RefusesALocalOrientation) {
  FatalProbe probe;
  const T props[2] = {K, G};
  const T emec0[6] = {0, 0, 0, 0, 0, 0};
  const T emec[6] = {1.0e-3, 0, 0, 0, 0, 0};
  T so[1] = {0}, sn[1] = {0};
  T stress[6] = {9, 9, 9, 9, 9, 9};
  T stiff[21];
  for (int i = 0; i < 21; ++i) stiff[i] = 9.0;

  clx_call c;
  c.mprops = props;
  c.nconst = 2;
  c.iorien = 1;
  c.run(&clx_linear_elastic_, "LINELAS", emec, emec0, so, sn, stress, stiff);

  EXPECT_EQ(FatalProbe::count, 1);
  EXPECT_NE(FatalProbe::last.find("ORIENTATION"), std::string::npos);
  // Zeroed, so a returning handler cannot leave the solver on a stale buffer.
  for (int i = 0; i < 6; ++i) EXPECT_EQ(stress[i], 0.0);
  for (int i = 0; i < 21; ++i) EXPECT_EQ(stiff[i], 0.0);
}

/// A nonzero initial stress would make a preloaded model wrong from step 1.
TEST(CalculiXInterface, RefusesANonzeroInitialStress) {
  FatalProbe probe;
  const T props[2] = {K, G};
  const T emec0[6] = {0, 0, 0, 0, 0, 0};
  const T emec[6] = {1.0e-3, 0, 0, 0, 0, 0};
  const T beta[6] = {0, 0, 0, 0, 12.5, 0};
  T so[1] = {0}, sn[1] = {0};
  T stress[6] = {0, 0, 0, 0, 0, 0};
  T stiff[21] = {0};

  clx_call c;
  c.mprops = props;
  c.nconst = 2;
  c.beta = beta;
  c.run(&clx_linear_elastic_, "LINELAS", emec, emec0, so, sn, stress, stiff);

  EXPECT_EQ(FatalProbe::count, 1);
  EXPECT_NE(FatalProbe::last.find("INITIAL"), std::string::npos);
}

/// An all-zero beta is the normal case and must NOT trip the guard.
TEST(CalculiXInterface, AcceptsAZeroInitialStress) {
  FatalProbe probe;
  const T props[2] = {K, G};
  const T emec0[6] = {0, 0, 0, 0, 0, 0};
  const T emec[6] = {1.0e-3, 0, 0, 0, 0, 0};
  const T beta[6] = {0, 0, 0, 0, 0, 0};
  T so[1] = {0}, sn[1] = {0};
  T stress[6] = {0, 0, 0, 0, 0, 0};
  T stiff[21] = {0};

  clx_call c;
  c.mprops = props;
  c.nconst = 2;
  c.beta = beta;
  c.run(&clx_linear_elastic_, "LINELAS", emec, emec0, so, sn, stress, stiff);

  EXPECT_EQ(FatalProbe::count, 0);
  EXPECT_GT(stress[0], 0.0);
}

/// kode carries the count as -100 - nconst; too few constants is a setup fault.
TEST(CalculiXInterface, DecodesTheConstantCountFromKode) {
  FatalProbe probe;
  const T props[2] = {K, G};
  const T emec0[6] = {0, 0, 0, 0, 0, 0};
  const T emec[6] = {1.0e-3, 0, 0, 0, 0, 0};
  T so[1] = {0}, sn[1] = {0};
  T stress[6] = {0, 0, 0, 0, 0, 0};
  T stiff[21] = {0};

  clx_call c;
  c.mprops = props;
  c.nconst = 1;  // kode = -101, but the model needs two constants
  c.run(&clx_linear_elastic_, "LINELAS", emec, emec0, so, sn, stress, stiff);

  EXPECT_EQ(FatalProbe::count, 1);
  EXPECT_NE(FatalProbe::last.find("constant"), std::string::npos);
}

/// A symbol bound to a name nothing registered is unrecoverable.
TEST(CalculiXInterface, UnknownModelIsFatal) {
  FatalProbe probe;
  const T emec0[6] = {0, 0, 0, 0, 0, 0};
  const T emec[6] = {1.0e-3, 0, 0, 0, 0, 0};
  T so[1] = {0}, sn[1] = {0};
  T stress[6] = {5, 5, 5, 5, 5, 5};
  T stiff[21];
  for (int i = 0; i < 21; ++i) stiff[i] = 5.0;

  clx_call c;
  c.run(&clx_missing_, "NOSUCHMODEL", emec, emec0, so, sn, stress, stiff);

  EXPECT_EQ(FatalProbe::count, 1);
  for (int i = 0; i < 6; ++i) EXPECT_EQ(stress[i], 0.0);
  // The zeroed 6x6 is packed through like any other result.
  for (int i = 0; i < 21; ++i) EXPECT_EQ(stiff[i], 0.0);
}

/// icmd == 3 requests stress only; stiff must be left as the caller supplied it.
TEST(CalculiXInterface, StressOnlyLeavesStiffUntouched) {
  const T props[2] = {K, G};
  const T emec0[6] = {0, 0, 0, 0, 0, 0};
  const T emec[6] = {1.0e-3, 0, 0, 0, 0, 0};
  T so[1] = {0}, sn[1] = {0};
  T stress[6] = {0, 0, 0, 0, 0, 0};
  T stiff[21];
  for (int i = 0; i < 21; ++i) stiff[i] = -999.0;

  clx_call c;
  c.mprops = props;
  c.nconst = 2;
  c.icmd = 3;
  c.run(&clx_linear_elastic_, "LINELAS", emec, emec0, so, sn, stress, stiff);

  EXPECT_GT(stress[0], 0.0);
  for (int i = 0; i < 21; ++i)
    EXPECT_EQ(stiff[i], -999.0) << "stiff[" << i << "]";
}

}  // namespace
