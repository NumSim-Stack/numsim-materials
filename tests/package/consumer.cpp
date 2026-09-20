/// Smallest consumer that exercises what the install has to provide.
///
/// It must reach, through the installed headers alone:
///   * tmech            -- every material's tensor type
///   * the Mandel adaptor in solvers/unknown_layout.h, which is the part an
///     older system tmech does not have ("'mandel' is not a member of 'tmech'")
///   * nlohmann/json    -- the JSON factory
///   * Eigen            -- vector_newton's dense solve
///
/// It builds a real two-material graph and runs one update, so a package that
/// compiles but cannot link or run still fails here.
#include <cstdio>

#include <numsim-materials/core/material_context.h>
#include <numsim-materials/default_materials.h>
#include <numsim-materials/io/json_material_factory.h>
#include <numsim-materials/materials/linear_elasticity.h>
#include <numsim-materials/materials/tensor_component_stepper.h>
#include <numsim-materials/solvers/unknown_layout.h>
#include <numsim-materials/solvers/vector_newton.h>

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;

int main() {
  numsim::materials::material_context<policy> ctx;
  policy::ParameterHandler p;

  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.001});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", T{166.67});
  p.insert<T>("G", T{76.92});
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  ctx.finalize();
  ctx.update();

  const auto& stress = ctx.get<tmech::tensor<T, 3, 2>>("elastic", "stress");
  if (!(stress(0, 0) > T{0})) {
    std::puts("consumer: expected a positive stress after one increment");
    return 1;
  }

  // The factory registration path, which pulls in nlohmann/json.
  numsim::materials::register_default_materials<policy>();

  std::puts("consumer: ok");
  return 0;
}
