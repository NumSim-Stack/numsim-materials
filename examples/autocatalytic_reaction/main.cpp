#include <print>
#include <numsim-materials/default_materials.h>
#include <numsim-materials/core/material_context.h>
#include <numsim-materials/materials/scalar_stepper.h>
#include <numsim-materials/materials/autocatalytic_reaction.h>
#include <numsim-materials/solvers/backward_euler.h>

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;

int main() {
  numsim::materials::material_context<policy> ctx;
  typename policy::ParameterHandler parameter;

  // Time stepper (dt = 10s)
  parameter.clear();
  parameter.insert<std::string>("name", "time");
  parameter.insert<T>("increment", T{10});
  ctx.create<numsim::materials::scalar_stepper<policy>>(parameter);

  // Temperature stepper (80°C isothermal)
  parameter.clear();
  parameter.insert<std::string>("name", "temperature");
  parameter.insert<T>("increment", T{0});
  ctx.create<numsim::materials::scalar_stepper<policy>>(parameter);
  auto& temp = static_cast<numsim::materials::history_property<T>&>(
      *ctx.find_property("temperature", "state"));
  temp.old_value() = T{80};
  temp.new_value() = T{80};

  // Backward Euler solver
  parameter.clear();
  parameter.insert<std::string>("name", "solver");
  parameter.insert<std::string>("function", "curing");
  parameter.insert<T>("tolerance", T{5e-12});
  parameter.insert<int>("max_iter", 100);
  ctx.create<numsim::materials::backward_euler<policy>>(parameter);

  // Autocatalytic curing reaction
  parameter.clear();
  parameter.insert<std::string>("name", "curing");
  parameter.insert<T>("A", T{4e9});
  parameter.insert<T>("E", T{84000});
  parameter.insert<T>("m", T{0.05});
  parameter.insert<T>("n", T{2.2});
  parameter.insert<std::string>("solver_name", "solver");
  ctx.create<numsim::materials::autocatalytic_reaction<policy>>(parameter);

  ctx.finalize();

  auto& curing = static_cast<numsim::materials::history_property<T>&>(
      *ctx.find_property("curing", "current_state"));
  auto& time = static_cast<numsim::materials::history_property<T>&>(
      *ctx.find_property("time", "state"));

  for (int i = 0; i < 200; ++i) {
    ctx.update();
    if (i % 10 == 0)
      std::println("step {:3d}: time = {:6.0f}s, curing = {:.6e}",
                   i, time.new_value(), curing.new_value());
    ctx.commit();
  }

  return 0;
}
