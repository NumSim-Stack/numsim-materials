#include <iostream>
#include <numsim-materials/autocatalytic_reaction.h>
#include <numsim-materials/scalar_stepper.h>
#include <numsim-materials/scalar_euler_backward.h>

using material_policy = numsim::materials::material_policy_default;
using connection_source = numsim::materials::connection_source;

int main() {
  typename material_policy::ParameterHandler parameter;
  typename material_policy::MaterialHandler mat_handler;
  typename material_policy::PropertyHandler mat_props;
  typename material_policy::FunctionHandler mat_funcs;

  //setup parameter for autocatalytic reaction function
  parameter.insert<std::string>("name", "curing");
  parameter.insert<double>("A", 4000000000.0);
  parameter.insert<double>("E", 84000.0);
  parameter.insert<double>("m", 0.05);
  parameter.insert<double>("n", 2.2);
  parameter.insert<connection_source>("solver_source", connection_source{"solver", "delta"});
  numsim::materials::autocatalytic_reaction<material_policy> curing(parameter, mat_props, mat_funcs,
                                                                   mat_handler);

  //setup parameter for pseudo time stepper
  parameter.clear();
  parameter.insert<std::string>("name", "time");
  parameter.insert<double>("increment", 10);
  numsim::materials::scalar_stepper<material_policy> time(parameter, mat_props, mat_funcs, mat_handler);

  //setup parameter for pseudo temperature stepper
  parameter.clear();
  parameter.insert<std::string>("name", "temperature");
  parameter.insert<double>("increment", 1);
  numsim::materials::scalar_stepper<material_policy> temp(parameter, mat_props, mat_funcs, mat_handler);

  //setup parameter for backward euler solver
  parameter.clear();
  parameter.insert<std::string>("name", "solver");
  parameter.insert<connection_source>("function_source", connection_source{"curing", "jac_res"});
  numsim::materials::scalar_euler_backward<material_policy> solver(parameter, mat_props, mat_funcs, mat_handler);

  //final all property, material and functions queries
  mat_props.final_queries();
  mat_handler.final_queries();
  mat_funcs.final_queries();

  //loop over time steps
  for (int i{0}; i < 1000; ++i) {
    curing.update();
    std::cout << "step " << i << std::endl;
  }

  return 0;
}
