#ifndef NUMSIM_MATERIALS_MEAN_FIELD_KERNEL_H
#define NUMSIM_MATERIALS_MEAN_FIELD_KERNEL_H

#include <span>
#include <string>
#include <utility>
#include <vector>

#include <tmech/tmech.h>

namespace numsim::materials::mean_field {

/// What a kernel sees of one phase; the pointers are null for what it does not read.
template <typename T>
struct phase_view {
  T fraction;
  tmech::tensor<T, 3, 4> const *tangent;
  tmech::tensor<T, 3, 4> const *eshelby;
};

/// The phase properties a kernel reads, so a node wires only those.
struct kernel_needs {
  bool tangents;
  bool eshelby;
};

/// A mean-field scheme as an object: the strain concentration tensors A_r of phases (matrix first), with
/// sum_r c_r A_r = I. The global composite and the strain localisation hold the same kernel.
template <typename Traits>
class kernel {
public:
  using value_type = typename Traits::value_type;
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  virtual ~kernel() = default;
  [[nodiscard]] virtual kernel_needs needs() const noexcept = 0;
  /// The phase names in the order of concentrations(), matrix first; once, before the first call.
  virtual void bind(std::vector<std::string> const &) {}
  virtual void concentrations(std::span<phase_view<value_type> const> phases, std::span<tensor4> A) = 0;
};

/// Parameters checked against Derived::parameters(), like a material's.
template <typename Derived, typename Traits>
class kernel_base : public kernel<Traits> {
public:
  using parameter_handler = typename Traits::ParameterHandler;
  using input_parameter_controller = typename Traits::InputParameterController;

  static input_parameter_controller parameters() { return {}; }

protected:
  explicit kernel_base(parameter_handler p) : m_parameters(std::move(p)) {
    static auto const check{Derived::parameters()};
    check.check_parameter(m_parameters);
  }

  template <typename U>
  [[nodiscard]] U const &get(std::string const &key) const {
    return m_parameters.template get<U>(key);
  }

private:
  parameter_handler m_parameters;
};

} // namespace numsim::materials::mean_field

#endif // NUMSIM_MATERIALS_MEAN_FIELD_KERNEL_H
