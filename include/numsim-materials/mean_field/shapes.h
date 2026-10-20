#ifndef NUMSIM_MATERIALS_MEAN_FIELD_SHAPES_H
#define NUMSIM_MATERIALS_MEAN_FIELD_SHAPES_H

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/object_spec.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/materials/detail/direction.h"

namespace numsim::materials::mean_field {

/// An inclusion shape as data: its Eshelby tensor in an isotropic medium of Poisson's ratio nu. For the
/// kernels whose reference medium is the unknown (self-consistent); the others consume Eshelby properties.
template <typename Traits>
class shape {
public:
  using value_type = typename Traits::value_type;
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  virtual ~shape() = default;
  [[nodiscard]] virtual tensor4 eshelby(value_type nu) const = 0;
};

template <typename Derived, typename Traits>
class shape_base : public shape<Traits> {
public:
  using parameter_handler = typename Traits::ParameterHandler;
  using input_parameter_controller = typename Traits::InputParameterController;

  static input_parameter_controller parameters() { return {}; }

protected:
  explicit shape_base(parameter_handler p) : m_parameters(std::move(p)) {
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

template <typename Traits>
class sphere final : public shape_base<sphere<Traits>, Traits> {
public:
  using base = shape_base<sphere<Traits>, Traits>;
  using typename shape<Traits>::value_type;
  using typename shape<Traits>::tensor4;
  explicit sphere(typename base::parameter_handler p) : base(std::move(p)) {}
  [[nodiscard]] tensor4 eshelby(value_type nu) const override { return eshelby::sphere(nu); }
};

/// Parameters: "direction" (default x1).
template <typename Traits>
class cylinder final : public shape_base<cylinder<Traits>, Traits> {
public:
  using base = shape_base<cylinder<Traits>, Traits>;
  using typename shape<Traits>::value_type;
  using typename shape<Traits>::tensor4;
  explicit cylinder(typename base::parameter_handler p)
      : base(std::move(p)), m_direction(detail::unit_direction(base::template get<std::vector<value_type>>("direction"), "cylinder")) {}
  static typename base::input_parameter_controller parameters() {
    typename base::input_parameter_controller para;
    optional<std::vector<value_type>>(para, "direction", std::vector<value_type>{1, 0, 0});
    return para;
  }
  [[nodiscard]] tensor4 eshelby(value_type nu) const override { return eshelby::cylinder(nu, m_direction); }

private:
  tmech::tensor<value_type, 3, 1> m_direction;
};

/// Parameters: "aspect_ratio" (> 0), "direction" of the symmetry axis (default x1).
template <typename Traits>
class spheroid final : public shape_base<spheroid<Traits>, Traits> {
public:
  using base = shape_base<spheroid<Traits>, Traits>;
  using typename shape<Traits>::value_type;
  using typename shape<Traits>::tensor4;
  explicit spheroid(typename base::parameter_handler p)
      : base(std::move(p)), m_aspect_ratio(base::template get<value_type>("aspect_ratio")),
        m_rotation(rotation_to(detail::unit_direction(base::template get<std::vector<value_type>>("direction"), "spheroid"))) {
    if (!(m_aspect_ratio > 0))
      throw std::invalid_argument("spheroid: aspect_ratio must be positive");
  }
  static typename base::input_parameter_controller parameters() {
    typename base::input_parameter_controller para;
    required<value_type>(para, "aspect_ratio");
    optional<std::vector<value_type>>(para, "direction", std::vector<value_type>{1, 0, 0});
    return para;
  }
  [[nodiscard]] tensor4 eshelby(value_type nu) const override {
    return rotate(eshelby::ellipsoid(nu, m_aspect_ratio, value_type{1}, value_type{1}), m_rotation);
  }

private:
  value_type m_aspect_ratio;
  tmech::tensor<value_type, 3, 2> m_rotation;
};

/// Parameters: "semi_axes" along x1, x2, x3.
template <typename Traits>
class ellipsoid final : public shape_base<ellipsoid<Traits>, Traits> {
public:
  using base = shape_base<ellipsoid<Traits>, Traits>;
  using typename shape<Traits>::value_type;
  using typename shape<Traits>::tensor4;
  explicit ellipsoid(typename base::parameter_handler p) : base(std::move(p)), m_a(base::template get<std::vector<value_type>>("semi_axes")) {
    if (m_a.size() != 3)
      throw std::invalid_argument("ellipsoid: semi_axes takes three values");
  }
  static typename base::input_parameter_controller parameters() {
    typename base::input_parameter_controller para;
    required<std::vector<value_type>>(para, "semi_axes");
    return para;
  }
  [[nodiscard]] tensor4 eshelby(value_type nu) const override { return eshelby::ellipsoid(nu, m_a[0], m_a[1], m_a[2]); }

private:
  std::vector<value_type> m_a;
};

template <typename Traits>
using shape_registry_type = spec_registry<shape<Traits>, typename Traits::InputParameterController, typename Traits::ParameterHandler>;

/// The shapes a self-consistent kernel can name; sphere, cylinder, spheroid and ellipsoid built in.
template <typename Traits>
shape_registry_type<Traits> &shape_registry() {
  static shape_registry_type<Traits> registry{[] {
    shape_registry_type<Traits> r{"shape"};
    r.template add<sphere<Traits>>("sphere").template add<cylinder<Traits>>("cylinder");
    r.template add<spheroid<Traits>>("spheroid").template add<ellipsoid<Traits>>("ellipsoid");
    return r;
  }()};
  return registry;
}

} // namespace numsim::materials::mean_field

#endif // NUMSIM_MATERIALS_MEAN_FIELD_SHAPES_H
