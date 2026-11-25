#ifndef NUMSIM_MATERIALS_TESTS_TEST_SUPPORT_H
#define NUMSIM_MATERIALS_TESTS_TEST_SUPPORT_H

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"

#include <tmech/tmech.h>

#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

/// Helpers shared by the test files.
namespace numsim::materials::test {

/// One parameter for add(): strings are stored as std::string, floating-point values as the context's
/// value_type, everything else with the type it is written with (write 1.0, not 1, for a value_type).
template <typename Context> class parameter {
public:
  using handler = typename Context::parameter_handler;
  using value_type = typename Context::value_type;

  template <typename V>
  parameter(std::string key, V &&value) : m_insert{make(std::move(key), std::forward<V>(value))} {}

  void insert_into(handler &h) const { m_insert(h); }

private:
  std::function<void(handler &)> m_insert;

  template <typename V> static std::function<void(handler &)> make(std::string key, V &&value) {
    using stored = std::remove_cvref_t<V>;
    if constexpr (std::is_convertible_v<V, std::string_view>)
      return [key = std::move(key), v = std::string{std::string_view{value}}](handler &h) {
        h.template insert<std::string>(key, std::string{v});
      };
    else if constexpr (std::floating_point<stored>)
      return [key = std::move(key), v = static_cast<value_type>(value)](handler &h) { h.template insert<value_type>(key, value_type{v}); };
    else
      return [key = std::move(key), v = stored(std::forward<V>(value))](handler &h) { h.template insert<stored>(key, stored(v)); };
  }
};

/// ctx.create<Material>() with the parameters "name" = name and `parameters`.
template <typename Material, typename Context>
Material &add(Context &ctx, std::string name, std::initializer_list<parameter<std::type_identity_t<Context>>> parameters = {}) {
  typename Context::parameter_handler p;
  p.template insert<std::string>("name", std::move(name));
  for (auto const &entry : parameters)
    entry.insert_into(p);
  return ctx.template create<Material>(p);
}

/// A constant_scalar "name" with the given value.
template <typename Traits>
constant_scalar<Traits> &constant(material_context<Traits> &ctx, std::string name, typename Traits::value_type value) {
  return add<constant_scalar<Traits>>(ctx, std::move(name), {{"value", value}});
}

/// |a - b| / |b|.
template <typename A> auto relative(A const &a, A const &b) { return tmech::norm(A{a - b}) / tmech::norm(b); }

/// The strain with e in its (0, 0) component.
template <std::floating_point T> tmech::tensor<T, 3, 2> uniaxial(T e) {
  tmech::tensor<T, 3, 2> eps;
  eps(0, 0) = e;
  return eps;
}

/// The deformation gradient diag(l1, l2, l3).
template <std::floating_point T> tmech::tensor<T, 3, 2> stretch(T l1, T l2 = T{1}, T l3 = T{1}) {
  tmech::tensor<T, 3, 2> F;
  F(0, 0) = l1;
  F(1, 1) = l2;
  F(2, 2) = l3;
  return F;
}

/// A blank-padded Fortran CHARACTER*80, as the UMAT entry points receive the material name.
struct fortran_name {
  char buf[80];
  explicit fortran_name(std::string const &s) {
    for (auto &c : buf)
      c = ' ';
    for (std::size_t i = 0; i < s.size() && i < 80; ++i)
      buf[i] = s[i];
  }
};

} // namespace numsim::materials::test

#endif // NUMSIM_MATERIALS_TESTS_TEST_SUPPORT_H
