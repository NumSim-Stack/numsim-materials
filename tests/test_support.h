#ifndef NUMSIM_MATERIALS_TESTS_TEST_SUPPORT_H
#define NUMSIM_MATERIALS_TESTS_TEST_SUPPORT_H

#include <concepts>
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

} // namespace numsim::materials::test

#endif // NUMSIM_MATERIALS_TESTS_TEST_SUPPORT_H
