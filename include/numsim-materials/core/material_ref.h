#ifndef MATERIAL_REF_H
#define MATERIAL_REF_H

#include <stdexcept>
#include <string>

namespace numsim::materials {

template<typename Traits>
class material_interface;

/// Type-erased base for lazy material references.
/// Resolved at finalize() time, same pattern as input_wire_base for properties.
template<typename Traits>
class material_ref_base {
public:
  virtual ~material_ref_base() = default;
  virtual const std::string& target_name() const noexcept = 0;
  virtual bool is_wired() const noexcept = 0;
  virtual void wire(material_interface<Traits>& target) = 0;
};

/// Typed lazy reference to another material.
/// Stores the name at construction, pointer resolved at finalize().
/// After finalize(), get() is noexcept — guaranteed wired.
template<typename T, typename Traits>
class material_ref final : public material_ref_base<Traits> {
public:
  explicit material_ref(std::string name) : m_name(std::move(name)) {}

  const std::string& target_name() const noexcept override { return m_name; }
  bool is_wired() const noexcept override { return m_ptr != nullptr; }

  void wire(material_interface<Traits>& target) override {
    m_ptr = dynamic_cast<T*>(&target);
    if (!m_ptr)
      throw std::runtime_error(
          "material_ref::wire(): material '" + m_name + "' is not of the requested type");
  }

  /// The referenced material. Only valid after finalize() has wired it.
  ///
  /// The guard used to be an assert alone, and the Release configuration is
  /// -O3 -DNDEBUG, so the only thing standing between a use-before-wire and a
  /// null dereference compiled away in exactly the builds that run real
  /// analyses. wire_materials() does reject a missing or wrongly-typed target,
  /// but it runs at finalize(): anything that reaches get() earlier -- a
  /// compute callback invoked by hand, a constructor that resolves too eagerly
  /// -- got no diagnosis at all, just a crash somewhere else.
  ///
  /// A throw rather than a finalize-time sweep, because after wire_materials()
  /// succeeds every ref is wired by construction, so a sweep would be
  /// unreachable and untestable. The cost is one correctly-predicted branch on
  /// a path that is a plain pointer dereference and not a measured hotspot.
  const T& get() const {
    if (!m_ptr) throw_unwired();
    return *m_ptr;
  }

  T& get() {
    if (!m_ptr) throw_unwired();
    return *m_ptr;
  }

private:
  [[noreturn]] void throw_unwired() const {
    throw std::logic_error(
        "material_ref::get(): reference to material '" + m_name +
        "' is not wired yet. Material references resolve during "
        "material_context::finalize(); this was reached before that.");
  }

private:
  std::string m_name;
  T* m_ptr{nullptr};
};

} // namespace numsim::materials

#endif // MATERIAL_REF_H
