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
/// get() throws before finalize() has wired it (a null target would otherwise surface only in Debug).
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

  /// The referenced material; throws until finalize() has wired it.
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

  std::string m_name;
  T* m_ptr{nullptr};
};

} // namespace numsim::materials

#endif // MATERIAL_REF_H
