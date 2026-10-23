#ifndef INPUT_TYPES_H
#define INPUT_TYPES_H
#include <algorithm>
#include <vector>
#include <numsim-core/property_graph/input_types.h>
#include "numsim-materials/core/property_traits.h"

/// The read check and the verification of constants (property_engine). CMake defines it for every user of the
/// target (option NUMSIM_MATERIALS_DEBUG_CHECKS); without CMake it follows NDEBUG and must agree across the
/// translation units of a program.
#ifndef NUMSIM_MATERIALS_DEBUG_CHECKS
#ifdef NDEBUG
#define NUMSIM_MATERIALS_DEBUG_CHECKS 0
#else
#define NUMSIM_MATERIALS_DEBUG_CHECKS 1
#endif
#endif

namespace numsim::materials {
using numsim_core::input_wire_base;
using numsim_core::connection_source;

namespace detail {
/// The inputs the running callback declared (null: it declared none); with the read check on, the first read
/// of any other input is noted (property_engine checks it).
struct read_scope {
  std::vector<input_wire_base const*> const* declared;
  input_wire_base const* undeclared{nullptr};
};
inline thread_local read_scope* current_reads{nullptr};

inline void note_read([[maybe_unused]] input_wire_base const* input) noexcept {
#if NUMSIM_MATERIALS_DEBUG_CHECKS
  if (auto* scope{current_reads}; scope && scope->declared && !scope->undeclared && !std::ranges::contains(*scope->declared, input))
    scope->undeclared = input;
#endif
}
} // namespace detail

/// numsim-core's input with its reads noted (read check on), so the engine checks them against the declared ones.
template <typename T, typename Traits = property_traits>
class input_property : public numsim_core::input_property<T, Traits> {
public:
  using numsim_core::input_property<T, Traits>::input_property;
  const T& get() const noexcept {
    detail::note_read(this);
    return numsim_core::input_property<T, Traits>::get();
  }
};

template <typename T, typename Traits = property_traits>
class input_history : public numsim_core::input_history<T, Traits> {
public:
  using numsim_core::input_history<T, Traits>::input_history;
  const T& old_value() const noexcept {
    detail::note_read(this);
    return numsim_core::input_history<T, Traits>::old_value();
  }
  const T& new_value() const noexcept {
    detail::note_read(this);
    return numsim_core::input_history<T, Traits>::new_value();
  }
};
} // namespace numsim::materials
#endif
