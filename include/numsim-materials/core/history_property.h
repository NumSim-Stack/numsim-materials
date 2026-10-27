#ifndef HISTORY_PROPERTY_H
#define HISTORY_PROPERTY_H
#include <cstddef>
#include <istream>
#include <ostream>
#include <tmech/tmech.h>

// Checkpoint hooks for tensor history (found by ADL from numsim-core's history_property). Declared before
// numsim-core's header, which every numsim-materials header reaches through this one only; belongs in tmech.
namespace tmech {
template <typename T, std::size_t Dim, std::size_t Rank>
void numsim_serialize(std::ostream& os, tensor<T, Dim, Rank> const& old_v, tensor<T, Dim, Rank> const& new_v) {
  for (auto const* value : {&old_v, &new_v})
    os.write(reinterpret_cast<char const*>(value->raw_data()), sizeof(T) * value->size());
}

template <typename T, std::size_t Dim, std::size_t Rank>
void numsim_deserialize(std::istream& is, tensor<T, Dim, Rank>& old_v, tensor<T, Dim, Rank>& new_v) {
  for (auto* value : {&old_v, &new_v})
    is.read(reinterpret_cast<char*>(value->raw_data()), sizeof(T) * value->size());
}
} // namespace tmech

#include <numsim-core/property_graph/history_property.h>
namespace numsim::materials {
  using numsim_core::history_property;
  using numsim_core::make_history_property;
}
#endif
