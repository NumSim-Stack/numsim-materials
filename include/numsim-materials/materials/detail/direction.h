#ifndef NUMSIM_MATERIALS_DETAIL_DIRECTION_H
#define NUMSIM_MATERIALS_DETAIL_DIRECTION_H

#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>

namespace numsim::materials::detail {

/// A unit vector from a three-component parameter; throws naming `who` otherwise.
template <typename T>
tmech::tensor<T, 3, 1> unit_direction(std::vector<T> const &d, std::string const &who) {
  if (d.size() != 3)
    throw std::invalid_argument(who + ": direction takes three components");
  tmech::tensor<T, 3, 1> n{tmech::adaptor<T const, 3, 1, tmech::full<3>>(d.data())};
  auto const length{tmech::norm(n)};
  if (!(length > 0))
    throw std::invalid_argument(who + ": direction must not be zero");
  return tmech::tensor<T, 3, 1>{n / length};
}

} // namespace numsim::materials::detail

#endif // NUMSIM_MATERIALS_DETAIL_DIRECTION_H
