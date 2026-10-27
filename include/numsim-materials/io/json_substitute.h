#ifndef NUMSIM_MATERIALS_JSON_SUBSTITUTE_H
#define NUMSIM_MATERIALS_JSON_SUBSTITUTE_H

#include <string>
#include <utility>

namespace numsim::materials {

/// `j` with every occurrence of `placeholder` in its string values replaced by `value`; keys stay.
template <typename JsonType>
JsonType substitute(JsonType j, std::string const &placeholder, std::string const &value) {
  if (j.is_string()) {
    auto s{j.template get<std::string>()};
    for (auto pos{s.find(placeholder)}; pos != std::string::npos; pos = s.find(placeholder, pos + value.size()))
      s.replace(pos, placeholder.size(), value);
    return JsonType(s);
  }
  for (auto &v : j) // objects and arrays; a number iterates over itself and is left alone
    if (v.is_string() || v.is_object() || v.is_array())
      v = substitute(std::move(v), placeholder, value);
  return j;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_JSON_SUBSTITUTE_H
