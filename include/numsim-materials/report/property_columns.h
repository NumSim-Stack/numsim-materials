#ifndef NUMSIM_MATERIALS_REPORT_PROPERTY_COLUMNS_H
#define NUMSIM_MATERIALS_REPORT_PROPERTY_COLUMNS_H

#include <array>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/property_value.h"
#include "numsim-materials/core/tensor_identities.h"

/// Report columns wired to graph properties: "material::property" or "material::property[i,j,...]" (1-based).
namespace numsim::materials::report {

template <typename Traits>
class property_columns {
public:
  using value_type = typename Traits::value_type;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  property_columns(material_context<Traits> &ctx, std::vector<std::pair<std::string, std::string>> const &wiring) {
    for (auto const &[name, source] : wiring)
      add(ctx, name, source, false);
  }

  template <typename Spec>
  property_columns(material_context<Traits> &ctx, std::vector<Spec> const &wiring) {
    for (auto const &c : wiring)
      add(ctx, c.name, c.source, c.full);
  }

  [[nodiscard]] std::vector<std::string> const &names() const noexcept { return m_names; }

  /// Current values of the wired properties, one per column.
  [[nodiscard]] std::vector<value_type> values() const {
    std::vector<value_type> v;
    v.reserve(m_read.size());
    for (auto const &read : m_read)
      v.push_back(read());
    return v;
  }

private:
  /// `full`: a rank-2 tensor without component gives nine columns instead of the symmetric six.
  void add(material_context<Traits> &ctx, std::string const &name, std::string const &source, bool full) {
    auto const sep{source.find("::")};
    auto const open{source.find('[')};
    bool const bracketed{open != std::string::npos};
    if (sep == std::string::npos || sep == 0 || (bracketed && (open < sep + 3 || source.back() != ']')))
      throw std::invalid_argument("column '" + name + "': expected \"material::property[i,j]\", got '" + source + "'");
    auto const material{source.substr(0, sep)};
    auto const prop{source.substr(sep + 2, bracketed ? open - sep - 2 : std::string::npos)};
    auto *p{ctx.find_property(material, prop)};
    if (p == nullptr)
      throw std::invalid_argument("column '" + name + "': no property '" + material + "::" + prop + "'");
    auto const idx{indices(name, bracketed ? source.substr(open + 1, source.size() - open - 2) : std::string{})};

    if (auto const *s{property_value<value_type>(p)}) {
      require(name, idx.empty(), "a scalar takes no component");
      push(name, [s] { return *s; });
    } else if (auto const *flag{property_value<int>(p)}) { // predicates such as is_yielding
      require(name, idx.empty(), "a flag takes no component");
      push(name, [flag] { return static_cast<value_type>(*flag); });
    } else if (auto const *t2{property_value<tensor2>(p)}) {
      if (idx.empty()) {
        for (std::size_t a{0}; a < (full ? full_components.size() : voigt_components.size()); ++a)
          push(name + "_" + std::string{full_components[a]},
               [t2, i = full_indices[a].first, j = full_indices[a].second] { return (*t2)(i, j); });
      } else {
        require(name, idx.size() == 2, "a rank-2 tensor takes two indices");
        push(name, [t2, i = idx[0], j = idx[1]] { return (*t2)(i, j); });
      }
    } else if (auto const *t4{property_value<tensor4>(p)}) {
      require(name, idx.size() == 4, "a rank-4 tensor takes four indices");
      push(name, [t4, idx] { return (*t4)(idx[0], idx[1], idx[2], idx[3]); });
    } else {
      throw std::invalid_argument("column '" + name + "': '" + source + "' is not a scalar, a flag, a rank-2 or a rank-4 3D tensor property");
    }
  }

  /// "1,2" -> {0, 1}
  static std::vector<std::size_t> indices(std::string const &name, std::string const &list) {
    std::vector<std::size_t> idx;
    std::size_t pos{0};
    while (pos < list.size()) {
      auto const end{std::min(list.find(',', pos), list.size())};
      auto const item{list.substr(pos, end - pos)};
      require(name, item.find_first_not_of(" 0123456789") == std::string::npos && item.find_first_of("0123456789") != std::string::npos,
              "indices are comma-separated numbers");
      auto const i{std::stoul(item)};
      require(name, i >= 1 && i <= 3, "indices run from 1 to 3");
      idx.push_back(i - 1);
      pos = end + 1;
    }
    return idx;
  }

  static void require(std::string const &name, bool ok, char const *what) {
    if (!ok)
      throw std::invalid_argument("column '" + name + "': " + what);
  }

  void push(std::string name, std::function<value_type()> read) {
    m_names.push_back(std::move(name));
    m_read.push_back(std::move(read));
  }

  std::vector<std::string> m_names;
  std::vector<std::function<value_type()>> m_read;
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_PROPERTY_COLUMNS_H
