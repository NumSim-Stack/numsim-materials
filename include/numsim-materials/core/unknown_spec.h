#ifndef UNKNOWN_SPEC_H
#define UNKNOWN_SPEC_H

#include <string>
#include <utility>
#include <vector>

namespace numsim::materials {

/// Runtime description of the unknown set of a coupled local system.
///
/// Lives in core/ rather than solvers/ so that the JSON layer can read it
/// without depending on the solver or on tmech. A bounded switch in
/// vector_newton turns each spec into the corresponding compile-time kind in
/// solvers/unknown_layout.h; the set is deliberately small and closed so that
/// switch stays exhaustive.
///
/// Symmetry has to be declared here because it cannot be recovered from the
/// C++ type: tmech's tensor<T,Dim,2> is symmetry-agnostic storage, and
/// property_traits carries no shape metadata.
enum class unknown_kind { scalar, sym_tensor };

struct unknown_spec {
  std::string name;
  unknown_kind kind{unknown_kind::scalar};
};

/// Identifies a Jacobian block by the names of its row and column unknowns.
/// Used to declare structurally-zero blocks, which are then never wired.
using block_ref = std::pair<std::string, std::string>;

} // namespace numsim::materials

#endif // UNKNOWN_SPEC_H
