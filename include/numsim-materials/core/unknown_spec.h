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

/// How a residual system (solvers/system.h) updates its unknowns.
enum class system_method { newton, newton_fd, picard };

/// An unknown of a residual system: it starts from `seed` (a tensor from zero) or, with `history`, from its
/// value at the end of the last step, kept per point.
struct system_unknown {
  std::string name;
  unknown_kind kind{unknown_kind::scalar};
  double seed{0};
  bool history{false};
};

/// One equation of a residual system: the residual "material::property" and the unknown it is solved for.
struct system_equation {
  std::string residual;
  std::string unknown;
};

} // namespace numsim::materials

#endif // UNKNOWN_SPEC_H
