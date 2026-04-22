#ifndef NUMSIM_MATERIALS_BUTCHER_TABLEAU_H
#define NUMSIM_MATERIALS_BUTCHER_TABLEAU_H

#include <vector>

namespace numsim::materials {

/// Runtime Butcher tableau for Runge-Kutta methods.
///
///   c₁ | a₁₁ a₁₂ ... a₁ₛ
///   c₂ | a₂₁ a₂₂ ... a₂ₛ
///   ...
///   cₛ | aₛ₁ aₛ₂ ... aₛₛ
///   ---|--------------------
///      | b₁  b₂  ... bₛ
struct butcher_tableau {
  int stages;
  std::vector<std::vector<double>> a;
  std::vector<double> b;
  std::vector<double> c;

  bool is_explicit() const {
    for (int i = 0; i < stages; ++i)
      for (int j = i; j < stages; ++j)
        if (a[i][j] != 0.0) return false;
    return true;
  }

  bool is_dirk() const {
    for (int i = 0; i < stages; ++i)
      for (int j = i + 1; j < stages; ++j)
        if (a[i][j] != 0.0) return false;
    return true;
  }
};

// --- Factory functions ---

inline butcher_tableau forward_euler() {
  return {1, {{0}}, {1}, {0}};
}

inline butcher_tableau explicit_midpoint() {
  return {2,
    {{0, 0}, {0.5, 0}},
    {0, 1},
    {0, 0.5}};
}

inline butcher_tableau rk4() {
  return {4,
    {{0, 0, 0, 0},
     {0.5, 0, 0, 0},
     {0, 0.5, 0, 0},
     {0, 0, 1, 0}},
    {1.0/6, 1.0/3, 1.0/3, 1.0/6},
    {0, 0.5, 0.5, 1}};
}

inline butcher_tableau implicit_euler() {
  return {1, {{1}}, {1}, {1}};
}

inline butcher_tableau implicit_midpoint() {
  return {1, {{0.5}}, {1}, {0.5}};
}

inline butcher_tableau crank_nicolson() {
  return {2,
    {{0, 0}, {0.5, 0.5}},
    {0.5, 0.5},
    {0, 1}};
}

/// 2-stage, 3rd-order DIRK (Alexander, 1977)
inline butcher_tableau sdirk3() {
  constexpr double g = 0.4358665215084590;
  return {2,
    {{g, 0}, {1 - g, g}},
    {1 - g, g},
    {g, 1}};
}

/// 2-stage Gauss-Legendre (fully implicit, order 4)
inline butcher_tableau gauss_legendre_4() {
  constexpr double s = 0.28867513459481287;  // 1/(2*sqrt(3))
  return {2,
    {{0.25, 0.25 - s},
     {0.25 + s, 0.25}},
    {0.5, 0.5},
    {0.5 - s, 0.5 + s}};
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_BUTCHER_TABLEAU_H
