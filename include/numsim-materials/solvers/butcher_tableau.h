#ifndef BUTCHER_TABLEAU_H
#define BUTCHER_TABLEAU_H

#include <Eigen/Dense>

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
  Eigen::MatrixXd a;
  Eigen::VectorXd b;
  Eigen::VectorXd c;

  int stages() const { return static_cast<int>(b.size()); }

  bool is_explicit() const {
    const int s = stages();
    for (int i = 0; i < s; ++i)
      for (int j = i; j < s; ++j)
        if (a(i, j) != 0.0) return false;
    return true;
  }

  bool is_dirk() const {
    const int s = stages();
    for (int i = 0; i < s; ++i)
      for (int j = i + 1; j < s; ++j)
        if (a(i, j) != 0.0) return false;
    return true;
  }
};

// --- Factory functions ---

inline butcher_tableau forward_euler() {
  butcher_tableau t;
  t.a = Eigen::MatrixXd::Zero(1, 1);
  t.b = Eigen::VectorXd{{1}};
  t.c = Eigen::VectorXd{{0}};
  return t;
}

inline butcher_tableau explicit_midpoint() {
  butcher_tableau t;
  t.a = Eigen::MatrixXd::Zero(2, 2);
  t.a(1, 0) = 0.5;
  t.b = Eigen::VectorXd{{0, 1}};
  t.c = Eigen::VectorXd{{0, 0.5}};
  return t;
}

inline butcher_tableau rk4() {
  butcher_tableau t;
  t.a = Eigen::MatrixXd::Zero(4, 4);
  t.a(1, 0) = 0.5;
  t.a(2, 1) = 0.5;
  t.a(3, 2) = 1.0;
  t.b = Eigen::VectorXd{{1.0/6, 1.0/3, 1.0/3, 1.0/6}};
  t.c = Eigen::VectorXd{{0, 0.5, 0.5, 1}};
  return t;
}

inline butcher_tableau implicit_euler() {
  butcher_tableau t;
  t.a = Eigen::MatrixXd{{1.0}};
  t.b = Eigen::VectorXd{{1}};
  t.c = Eigen::VectorXd{{1}};
  return t;
}

inline butcher_tableau implicit_midpoint() {
  butcher_tableau t;
  t.a = Eigen::MatrixXd{{0.5}};
  t.b = Eigen::VectorXd{{1}};
  t.c = Eigen::VectorXd{{0.5}};
  return t;
}

inline butcher_tableau crank_nicolson() {
  butcher_tableau t;
  t.a = Eigen::MatrixXd::Zero(2, 2);
  t.a(1, 0) = 0.5;
  t.a(1, 1) = 0.5;
  t.b = Eigen::VectorXd{{0.5, 0.5}};
  t.c = Eigen::VectorXd{{0, 1}};
  return t;
}

/// 2-stage, 3rd-order DIRK (Alexander, 1977)
inline butcher_tableau sdirk3() {
  constexpr double g = 0.4358665215084590;
  butcher_tableau t;
  t.a = Eigen::MatrixXd::Zero(2, 2);
  t.a(0, 0) = g;
  t.a(1, 0) = 1 - g;
  t.a(1, 1) = g;
  t.b = Eigen::VectorXd{{1 - g, g}};
  t.c = Eigen::VectorXd{{g, 1}};
  return t;
}

/// 2-stage Gauss-Legendre (fully implicit, order 4)
inline butcher_tableau gauss_legendre_4() {
  constexpr double s = 0.28867513459481287;  // 1/(2*sqrt(3))
  butcher_tableau t;
  t.a = Eigen::MatrixXd(2, 2);
  t.a(0, 0) = 0.25;       t.a(0, 1) = 0.25 - s;
  t.a(1, 0) = 0.25 + s;   t.a(1, 1) = 0.25;
  t.b = Eigen::VectorXd{{0.5, 0.5}};
  t.c = Eigen::VectorXd{{0.5 - s, 0.5 + s}};
  return t;
}

} // namespace numsim::materials

#endif // BUTCHER_TABLEAU_H
