"""
CO-1: Mathematical Foundations for ML - Calculus and Optimization Module.
Implements:
- Partial derivatives and Chain Rule
- Gradient vector and Hessian matrix computation
- Convexity verification via Hessian eigenvalues and chord inequality
- Interactive Gradient Descent simulator (convergence, oscillation, divergence)
- Constrained Optimization: Lagrange Multipliers and KKT Conditions verification
"""

from typing import Dict, Any, List, Tuple, Callable
import numpy as np
from scipy.optimize import minimize


def compute_gradient_and_hessian(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lambda_reg: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes Gradient and Hessian for Mean Squared Error loss:
        L(w) = 1/(2N) ||X w - y||^2 + (lambda_reg / 2) ||w||^2

    Gradient:
        grad = 1/N * X^T (X w - y) + lambda_reg * w

    Hessian:
        H = 1/N * X^T X + lambda_reg * I
    """
    w = np.asarray(w, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = X.shape

    # Loss computation
    residuals = np.dot(X, w) - y
    loss = (0.5 / N) * np.sum(residuals ** 2) + 0.5 * lambda_reg * np.sum(w ** 2)

    # Gradient: shape (D,)
    gradient = (1.0 / N) * np.dot(X.T, residuals) + lambda_reg * w

    # Hessian: shape (D, D)
    hessian = (1.0 / N) * np.dot(X.T, X) + lambda_reg * np.eye(D)

    # Check positive semi-definiteness via eigenvalues of Hessian
    eigvals = np.linalg.eigvalsh(hessian)
    min_eigval = float(np.min(eigvals))
    is_positive_definite = bool(min_eigval > 1e-9)
    is_strictly_convex = is_positive_definite

    return {
        "loss": float(loss),
        "weights": w.tolist(),
        "gradient": gradient.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "hessian": hessian.tolist(),
        "hessian_eigenvalues": eigvals.tolist(),
        "min_eigenvalue": min_eigval,
        "is_positive_definite": is_positive_definite,
        "is_strictly_convex": is_strictly_convex,
    }


def verify_convexity_chord_inequality(
    func: Callable[[np.ndarray], float],
    x1: np.ndarray,
    x2: np.ndarray,
    num_points: int = 15,
) -> Dict[str, Any]:
    """
    Numerically verifies Jensen's convexity definition:
        f(t * x1 + (1 - t) * x2) <= t * f(x1) + (1 - t) * f(x2)  for t in [0, 1]
    """
    t_values = np.linspace(0.0, 1.0, num_points)
    f_x1 = func(x1)
    f_x2 = func(x2)

    f_convex_combo = []
    chord_linear_combo = []
    satisfied_list = []

    for t in t_values:
        point = t * x1 + (1.0 - t) * x2
        f_val = float(func(point))
        chord_val = float(t * f_x1 + (1.0 - t) * f_x2)
        f_convex_combo.append(f_val)
        chord_linear_combo.append(chord_val)
        satisfied_list.append(bool(f_val <= chord_val + 1e-7))

    return {
        "t_values": t_values.tolist(),
        "f_convex_combo": f_convex_combo,
        "chord_linear_combo": chord_linear_combo,
        "all_satisfied": bool(all(satisfied_list)),
        "max_slack": float(np.max(np.array(chord_linear_combo) - np.array(f_convex_combo))),
    }


def simulate_gradient_descent_1d_or_2d(
    objective_type: str = "quadratic",
    learning_rate: float = 0.1,
    n_iterations: int = 40,
    initial_point: Tuple[float, float] = (4.0, 4.0),
) -> Dict[str, Any]:
    """
    Simulates gradient descent trajectory on standard academic optimization surfaces:
    1. Quadratic: f(x, y) = 0.5 * (x^2 + 10 * y^2) (anisotropic valley)
    2. Spherical: f(x, y) = x^2 + y^2
    Demonstrates convergence, overshooting, or oscillation based on learning rate.
    """
    x, y = float(initial_point[0]), float(initial_point[1])
    trajectory = [[x, y]]
    losses = []

    def get_val_and_grad(px: float, py: float):
        if objective_type == "quadratic":
            val = 0.5 * (px ** 2 + 10.0 * (py ** 2))
            gx = px
            gy = 10.0 * py
        else:  # spherical
            val = px ** 2 + py ** 2
            gx = 2.0 * px
            gy = 2.0 * py
        return val, gx, gy

    for _ in range(n_iterations):
        val, gx, gy = get_val_and_grad(x, y)
        losses.append(float(val))

        # Check divergence to avoid overflow
        if abs(x) > 1e4 or abs(y) > 1e4 or np.isnan(val):
            break

        x = x - learning_rate * gx
        y = y - learning_rate * gy
        trajectory.append([float(x), float(y)])

    final_val, _, _ = get_val_and_grad(x, y)
    losses.append(float(final_val))

    status = "converged" if losses[-1] < 1e-3 else ("diverged" if losses[-1] > 1e3 else "iterating")

    return {
        "learning_rate": learning_rate,
        "objective_type": objective_type,
        "trajectory": trajectory,
        "losses": losses,
        "final_point": [float(x), float(y)],
        "final_loss": float(losses[-1]),
        "status": status,
    }


def demonstrate_lagrange_and_kkt() -> Dict[str, Any]:
    """
    Demonstrates Constrained Optimization, Lagrange Multipliers, and KKT Conditions:
    Problem:
        Minimize f(x1, x2) = (x1 - 8)^2 + (x2 - 6)^2
        Subject to inequality constraint g(x1, x2) = 12 - (x1 + x2) <= 0
        (i.e., Total academic dedication x1 + x2 >= 12)

    KKT Conditions:
    1. Stationarity: grad f(x*) + lambda * grad g(x*) = 0
    2. Primal Feasibility: g(x*) <= 0
    3. Dual Feasibility: lambda >= 0
    4. Complementary Slackness: lambda * g(x*) = 0
    """
    # Objective function
    def obj(x):
        return (x[0] - 8.0) ** 2 + (x[1] - 6.0) ** 2

    def grad_obj(x):
        return np.array([2.0 * (x[0] - 8.0), 2.0 * (x[1] - 6.0)])

    # Constraint g(x) = 12 - (x1 + x2) <= 0 -> in scipy 'ineq' means constr >= 0
    # so we formulate x1 + x2 - 12 >= 0
    constraints = {"type": "ineq", "fun": lambda x: x[0] + x[1] - 12.0}

    # Initial guess
    x0 = np.array([2.0, 2.0])
    res = minimize(obj, x0, method="SLSQP", constraints=constraints)

    x_opt = res.x
    f_opt = float(res.fun)

    # Check unconstrained minimizer
    unconstrained_min = np.array([8.0, 6.0])
    sum_unconstrained = np.sum(unconstrained_min)  # 14 >= 12, constraint is inactive!

    # Let's formulate a case where constraint is active: x1 + x2 >= 16
    constraints_active = {"type": "ineq", "fun": lambda x: x[0] + x[1] - 16.0}
    res_active = minimize(obj, x0, method="SLSQP", constraints=constraints_active)
    x_opt_active = res_active.x
    f_opt_active = float(res_active.fun)

    # Constraint value g(x*) = 16 - (x1* + x2*)
    g_val = float(16.0 - (x_opt_active[0] + x_opt_active[1]))

    # Gradient of objective at optimum
    grad_at_opt = grad_obj(x_opt_active)
    # Gradient of constraint g(x) = 16 - x1 - x2 is [-1, -1]
    # By stationarity: grad_f + lambda * [-1, -1] = 0 => lambda = grad_f_1 = grad_f_2
    lambda_val = float(grad_at_opt[0])  # because grad_g = -1

    complementary_slackness = float(lambda_val * g_val)
    primal_feasible = bool(g_val <= 1e-5)
    dual_feasible = bool(lambda_val >= -1e-5)
    slackness_satisfied = bool(abs(complementary_slackness) < 1e-4)

    return {
        "problem_description": "Min (x1-8)^2 + (x2-6)^2 s.t. x1 + x2 >= 16",
        "optimal_x": x_opt_active.tolist(),
        "optimal_cost": f_opt_active,
        "constraint_g_val": g_val,
        "lagrange_multiplier_lambda": lambda_val,
        "grad_f_at_opt": grad_at_opt.tolist(),
        "kkt_conditions": {
            "primal_feasibility_g_le_0": primal_feasible,
            "dual_feasibility_lambda_ge_0": dual_feasible,
            "complementary_slackness_lambda_g_eq_0": slackness_satisfied,
            "slackness_product": complementary_slackness,
        },
        "all_kkt_satisfied": primal_feasible and dual_feasible and slackness_satisfied,
    }
