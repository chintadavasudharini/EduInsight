"""
Page 9: Applied Mathematics Lab (CO-1: Mathematical Foundations for ML).
Dedicated mathematical laboratory demonstrating:
1. Linear Algebra: Vectors, Matrices, Covariance Eigendecomposition, and SVD.
2. Probability & Statistics: Summary statistics, Expectation identity, Bayes Theorem,
   Confidence intervals, Hypothesis testing, and MLE vs MAP estimation.
3. Calculus & Optimization: Gradients & Hessians, Convexity verification, Interactive
   Gradient Descent simulator, and Lagrange Multipliers / KKT conditions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.utils.config import FEATURE_NAMES, TARGET_REGRESSION
from src.data.loader import prepare_dataset
from src.mathematics.linear_algebra import (
    compute_vector_operations,
    compute_matrix_properties,
    compute_covariance_and_eigendecomposition,
    compute_svd,
)
from src.mathematics.probability_stats import (
    compute_descriptive_statistics,
    compute_expectation_and_variance_proof,
    compute_bayes_theorem_student_case,
    compute_confidence_intervals_and_ttest,
    compute_mle_vs_map,
)
from src.mathematics.calculus_optimization import (
    compute_gradient_and_hessian,
    verify_convexity_chord_inequality,
    simulate_gradient_descent_1d_or_2d,
    demonstrate_lagrange_and_kkt,
)


def render_page():
    st.markdown('<div class="main-header">📐 Applied Mathematics Laboratory (CO-1)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Working mathematical demonstrations using NumPy & SciPy derived directly from student records.</div>',
        unsafe_allow_html=True,
    )

    df = prepare_dataset()
    X_mat = df[FEATURE_NAMES].values
    y_vec = df[TARGET_REGRESSION].values

    tab_linalg, tab_prob_stats, tab_calc_opt = st.tabs([
        "🔢 Module 1: Linear Algebra & Matrix Decomposition",
        "🎲 Module 2: Probability, Bayes, Inference & MLE/MAP",
        "📉 Module 3: Calculus, Convexity & Constrained Optimization",
    ])

    # -------------------------------------------------------------
    # MODULE 1: LINEAR ALGEBRA
    # -------------------------------------------------------------
    with tab_linalg:
        st.subheader("1. Vector Geometry & Operations on Student Profiles")
        st.markdown(
            "Each student $i$ is represented as a feature vector $\\mathbf{x}_i \\in \\mathbb{R}^5$ in the academic feature space."
        )

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            idx1 = st.selectbox("Select Student 1:", list(range(10)), index=0, format_func=lambda i: f"Student {i+1} ({df.iloc[i]['student_id']})")
        with col_v2:
            idx2 = st.selectbox("Select Student 2:", list(range(10)), index=1, format_func=lambda i: f"Student {i+1} ({df.iloc[i]['student_id']})")

        vec1 = X_mat[idx1]
        vec2 = X_mat[idx2]

        vec_metrics = compute_vector_operations(vec1, vec2)

        st.write(f"**$\\mathbf{v}_1$:** `{vec1.round(2).tolist()}`")
        st.write(f"**$\\mathbf{v}_2$:** `{vec2.round(2).tolist()}`")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Dot Product (v1 · v2)", f"{vec_metrics['dot_product']:.1f}")
        with c2:
            st.metric("Cosine Similarity", f"{vec_metrics['cosine_similarity']:.4f}")
        with c3:
            st.metric("Euclidean Distance (L2)", f"{vec_metrics['euclidean_distance']:.2f}")
        with c4:
            st.metric("Manhattan Norm ||v1||₁", f"{vec_metrics['norm_l1_v1']:.1f}")

        st.markdown("---")
        st.subheader("2. Covariance Matrix Eigendecomposition")
        st.markdown(
            "$$\\mathbf{\\Sigma} = \\frac{1}{N-1} (\\mathbf{X} - \\mathbf{\\mu})^T (\\mathbf{X} - \\mathbf{\\mu}), \\quad "
            "\\mathbf{\\Sigma} \\mathbf{v}_i = \\lambda_i \\mathbf{v}_i$$"
        )

        eig_res = compute_covariance_and_eigendecomposition(X_mat, FEATURE_NAMES)

        cov_df = pd.DataFrame(eig_res["covariance_matrix"], index=FEATURE_NAMES, columns=FEATURE_NAMES).round(3)
        st.markdown("##### Sample Covariance Matrix $\\mathbf{\\Sigma} \\in \\mathbb{R}^{5 \\times 5}$")
        st.dataframe(cov_df, use_container_width=True)

        st.markdown("##### Eigenvalues $\\lambda_i$ and Explained Variance Ratios")
        evals_df = pd.DataFrame({
            "Eigenvalue (λ)": np.round(eig_res["eigenvalues"], 4),
            "Variance Ratio": [f"{r*100:.2f}%" for r in eig_res["explained_variance_ratio"]],
            "Cumulative Variance": [f"{r*100:.2f}%" for r in eig_res["cumulative_variance_ratio"]],
        }, index=[f"Component {i+1}" for i in range(5)])
        st.dataframe(evals_df, use_container_width=True)

        st.markdown("---")
        st.subheader("3. Singular Value Decomposition (SVD)")
        st.markdown(
            "$$\\mathbf{X} = \\mathbf{U} \\mathbf{\\Sigma}_{\\text{svd}} \\mathbf{V}^T, \\quad "
            "\\mathbf{X}_k = \\sum_{i=1}^k \\sigma_i \\mathbf{u}_i \\mathbf{v}_i^T$$"
        )

        k_svd = st.slider("Rank-k Truncation:", 1, 5, 2)
        svd_res = compute_svd(X_mat, k=k_svd)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Top Singular Values", f"{np.round(svd_res['singular_values_top_k'], 2).tolist()}")
        with m2:
            st.metric("Frobenius Reconstruction Error", f"{svd_res['frobenius_error']:.2f}")
        with m3:
            st.metric("Relative Information Preserved", f"{(1.0 - svd_res['relative_error'])*100:.1f}%")

    # -------------------------------------------------------------
    # MODULE 2: PROBABILITY & STATISTICS
    # -------------------------------------------------------------
    with tab_prob_stats:
        st.subheader("1. Descriptive Statistics & Bessel's Correction")
        st.markdown(
            "Sample Variance: $s^2 = \\frac{1}{n-1} \\sum_{i=1}^n (x_i - \\bar{x})^2$ vs "
            "Population Variance: $\\sigma^2 = \\frac{1}{n} \\sum_{i=1}^n (x_i - \\mu)^2$"
        )

        stat_feat = st.selectbox("Select Feature for Statistical Analysis:", FEATURE_NAMES)
        stats_dict = compute_descriptive_statistics(df[stat_feat].values)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Sample Mean", f"{stats_dict['mean']:.3f}")
        with c2:
            st.metric("Sample Variance (ddof=1)", f"{stats_dict['sample_variance']:.3f}")
        with c3:
            st.metric("Population Variance (ddof=0)", f"{stats_dict['population_variance']:.3f}")
        with c4:
            st.metric("Bessel's Correction Ratio", f"{stats_dict['sample_variance'] / stats_dict['population_variance']:.4f}")

        st.markdown("---")
        st.subheader("2. Expectation & Variance Identity Proof")
        st.markdown(
            "Proof that: $\\text{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2$"
        )
        proof_res = compute_expectation_and_variance_proof(df[stat_feat].values[:100])

        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("E[X]", f"{proof_res['E_X']:.3f}")
            st.metric("(E[X])²", f"{proof_res['E_X_squared']:.3f}")
        with p2:
            st.metric("E[X²]", f"{proof_res['E_X2']:.3f}")
            st.metric("E[X²] - (E[X])²", f"{proof_res['Var_identity']:.3f}")
        with p3:
            st.metric("Direct Definition Var(X)", f"{proof_res['Var_direct']:.3f}")
            st.success(f"Identity Numerically Verified: **{proof_res['identity_verified']}**")

        st.markdown("---")
        st.subheader("3. Bayes' Theorem: Student Performance Case Study")
        st.markdown(
            "$$P(\\text{Excellent} | \\text{High Study}) = \\frac{P(\\text{High Study} | \\text{Excellent}) \\cdot P(\\text{Excellent})}{P(\\text{High Study})}$$"
        )

        study_thresh = st.slider("High Study Cutoff (Hours):", 5.0, 8.5, 6.5, 0.5)
        score_thresh = st.slider("Excellent Grade Cutoff (Marks):", 70.0, 90.0, 80.0, 2.0)

        bayes_res = compute_bayes_theorem_student_case(
            df["study_hours_per_day"].values,
            df[TARGET_REGRESSION].values,
            high_study_threshold=study_thresh,
            excellent_threshold=score_thresh,
        )

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Prior P(Excellent)", f"{bayes_res['prior_P_A']*100:.1f}%")
        with b2:
            st.metric("Marginal P(High Study)", f"{bayes_res['marginal_P_B']*100:.1f}%")
        with b3:
            st.metric("Likelihood P(Study | Excel)", f"{bayes_res['likelihood_P_B_given_A']*100:.1f}%")
        with b4:
            st.metric("Posterior P(Excel | Study)", f"{bayes_res['posterior_bayes_P_A_given_B']*100:.1f}%")

        st.caption(f"Bayes calculation matches empirical joint conditional exactly: **{bayes_res['formula_matched']}**")

        st.markdown("---")
        st.subheader("4. Maximum Likelihood Estimation (MLE) vs Maximum A Posteriori (MAP)")
        st.markdown(
            "Gaussian Likelihood: $X_i \\sim \\mathcal{N}(\\mu, \\sigma^2)$, Prior: $\\mu \\sim \\mathcal{N}(\\mu_0, \\sigma_0^2)$. "
            "MAP incorporates shrinkage towards prior belief $\\mu_0$."
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            p_mean = st.slider("Prior Mean (μ₀):", 50.0, 80.0, 65.0, 1.0)
        with col_m2:
            p_std = st.slider("Prior Uncertainty (σ₀):", 1.0, 15.0, 4.0, 0.5)

        mle_map = compute_mle_vs_map(df[TARGET_REGRESSION].values[:100], prior_mean=p_mean, prior_std=p_std)

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("MLE Mean (Data Only)", f"{mle_map['sample_mean_MLE']:.2f}")
        with m_c2:
            st.metric("Prior Mean", f"{mle_map['prior_mean']:.2f}")
        with m_c3:
            st.metric("MAP Mean (Regularized)", f"{mle_map['MAP_mean']:.2f}")
        with m_c4:
            st.metric("Prior Weighting", f"{mle_map['prior_weight']*100:.1f}%")

        st.info(f"Shrinkage Effect: MAP estimate is shifted by {mle_map['shrinkage_effect']:.2f} points toward prior.")

    # -------------------------------------------------------------
    # MODULE 3: CALCULUS, CONVEXITY & OPTIMIZATION
    # -------------------------------------------------------------
    with tab_calc_opt:
        st.subheader("1. Gradient Vector & Hessian Matrix for Loss Surface")
        st.markdown(
            "Loss function: $L(\\mathbf{w}) = \\frac{1}{2N} \\|\\mathbf{X}\\mathbf{w} - \\mathbf{y}\\|^2 + \\frac{\\lambda}{2} \\|\\mathbf{w}\\|^2$\n\n"
            "Gradient: $\\nabla L(\\mathbf{w}) = \\frac{1}{N}\\mathbf{X}^T(\\mathbf{X}\\mathbf{w} - \\mathbf{y}) + \\lambda \\mathbf{w}$\n\n"
            "Hessian: $\\mathbf{H} = \\frac{1}{N}\\mathbf{X}^T\\mathbf{X} + \\lambda \\mathbf{I}$"
        )

        reg_lambda = st.slider("Ridge Regularization Parameter (lambda):", 0.0, 5.0, 0.1, 0.1)
        w_sample = np.ones(5) * 2.0

        grad_hess = compute_gradient_and_hessian(w_sample, X_mat[:200], y_vec[:200], lambda_reg=reg_lambda)

        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("Loss Value L(w)", f"{grad_hess['loss']:.2f}")
        with h2:
            st.metric("Gradient Norm ||∇L||", f"{grad_hess['gradient_norm']:.2f}")
        with h3:
            st.metric("Min Hessian Eigenvalue", f"{grad_hess['min_eigenvalue']:.4f}")

        st.success(f"Hessian is Positive Definite: **{grad_hess['is_positive_definite']}** $\\implies$ Loss surface is **Strictly Convex**.")

        st.markdown("---")
        st.subheader("2. Convexity Verification via Jensen's Chord Inequality")
        st.markdown(
            "Function $f(\\mathbf{x})$ is convex if for all $\\mathbf{x}_1, \\mathbf{x}_2$ and $t \\in [0, 1]$: "
            "$$f(t \\mathbf{x}_1 + (1-t) \\mathbf{x}_2) \\le t f(\\mathbf{x}_1) + (1-t) f(\\mathbf{x}_2)$$"
        )

        def test_loss(w):
            res = np.dot(X_mat[:100], w) - y_vec[:100]
            return 0.5 * np.mean(res**2)

        chord_res = verify_convexity_chord_inequality(test_loss, np.zeros(5), np.ones(5)*3.0)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(chord_res["t_values"], chord_res["f_convex_combo"], "b-", label="f(t x₁ + (1-t) x₂) [Function]", linewidth=2)
        ax.plot(chord_res["t_values"], chord_res["chord_linear_combo"], "r--", label="t f(x₁) + (1-t) f(x₂) [Secant Chord]", linewidth=2)
        ax.fill_between(chord_res["t_values"], chord_res["f_convex_combo"], chord_res["chord_linear_combo"], color="gray", alpha=0.2)
        ax.set_xlabel("Interpolation Parameter t")
        ax.set_ylabel("Loss Value")
        ax.set_title("Secant Chord Lies Everywhere Above Convex Curve", fontsize=10, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(f"Jensen's chord inequality satisfied across all test points: **{chord_res['all_satisfied']}**")

        st.markdown("---")
        st.subheader("3. Interactive Gradient Descent Step Simulator")
        st.markdown("Observe the effect of the learning rate parameter $\\eta$ on convergence:")

        lr_sim = st.slider("Learning Rate (eta):", 0.01, 0.35, 0.08, 0.01)
        gd_sim = simulate_gradient_descent_1d_or_2d(
            objective_type="quadratic",
            learning_rate=lr_sim,
            n_iterations=30,
            initial_point=(4.0, 3.5),
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

        # Trajectory in 2D contour
        traj = np.array(gd_sim["trajectory"])
        x_grid = np.linspace(-5, 5, 100)
        y_grid = np.linspace(-5, 5, 100)
        Xg, Yg = np.meshgrid(x_grid, y_grid)
        Zg = 0.5 * (Xg**2 + 10.0 * Yg**2)

        ax1.contour(Xg, Yg, Zg, levels=20, cmap="viridis")
        ax1.plot(traj[:, 0], traj[:, 1], "ro-", markersize=4, label="GD Steps")
        ax1.plot(0, 0, "g*", markersize=12, label="Global Optimum")
        ax1.set_title(f"Optimization Trajectory (η={lr_sim})", fontsize=10, fontweight="bold")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Loss curve
        ax2.plot(gd_sim["losses"], "m^-", linewidth=1.5)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Objective Loss")
        ax2.set_title(f"Loss Convergence Curve ({gd_sim['status'].upper()})", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.subheader("4. Constrained Optimization: Lagrange Multipliers & KKT Conditions")
        st.markdown(
            "Problem: Minimize deviation from ideal schedule $\\min (x_1 - 8)^2 + (x_2 - 6)^2$ "
            "subject to dedication requirement $g(x_1, x_2) = 16 - (x_1 + x_2) \\le 0$."
        )

        kkt_res = demonstrate_lagrange_and_kkt()

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Optimal Solution (x₁*, x₂*)", f"{np.round(kkt_res['optimal_x'], 2).tolist()}")
        with k2:
            st.metric("Lagrange Multiplier (λ*)", f"{kkt_res['lagrange_multiplier_lambda']:.4f}")
        with k3:
            st.metric("Constraint Slackness (λ · g)", f"{kkt_res['kkt_conditions']['slackness_product']:.5f}")

        st.success(
            f"✅ **KKT Conditions Verification:**\n\n"
            f"- Primal Feasibility ($g(x^*) \\le 0$): `{kkt_res['kkt_conditions']['primal_feasibility_g_le_0']}`\n"
            f"- Dual Feasibility ($\\lambda^* \\ge 0$): `{kkt_res['kkt_conditions']['dual_feasibility_lambda_ge_0']}`\n"
            f"- Complementary Slackness ($\\lambda^* g(x^*) = 0$): `{kkt_res['kkt_conditions']['complementary_slackness_lambda_g_eq_0']}`"
        )


if __name__ == "__main__":
    render_page()
