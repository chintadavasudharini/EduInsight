"""
CO-2: Supervised Learning - Bias-Variance Tradeoff Module.
Implements empirical decomposition of expected test error:
    Expected MSE = Bias^2 + Variance + Irreducible Noise
Demonstrated across polynomial model complexities and regularization strengths.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.config import DEFAULT_RANDOM_SEED


def compute_bias_variance_decomposition(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    degrees: List[int] = [1, 2, 3, 4, 5],
    n_bootstraps: int = 35,
    sample_ratio: float = 0.70,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Empirically estimates Bias^2 and Variance across polynomial degrees:
        Bias^2 = 1/N_test sum_i ( E_D[f_hat_D(x_i)] - y_true(x_i) )^2
        Variance = 1/N_test sum_i E_D [ ( f_hat_D(x_i) - E_D[f_hat_D(x_i)] )^2 ]
        Total MSE = Bias^2 + Variance
    """
    rng = np.random.default_rng(seed)
    N_full = len(X_train_full)
    N_test = len(X_test)
    sample_size = int(N_full * sample_ratio)

    results = []

    for deg in degrees:
        predictions = np.zeros((n_bootstraps, N_test))

        for b in range(n_bootstraps):
            # Bootstrap sample from training data
            idx = rng.choice(N_full, size=sample_size, replace=True)
            X_b = X_train_full[idx]
            y_b = y_train_full[idx]

            # Fit pipeline
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ])
            model.fit(X_b, y_b)
            predictions[b, :] = model.predict(X_test)

        # Expected prediction at each test point across datasets
        expected_pred = np.mean(predictions, axis=0)  # shape (N_test,)

        # Bias^2
        bias_squared = float(np.mean((expected_pred - y_test) ** 2))

        # Variance
        variance = float(np.mean(np.var(predictions, axis=0)))

        # Total Mean Squared Error across all bootstrap fits
        total_mse = bias_squared + variance

        results.append({
            "Degree": deg,
            "Bias_Squared": round(bias_squared, 4),
            "Variance": round(variance, 4),
            "Total_Expected_MSE": round(total_mse, 4),
        })

    summary_df = pd.DataFrame(results)

    explanation = (
        "Underfitting (High Bias) occurs at low degree (Deg 1): the model is too simple to capture "
        "the underlying quadratic/interaction dynamics. As model complexity increases, Bias^2 drops. "
        "However, at higher degrees, Variance escalates as the model overfits to idiosyncrasies of specific "
        "training draws. The sweet spot minimizes total expected test error."
    )

    return {
        "summary_table": summary_df,
        "degrees": degrees,
        "bias_squared": summary_df["Bias_Squared"].tolist(),
        "variance": summary_df["Variance"].tolist(),
        "total_mse": summary_df["Total_Expected_MSE"].tolist(),
        "explanation": explanation,
    }
