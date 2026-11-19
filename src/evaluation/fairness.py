"""
CO-5: Model Evaluation and Selection - Algorithmic Fairness Analysis Module.
Implements:
- Demographic Parity (Statistical Parity Ratio & Difference)
- Equal Opportunity (True Positive Rate Parity)
- Equalized Odds (TPR and FPR parity across groups)

IMPORTANT ACADEMIC DISCLAIMER:
Fairness evaluation here is demonstrated strictly using synthetic group partitions
(e.g., Campus A vs Campus B) for educational purposes.
Do not infer real-world fairness conclusions from synthetic partitions, and
ensure sensitive attributes are managed ethically in practical deployments.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def compute_fairness_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_true_binary: str,
    y_pred_binary: str,
) -> Dict[str, Any]:
    """
    Computes standard fairness criteria across groups:
    1. Selection Rate: P(Y_hat = 1 | Group = g)
    2. Demographic Parity Difference: |SR_g1 - SR_g2|
    3. Demographic Parity Ratio (Disparate Impact): min(SR) / max(SR)
    4. True Positive Rate (Recall): P(Y_hat = 1 | Y = 1, Group = g)
    5. Equal Opportunity Difference: |TPR_g1 - TPR_g2|
    6. False Positive Rate: P(Y_hat = 1 | Y = 0, Group = g)
    7. Equalized Odds Difference: max(|TPR_g1 - TPR_g2|, |FPR_g1 - FPR_g2|)
    """
    groups = df[group_col].unique()
    group_stats = {}

    for g in groups:
        sub = df[df[group_col] == g]
        y_t = sub[y_true_binary].values
        y_p = sub[y_pred_binary].values

        n = len(sub)
        selection_rate = float(np.mean(y_p == 1)) if n > 0 else 0.0

        positives = (y_t == 1)
        negatives = (y_t == 0)

        tpr = float(np.mean(y_p[positives] == 1)) if np.sum(positives) > 0 else 0.0
        fpr = float(np.mean(y_p[negatives] == 1)) if np.sum(negatives) > 0 else 0.0

        group_stats[str(g)] = {
            "n_samples": n,
            "selection_rate": round(selection_rate, 4),
            "tpr_recall": round(tpr, 4),
            "fpr": round(fpr, 4),
        }

    # Compare between the two primary groups
    if len(groups) >= 2:
        g1, g2 = str(groups[0]), str(groups[1])
        sr1 = group_stats[g1]["selection_rate"]
        sr2 = group_stats[g2]["selection_rate"]
        dp_diff = abs(sr1 - sr2)
        dp_ratio = min(sr1, sr2) / (max(sr1, sr2) + 1e-9)

        tpr1 = group_stats[g1]["tpr_recall"]
        tpr2 = group_stats[g2]["tpr_recall"]
        eq_opp_diff = abs(tpr1 - tpr2)

        fpr1 = group_stats[g1]["fpr"]
        fpr2 = group_stats[g2]["fpr"]
        eq_odds_diff = max(eq_opp_diff, abs(fpr1 - fpr2))
    else:
        dp_diff = 0.0
        dp_ratio = 1.0
        eq_opp_diff = 0.0
        eq_odds_diff = 0.0

    # 80% rule for disparate impact (EEOC guideline)
    four_fifths_rule_met = bool(dp_ratio >= 0.80)

    return {
        "group_stats": group_stats,
        "demographic_parity_difference": round(dp_diff, 4),
        "demographic_parity_ratio": round(dp_ratio, 4),
        "four_fifths_rule_met": four_fifths_rule_met,
        "equal_opportunity_difference": round(eq_opp_diff, 4),
        "equalized_odds_difference": round(eq_odds_diff, 4),
    }
