"""
CO-2: Supervised Learning - Classification Module.
Implements:
- Logistic Regression (multinomial, L2)
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (RBF, Linear, Polynomial kernels)
- XGBoost Classifier (if available)
- LightGBM / CatBoost (if installed)
Includes leak-free pipeline preprocessing and class-weight balancing.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

from src.utils.config import (
    FEATURE_NAMES,
    TARGET_CLASSIFICATION,
    DEFAULT_RANDOM_SEED,
    CATEGORY_LABELS,
    LABEL_TO_INT,
)


def get_classification_models(seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, Pipeline]:
    """
    Constructs dictionary of classification Pipelines.
    Each pipeline encapsulates feature scaling and the estimator.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            ))
        ]),
        "Naive Bayes (Gaussian)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB())
        ]),
        "K-Nearest Neighbors (KNN)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
        "Decision Tree": Pipeline([
            ("model", DecisionTreeClassifier(
                max_depth=5,
                class_weight="balanced",
                random_state=seed,
            ))
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ))
        ]),
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ))
        ]),
        "SVM (Linear Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ))
        ]),
        "SVM (Polynomial Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="poly",
                degree=3,
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ))
        ]),
    }

    # Optional XGBoost integration
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=seed,
                eval_metric="mlogloss",
                verbosity=0,
            ))
        ])
    except ImportError:
        pass

    # Optional LightGBM integration
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=seed,
                verbose=-1,
            ))
        ])
    except ImportError:
        pass

    # Optional CatBoost integration
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", CatBoostClassifier(
                iterations=100,
                depth=4,
                random_seed=seed,
                verbose=0,
            ))
        ])
    except ImportError:
        pass

    return models


def train_and_evaluate_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits classification models, measures standard metrics, and computes confusion matrices.
    """
    models = get_classification_models(seed=seed)
    results = []
    fitted_models = {}

    # Map string labels to numeric for algorithms like XGBoost
    y_train_num = y_train.map(LABEL_TO_INT).values
    y_test_num = y_test.map(LABEL_TO_INT).values

    num_classes = len(CATEGORY_LABELS)

    for name, pipeline in models.items():
        # Fit on training data
        pipeline.fit(X_train[FEATURE_NAMES], y_train_num)
        y_pred_num = pipeline.predict(X_test[FEATURE_NAMES])

        # Probabilities for ROC-AUC
        y_prob = None
        roc_auc = float("nan")
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test[FEATURE_NAMES])
                roc_auc = roc_auc_score(
                    y_test_num, y_prob, multi_class="ovr", average="macro"
                )
            except Exception:
                roc_auc = float("nan")

        acc = accuracy_score(y_test_num, y_pred_num)
        prec = precision_score(y_test_num, y_pred_num, average="macro", zero_division=0)
        rec = recall_score(y_test_num, y_pred_num, average="macro", zero_division=0)
        f1 = f1_score(y_test_num, y_pred_num, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_test_num, y_pred_num)
        cm = confusion_matrix(y_test_num, y_pred_num, labels=list(range(num_classes)))

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (Macro)": round(prec, 4),
            "Recall (Macro)": round(rec, 4),
            "F1-Score (Macro)": round(f1, 4),
            "ROC-AUC (OVR)": round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
            "MCC": round(mcc, 4),
        })

        fitted_models[name] = {
            "pipeline": pipeline,
            "y_pred": y_pred_num,
            "y_prob": y_prob,
            "confusion_matrix": cm,
            "accuracy": acc,
            "f1": f1,
            "mcc": mcc,
        }

    summary_df = pd.DataFrame(results).sort_values(by="F1-Score (Macro)", ascending=False).reset_index(drop=True)
    return summary_df, fitted_models
