"""
Main Training Pipeline for Student Performance ML Project.
PROJECT TITLE: Student Performance Prediction and Learning Pattern Analysis Using Machine Learning with Applied Mathematics.
Executes reproducible end-to-end training:
1. Generates and splits dataset (Train/Val/Test) with zero leakage.
2. Fits and serializes preprocessors.
3. Trains Supervised Regression and Classification models.
4. Fits Unsupervised Clustering and Anomaly Detection models.
5. Trains PyTorch Neural Network.
6. Generates Model Cards and MLOps metadata.
Usage:
    python -m src.train
"""

import sys
from pathlib import Path
import joblib
import torch
import pandas as pd
import numpy as np

from src.utils.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_SIZE,
    FEATURE_NAMES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    REGRESSION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    PREPROCESSOR_PATH,
    CLUSTERING_MODEL_PATH,
    ANOMALY_MODEL_PATH,
    OPTIMIZER_MODEL_PATH,
    LABEL_TO_INT,
    METADATA_DIR,
)
from src.utils.helpers import set_seed
from src.data.loader import split_and_save_data, prepare_dataset
from src.preprocessing.pipeline import StudentFeaturePreprocessor
from src.supervised.regression import train_and_evaluate_regressors
from src.supervised.classification import train_and_evaluate_classifiers
from src.unsupervised.clustering import perform_kmeans_clustering
from src.unsupervised.anomaly import detect_anomalies_isolation_forest
from src.optimization.torch_models import StudentMLP
from src.optimization.optimizers_lab import get_optimizer
from src.mlops.model_card import generate_model_card
from src.mlops.versioning import track_experiment_run


def run_full_training_pipeline(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Executes the full automated training workflow."""
    print("=" * 70)
    print("🎓 STUDENT PERFORMANCE PREDICTION: ACADEMIC TRAINING PIPELINE")
    print("=" * 70)
    set_seed(seed)

    # 1. Dataset Generation & Splitting
    print(f"\n[1/6] Generating reproducible synthetic dataset (N={DEFAULT_SAMPLE_SIZE})...")
    raw_df = prepare_dataset(n_samples=DEFAULT_SAMPLE_SIZE, seed=seed, force_regenerate=True)
    train_df, val_df, test_df = split_and_save_data(raw_df, seed=seed)
    print(f"      Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

    # 2. Preprocessor Fitting & Serialization
    print("\n[2/6] Fitting and saving feature preprocessor...")
    preprocessor = StudentFeaturePreprocessor(feature_cols=FEATURE_NAMES)
    preprocessor.fit(train_df)
    preprocessor.save(PREPROCESSOR_PATH)
    print(f"      Preprocessor saved -> {PREPROCESSOR_PATH}")

    # 3. Supervised Regression
    print("\n[3/6] Training Supervised Regression Models...")
    y_train_reg = train_df[TARGET_REGRESSION]
    y_test_reg = test_df[TARGET_REGRESSION]
    reg_summary, reg_models = train_and_evaluate_regressors(
        train_df, y_train_reg, test_df, y_test_reg, seed=seed
    )
    print(reg_summary.to_string(index=False))

    best_reg_name = reg_summary.iloc[0]["Model"]
    best_reg_pipeline = reg_models[best_reg_name]["pipeline"]
    joblib.dump(best_reg_pipeline, REGRESSION_MODEL_PATH)
    print(f"      Best Regression Model: '{best_reg_name}' -> Saved to {REGRESSION_MODEL_PATH}")

    # 4. Supervised Classification
    print("\n[4/6] Training Supervised Classification Models...")
    y_train_clf = train_df[TARGET_CLASSIFICATION]
    y_test_clf = test_df[TARGET_CLASSIFICATION]
    clf_summary, clf_models = train_and_evaluate_classifiers(
        train_df, y_train_clf, test_df, y_test_clf, seed=seed
    )
    print(clf_summary.to_string(index=False))

    best_clf_name = clf_summary.iloc[0]["Model"]
    best_clf_pipeline = clf_models[best_clf_name]["pipeline"]
    joblib.dump(best_clf_pipeline, CLASSIFICATION_MODEL_PATH)
    print(f"      Best Classification Model: '{best_clf_name}' -> Saved to {CLASSIFICATION_MODEL_PATH}")

    # 5. Unsupervised Models (Clustering & Anomaly Detection)
    print("\n[5/6] Fitting Unsupervised Clustering and Anomaly Detection models...")
    kmeans_res = perform_kmeans_clustering(train_df, k=3, seed=seed)
    joblib.dump(kmeans_res["model"], CLUSTERING_MODEL_PATH)
    print(f"      K-Means (k=3, Silhouette={kmeans_res['silhouette_score']:.3f}) saved -> {CLUSTERING_MODEL_PATH}")

    iso_res = detect_anomalies_isolation_forest(train_df, contamination=0.05, seed=seed)
    joblib.dump(iso_res["model"], ANOMALY_MODEL_PATH)
    print(f"      Isolation Forest ({iso_res['n_anomalies']} anomalies flagged) saved -> {ANOMALY_MODEL_PATH}")

    # 6. PyTorch Neural Network Training
    print("\n[6/6] Training PyTorch StudentMLP Optimizer Model...")
    X_tr_sc = preprocessor.transform(train_df)
    y_vals = y_train_reg.values
    y_mean, y_std = float(np.mean(y_vals)), float(np.std(y_vals))
    y_norm = (y_vals - y_mean) / (y_std + 1e-12)

    y_tr_t = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)
    X_tr_t = torch.tensor(X_tr_sc, dtype=torch.float32)

    torch_model = StudentMLP(in_features=5, hidden_dim1=16, hidden_dim2=8, init_type="xavier")
    optimizer = torch.optim.AdamW(torch_model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    torch_model.train()
    for ep in range(80):
        optimizer.zero_grad()
        out = torch_model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()

    unnorm_rmse = float(np.sqrt(loss.item())) * y_std
    OPTIMIZER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": torch_model.state_dict(),
        "y_mean": y_mean,
        "y_std": y_std,
    }, OPTIMIZER_MODEL_PATH)
    print(f"      PyTorch StudentMLP (Normalized MSE: {loss.item():.4f}, Unnormalized RMSE: {unnorm_rmse:.2f}) saved -> {OPTIMIZER_MODEL_PATH}")

    # Generate Model Cards & Metadata
    print("\nGenerating Model Cards and MLOps Metadata...")
    generate_model_card(
        model_name="Student_Exam_Regressor",
        model_type=best_reg_name,
        metrics={
            "MAE": float(reg_models[best_reg_name]["mae"]),
            "RMSE": float(reg_models[best_reg_name]["rmse"]),
            "R2": float(reg_models[best_reg_name]["r2"]),
        },
        hyperparameters={"pipeline_steps": list(best_reg_pipeline.named_steps.keys())},
        training_samples=len(train_df),
        test_samples=len(test_df),
    )

    track_experiment_run(
        experiment_name="baseline_academic_run",
        parameters={"seed": seed, "best_reg": best_reg_name, "best_clf": best_clf_name},
        metrics={"reg_rmse": float(reg_models[best_reg_name]["rmse"]), "clf_f1": float(clf_models[best_clf_name]["f1"])},
        artifacts={"regression_model": str(REGRESSION_MODEL_PATH), "classification_model": str(CLASSIFICATION_MODEL_PATH)},
    )

    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   Artifacts generated in: {METADATA_DIR.parent}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_training_pipeline()
