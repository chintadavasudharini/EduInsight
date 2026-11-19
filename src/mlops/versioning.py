"""
CO-6: Production ML Engineering / MLOps - Versioning and Experiment Tracking Module.
Demonstrates:
- Cryptographic SHA-256 data and pipeline checksums (emulating DVC artifact tracking)
- Optional MLflow experiment tracking integration (with fallback to local JSON metadata)
- Git commit hash provenance tracking
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

from src.utils.config import METADATA_DIR, RAW_DATA_PATH


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 cryptographic hash of a data artifact."""
    if not filepath.exists():
        return "file_not_found"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_commit_hash() -> str:
    """Returns current git commit hash if available, or 'uncommitted'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "local_dev_workspace"


def track_experiment_run(
    experiment_name: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Logs experiment parameters, metrics, and artifact checksums.
    Attempts MLflow logging if installed; otherwise persists to local metadata JSON.
    """
    commit_hash = get_git_commit_hash()
    data_hash = compute_file_sha256(RAW_DATA_PATH)

    run_meta = {
        "experiment_name": experiment_name,
        "git_commit": commit_hash,
        "dataset_sha256": data_hash,
        "parameters": parameters,
        "metrics": metrics,
        "artifacts": artifacts or {},
    }

    # Optional MLflow logging
    mlflow_used = False
    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("git_commit", commit_hash)
            mlflow.set_tag("dataset_sha256", data_hash)
            mlflow_used = True
    except Exception:
        mlflow_used = False

    run_meta["mlflow_logged"] = mlflow_used

    # Always persist locally for offline reproducibility
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    exp_file = METADATA_DIR / f"{experiment_name.replace(' ', '_').lower()}_runs.jsonl"
    with open(exp_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_meta) + "\n")

    return run_meta
