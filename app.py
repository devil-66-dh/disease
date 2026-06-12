"""Disease prediction app.

This file is a Python equivalent of the logic in `disease prediction.ipynb`.
It trains a RandomForestClassifier on `Testing.csv` and can be used:

- As a CLI:  python app.py
- As an importable module: from app import train_and_save, predict

Note: For a real production UI, add Flask/FastAPI/Streamlit endpoints.
"""

from __future__ import annotations

import argparse
import os
import joblib

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



TARGET_COL = "prognosis"
DEFAULT_DATA_PATH = "Testing.csv"


@dataclass
class ModelBundle:
    model: Any
    feature_columns: List[str]


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in dataset. "
            f"Columns: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def build_pipeline() -> Pipeline:
    # RandomForestClassifier doesn't require feature scaling.
    # Keeping preprocessing minimal avoids failures when dataset contains non-numeric columns.
    return Pipeline(
        steps=[
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )



def train(csv_path: str = DEFAULT_DATA_PATH) -> Tuple[ModelBundle, Dict[str, Any]]:
    X, y = load_dataset(csv_path)

    feature_columns = list(X.columns)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }

    bundle = ModelBundle(model=pipeline, feature_columns=feature_columns)
    return bundle, metrics


def predict(bundle: ModelBundle, input_row: Dict[str, Any]) -> Any:
    """Predict prognosis from a single input row.

    input_row must contain all feature columns from training.
    Values are coerced to numeric when possible.
    """
    missing = [c for c in bundle.feature_columns if c not in input_row]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    extra = [c for c in input_row.keys() if c not in set(bundle.feature_columns)]
    if extra:
        # Not fatal; caller might include metadata fields.
        pass

    row_vals: List[float] = []
    for c in bundle.feature_columns:
        v = input_row[c]
        try:
            row_vals.append(float(v))
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{c}' must be numeric; got {v!r}")

    row = np.array([row_vals], dtype=float)
    return bundle.model.predict(row)[0]



def train_and_save(csv_path: str = DEFAULT_DATA_PATH, model_path: str = "app_bundle.joblib") -> Dict[str, Any]:
    """Train the model and persist it to disk.

    Returns metrics.
    """
    bundle, metrics = train(csv_path)

    payload = {
        "bundle": bundle,
    }

    out_path = model_path
    joblib.dump(payload, out_path)
    return metrics


def load_bundle(model_path: str = "app_bundle.joblib") -> ModelBundle:
    data = joblib.load(model_path)
    if isinstance(data, dict) and "bundle" in data:
        return data["bundle"]
    # Backward/alternate compatibility
    return data


def main() -> None:

    parser = argparse.ArgumentParser(description="Train and evaluate the disease model.")
    parser.add_argument("--csv", default=DEFAULT_DATA_PATH, help="Path to dataset CSV")
    parser.add_argument(
        "--save-model",
        default="app_bundle.joblib",
        help="Where to write trained bundle (default: app_bundle.joblib). If set to 'none', model is not saved.",
    )
    args = parser.parse_args()

    if args.save_model and str(args.save_model).lower() != "none":
        metrics = train_and_save(args.csv, model_path=args.save_model)
    else:
        _, metrics = train(args.csv)


    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Classification report:\n", metrics["report"])


if __name__ == "__main__":
    main()

