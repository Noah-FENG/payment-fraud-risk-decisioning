"""Training-only encoders and compact model bundles shared by fit and score."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


def canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A deterministic, non-fitting type adapter also applied during prediction."""
    result = frame.copy()
    for col in result:
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = pd.to_numeric(result[col], errors="raise").replace([np.inf, -np.inf], np.nan).astype("float32")
        else:
            result[col] = result[col].astype("string").fillna("__MISSING__").astype(object)
    return result


def make_pipeline(x: pd.DataFrame, kind: str, config: dict, params: dict | None = None):
    numerical = list(x.select_dtypes(include="number").columns)
    categorical = [c for c in x if c not in numerical]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if kind == "logistic":
        num_steps.append(("scale", StandardScaler()))
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                max_categories=200, sparse_output=True, dtype=np.float32)
        # saga did not converge within 500 iterations on the real feature matrix
        # (2026-10-21 run, recorded converged=false); lbfgs is the standard choice
        # for n_features << n_samples and converges reliably here. Same L2/C=1.0 model.
        estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=config["logistic_max_iter"],
                                       random_state=config["seed"], class_weight=None)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                 dtype=np.float32)
        estimator = XGBClassifier(**(params or {}), objective="binary:logistic", eval_metric="logloss",
                                  tree_method="hist", random_state=config["seed"],
                                  n_jobs=config["threads"], importance_type="gain")
    transform = ColumnTransformer([("numeric", Pipeline(num_steps), numerical),
                                   ("categorical", encoder, categorical)],
                                  sparse_threshold=1.0 if kind == "logistic" else 0.0)
    return Pipeline([("transform", transform), ("model", estimator)])


@dataclass
class SigmoidCalibrator:
    """Fit sigmoid(a * logit(raw_probability) + b) on calibration block only."""
    model: LogisticRegression | None = None

    @staticmethod
    def _input(probability):
        p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, probability, y):
        if len(np.unique(y)) != 2:
            raise ValueError("Calibration block needs both classes; change development design explicitly")
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.model.fit(self._input(probability), y)
        return self

    def predict(self, probability):
        if self.model is None:
            raise ValueError("Calibrator is not fitted")
        return self.model.predict_proba(self._input(probability))[:, 1]


@dataclass
class ModelBundle:
    name: str
    pipeline: object
    columns: list[str]
    include_anonymous: bool
    include_relations: bool
    calibrator: SigmoidCalibrator | None = None

    def predict(self, features: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        missing = set(self.columns) - set(features)
        if missing:
            raise ValueError(f"Missing prediction columns: {sorted(missing)}")
        x = canonical_frame(features[self.columns])
        raw = self.pipeline.predict_proba(x)[:, 1]
        return self.calibrator.predict(raw) if calibrated and self.calibrator is not None else raw


def timed_fit(pipeline, x, y):
    start = time.perf_counter()
    pipeline.fit(x, y)
    return time.perf_counter() - start
