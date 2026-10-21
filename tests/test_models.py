"""Small deterministic fixtures exercise learned preprocessing and calibration.

These tests do not estimate IEEE-CIS model performance.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud.models import ModelBundle, SigmoidCalibrator, canonical_frame, make_pipeline


def _training_fixture():
    n = 160
    return pd.DataFrame({
        "amount": np.tile([10.0, 20.0, np.nan, 40.0], n // 4),
        "all_missing": np.full(n, np.nan),
        "product": ["A"] * (n // 2) + ["B"] * (n // 2),
    }), np.tile([0, 1], n // 2)


@pytest.mark.parametrize("kind", ["logistic", "xgboost"])
def test_train_only_encoders_unknowns_missing_and_serialized_scoring(kind, tmp_path):
    training, y = _training_fixture()
    x = canonical_frame(training)
    config = {"seed": 42, "threads": 1, "logistic_max_iter": 2000}
    params = {"n_estimators": 3, "max_depth": 2} if kind == "xgboost" else None
    pipeline = make_pipeline(x, kind, config, params)
    pipeline.fit(x, y)
    bundle = ModelBundle(kind, pipeline, list(x), True, False)
    transformer = pipeline.named_steps["transform"]
    imputer = transformer.named_transformers_["numeric"].named_steps["impute"]
    training_medians = imputer.statistics_.copy()
    encoder = transformer.named_transformers_["categorical"]
    vocabulary_before = [categories.copy() for categories in encoder.categories_]

    # Both an unseen category and an unseen missing-category token must score;
    # a prediction batch with extreme values must not refit the median or vocab.
    later = pd.DataFrame({
        "amount": [1e6, np.nan, 10],
        "all_missing": [np.nan, 5, np.nan],
        "product": pd.Categorical(["NEVER_SEEN", None, "A"]),
        "TransactionID": [999, 1000, 1001],
    })
    probabilities = bundle.predict(later)
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_array_equal(imputer.statistics_, training_medians)
    for before, after in zip(vocabulary_before, encoder.categories_, strict=True):
        np.testing.assert_array_equal(before, after)
        assert "NEVER_SEEN" not in after

    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict(later), probabilities, rtol=0, atol=0)
    # Column selection in the bundle defines the inference schema and order.
    np.testing.assert_allclose(
        loaded.predict(later[list(reversed(later.columns))]), probabilities, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="Missing prediction columns"):
        loaded.predict(later.drop(columns="amount"))


def test_calibration_is_a_separate_fit_and_never_refits_base_classifier():
    training, y = _training_fixture()
    x = canonical_frame(training)
    pipeline = make_pipeline(x, "logistic", {
        "seed": 42, "threads": 1, "logistic_max_iter": 2000,
    })
    pipeline.fit(x, y)
    coefficients_before = pipeline.named_steps["model"].coef_.copy()
    base = ModelBundle("logistic", pipeline, list(x), True, False)
    raw_before = base.predict(training, calibrated=False)

    # A distinct block is used to fit only the calibrator. Deliberately different
    # score-label pairs demonstrate that this fit can change probabilities.
    calibration_scores = np.array([0.05, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 0.95])
    calibration_labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    calibrator = SigmoidCalibrator().fit(calibration_scores, calibration_labels)
    base.calibrator = calibrator
    np.testing.assert_allclose(base.predict(training, calibrated=False), raw_before)
    np.testing.assert_array_equal(pipeline.named_steps["model"].coef_, coefficients_before)
    np.testing.assert_allclose(
        base.predict(training, calibrated=True), calibrator.predict(raw_before)
    )
    calibrator_parameters = calibrator.model.coef_.copy()
    _ = calibrator.predict([0, 1, 0.25, 0.75])
    np.testing.assert_array_equal(calibrator.model.coef_, calibrator_parameters)
    assert np.isfinite(calibrator.predict([0, 1])).all()


def test_calibration_requires_both_classes_and_fit_before_prediction():
    with pytest.raises(ValueError, match="both classes"):
        SigmoidCalibrator().fit([0.1, 0.2], [0, 0])
    with pytest.raises(ValueError, match="not fitted"):
        SigmoidCalibrator().predict([0.1])
