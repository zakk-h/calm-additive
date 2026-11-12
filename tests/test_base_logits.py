"""Regression tests for fixed observation-level classification logit offsets."""

import numpy as np
import pytest

from calm_additive import CALMClassifier
from calm_additive.fitters import EBMClassifier


def make_data(n=160, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    base = 0.8 * X[:, 0] - 0.4 * X[:, 1]
    correction = 0.9 * X[:, 2] * (X[:, 1] > 0)
    p = 1 / (1 + np.exp(-(base + correction)))
    y = rng.binomial(1, p)
    return X, y, base


def fit_offset_calm(X, y, score):
    return CALMClassifier(
        teacher="xgb",
        teacher_params={"n_estimators": 16, "n_jobs": 1, "max_depth": 2},
        curve_fitter_params={"outer_bags": 1, "max_rounds": 40, "n_jobs": 1},
        selection="all",
        max_conditional_interactions=None,
        max_partition_depth=1,
    ).fit(X, y, init_score=score)


def test_ebm_offsets_are_applied_at_fit_and_predict():
    X, y, base = make_data()
    ebm = EBMClassifier(outer_bags=1, max_rounds=40, n_jobs=1)
    ebm.fit(X, y, init_score=base)
    p = ebm.predict_proba(X, init_score=base)
    assert np.isfinite(p).all()
    assert (p > 0).all() and (p < 1).all()
    assert not np.allclose(p, ebm.predict_proba(X, init_score=np.zeros(len(X))))


def test_calm_array_offsets_affect_training_effects_and_predictions():
    X, y, base = make_data()
    model = fit_offset_calm(X, y, base)
    assert model.teacher_.offset_fitted_
    assert np.isfinite(model.effect_._y_pred).all() if model.effect_._y_pred is not None else True
    p = model.predict_proba(X, init_score=base)[:, 1]
    assert np.isfinite(p).all()
    assert np.array_equal(model.predict(X, init_score=base), (p > 0.5).astype(int))
    p_shifted = model.predict_proba(X, init_score=base + 1)[:, 1]
    np.testing.assert_allclose(
        np.log(p_shifted / (1 - p_shifted)) - np.log(p / (1 - p)),
        np.ones(len(X)),
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="init_score is required"):
        model.predict_proba(X)


def test_calm_callable_offsets_evaluate_on_new_rows():
    X, y, base = make_data()
    fn = lambda A: 0.8 * A[:, 0] - 0.4 * A[:, 1]
    model = fit_offset_calm(X, y, fn)
    X_new, _, expected = make_data(n=24, seed=25)
    np.testing.assert_allclose(
        model.predict_proba(X_new),
        model.predict_proba(X_new, init_score=expected),
        atol=1e-10,
    )


def test_offset_validation_and_unsupported_fitter():
    X, y, base = make_data()
    model = CALMClassifier()
    with pytest.raises(ValueError, match="finite 1D vector"):
        model.fit(X, y, init_score=base[:-1])
    with pytest.raises(ValueError, match="finite 1D vector"):
        model.fit(X, y, init_score=np.full(len(X), np.nan))
    with pytest.raises(ValueError, match="refitted curve_fitter"):
        CALMClassifier(curve_fitter="surrogate").fit(X, y, init_score=base)
    with pytest.raises(ValueError, match='selection_target="teacher"'):
        CALMClassifier(selection_target="y").fit(X, y, init_score=base)
