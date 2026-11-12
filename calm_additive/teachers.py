"""Black-box teachers — the reference model CALM distils its regions from.

Each wrapper exposes `fit(X, y)` and `forward(X)` (the `(N,)` view the effect
consumes: positive-class probability for classifiers, values for regressors).
Registered name: ``"xgb"`` (the paper's default).
"""

import xgboost as xgb

from calm_additive.engine import register_teacher

# the paper's settings; `teacher_params` overrides them key by key
_DEFAULTS = dict(learning_rate=0.1, n_estimators=300, n_jobs=-1, random_state=42)


class XGBClassifier:
    def __init__(self, **params):
        self.model = xgb.XGBClassifier(
            **{**_DEFAULTS, "eval_metric": "logloss", **params}
        )

    def fit(self, X, y, init_score=None):
        self.offset_fitted_ = init_score is not None
        self.model.fit(X, y, base_margin=init_score)
        self.is_fitted_ = True

    def forward(self, X):
        if getattr(self, 'offset_fitted_', False):
            return self.model.predict(X, output_margin=True)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)


class XGBRegressor:
    def __init__(self, **params):
        self.model = xgb.XGBRegressor(**{**_DEFAULTS, **params})

    def fit(self, X, y, init_score=None):
        self.offset_fitted_ = init_score is not None
        self.model.fit(X, y, base_margin=init_score)
        self.is_fitted_ = True

    def forward(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return self.forward(X)


def _xgb(task, **params):
    cls = XGBClassifier if task == "classification" else XGBRegressor
    return cls(**params)


register_teacher("xgb", _xgb)
