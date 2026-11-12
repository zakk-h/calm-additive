# CALM: interpretable by design

Accurate locally additive models with conditional feature effects for tabular data.
Python package for the NeurIPS 2026 paper
*Interpretability-by-Design with Accurate Locally Additive Models and Conditional Feature Effects*
([arXiv](https://arxiv.org/abs/2602.16503)).

**Project page:** https://givasile.github.io/calm-additive
**Documentation:** https://givasile.github.io/calm-additive/docs/

```bash
pip install calm-additive
```

```python
from calm_additive import CALMRegressor

model = CALMRegressor().fit(X_train, y_train)
model.predict(X_test)
model.print_summary()
```

Built on [effector](https://github.com/givasile/effector).


## Fixed base-logit offsets (fork extension)

`CALMClassifier.fit(X, y, init_score=...)` now accepts a fixed **binary-classification logit offset** for each observation. The model learns an additive correction on top of the supplied base logits:

```python
from calm_additive import CALMClassifier

# Option 1: one fixed vector on training, then a vector for each prediction batch.
model = CALMClassifier().fit(X_train, y_train, init_score=train_base_logits)
probability = model.predict_proba(X_test, init_score=test_base_logits)[:, 1]

# Option 2: a function mapping rows to base logits (also works for new rows).
model = CALMClassifier().fit(
    X_train, y_train,
    init_score=lambda X: base_model.decision_function(X),
)
probability = model.predict_proba(X_test)[:, 1]
```

Offsets are passed to the XGBoost teacher as `base_margin`, and to the
regional EBM as `init_score`. When offsets are present, the teacher exposes its
**learned correction in logit space** to the effect estimator so that
partition proposal and interaction selection focus on the remaining signal,
not on structure already modeled by the offset. The output probability is
`sigmoid(base_logit + CALM_correction_logit)`.

Only binary classification with the built-in `teacher="xgb"`,
`selection_target="teacher"`, and a refitted curve fitter (the default
`"ebm"`) is supported for offsets. The EBM receives offsets in training
and prediction. When a vector is used at fit time, supply the matching vector
for each prediction batch. Custom curve fitters need to implement
`fit(..., init_score=)` and `predict(..., init_score=)`.

The per-feature curves represent **corrections** to the fixed baseline;
any view which does not accept the offset should not be used to interpret
the complete probability.

## Acknowledgments

The work leading to these results has been funded by the European Union under Grant Agreement No. 101289094 (project AIRIS). Views and opinions expressed are however those of the authors and do not necessarily reflect those of the European Union or the granting authority (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.

The research leading to this work has received funding from the European Union’s Horizon Europe research and innovation program under Grant Agreement No: 101135826 (aidapt.eu).
