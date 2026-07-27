# trained_models.zip — contents specification

Thirty weight files: five architectures over the six folds of the rotation in
`src/lso_rotation.py`. Naming is what `src/validate_deposit.py` expects.

```
models/
├── hybrid_fold0.pt      hybrid_fold1.pt   ...  hybrid_fold5.pt
├── lstm_fold0.pt        lstm_fold1.pt     ...  lstm_fold5.pt
├── xgboost_fold0.json   xgboost_fold1.json ... xgboost_fold5.json
├── rf_fold0.joblib      rf_fold1.joblib   ...  rf_fold5.joblib
├── svm_fold0.joblib     svm_fold1.joblib  ...  svm_fold5.joblib
└── scalers/
    └── standardiser_op01.joblib ... standardiser_op24.joblib
```

The per-subject standardisers matter. Section 4.4 normalises each operator
against that operator's own baseline period, so a reader who reloads a model
without the matching scaler will get predictions on the wrong scale and will not
reproduce Table 4.

Record alongside the weights, in `models/environment.txt`:

- PyTorch, scikit-learn and XGBoost versions
- CUDA version, if the models were trained on GPU
- Python version
- Output of `pip freeze`

Version drift in a pickled scikit-learn estimator is the most common reason a
deposited model fails to load two years later.
