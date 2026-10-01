import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)

from api.ml_core import FEATURES, LABELS, rain_shap

DATA = "data/weather_forecast_data.csv"
MODEL_OUT = "ml_model/rain_rf_smote_pipeline.joblib"
META_OUT = "ml_model/model_meta.json"
METRICS_OUT = "ml_model/model_metrics.json"

df = pd.read_csv(DATA)
X = df[FEATURES]
y = df["Rain"].map({"no rain": 0, "rain": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
])

param_dist = {
    "smote__k_neighbors": [3, 5, 7, 9],
    "smote__sampling_strategy": [0.5, 0.75, 1.0],
    "rf__n_estimators": [200, 300, 500],
    "rf__max_depth": [None, 6, 10, 15, 20],
    "rf__min_samples_split": [2, 5, 10, 20],
    "rf__min_samples_leaf": [1, 2, 4, 8],
    "rf__max_features": ["sqrt", "log2", 0.5],
    "rf__bootstrap": [True, False],
    "rf__criterion": ["gini", "entropy"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(pipe, param_dist, n_iter=40, scoring="f1", cv=cv,
                            n_jobs=-1, random_state=42, verbose=1, refit=True)
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best CV F1:", round(search.best_score_, 4))
print("Best params:", search.best_params_)

# Threshold from out-of-fold predictions on TRAIN only (no test leakage)
oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
best_thr = float(thr[np.argmax(f1s)])
print("Best threshold:", round(best_thr, 3))

# Evaluate on the untouched test set
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= best_thr).astype(int)
print(classification_report(y_test, pred, digits=3))

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
fpr, tpr, _ = roc_curve(y_test, proba)
pr_p, pr_r, _ = precision_recall_curve(y_test, proba)


def thin(*arrays, n=120):
    """Downsample curves so the JSON stays small."""
    idx = np.unique(np.linspace(0, len(arrays[0]) - 1, n).astype(int))
    return [np.asarray(a)[idx].round(4).tolist() for a in arrays]


roc_fpr, roc_tpr = thin(fpr, tpr)
pr_prec, pr_rec = thin(pr_p, pr_r)

# Global SHAP importance on a sample of real test rows
rf = best.named_steps["rf"]
explainer = shap.TreeExplainer(rf)
sample = X_test.sample(min(300, len(X_test)), random_state=42)
vals, base = rain_shap(explainer, sample)
shap_pp = np.abs(vals).mean(axis=0) * 100
shap_share = shap_pp / shap_pp.sum()
imp = rf.feature_importances_

importance = sorted(
    [{"feature": f, "label": LABELS[f], "impurity": float(imp[i]),
      "shap_pp": float(shap_pp[i]), "shap_share": float(shap_share[i])}
     for i, f in enumerate(FEATURES)],
    key=lambda d: -d["shap_pp"],
)

metrics = {
    "model": "Random Forest + SMOTE",
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "rain_rate": float(y.mean()),
    "cv_f1": float(search.best_score_),
    "best_params": {k: (v if v is None or isinstance(v, (str, bool)) else float(v))
                    for k, v in search.best_params_.items()},
    "threshold": best_thr,
    "test": {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    },
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "roc": {"fpr": roc_fpr, "tpr": roc_tpr},
    "pr": {"precision": pr_prec, "recall": pr_rec},
    "importance": importance,
    "shap_base_value": round(base * 100, 2),
}
print(json.dumps(metrics["test"], indent=2))

# Atomic save: write temp file, verify it reloads, then move into place
os.makedirs("ml_model", exist_ok=True)
fd, tmp = tempfile.mkstemp(dir="ml_model", suffix=".joblib")
os.close(fd)
joblib.dump(best, tmp, compress=3)
joblib.load(tmp)
os.replace(tmp, MODEL_OUT)

with open(META_OUT, "w") as f:
    json.dump({"threshold": best_thr}, f)
with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f)

print("Saved:", MODEL_OUT, "|", round(os.path.getsize(MODEL_OUT) / 1e6, 2), "MB")
print("Saved:", META_OUT, "and", METRICS_OUT)