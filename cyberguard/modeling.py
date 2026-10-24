"""Training, evaluation, threshold tuning, persistence and prediction."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import SEED, registered_domain
from .features import FEATURE_NAMES, features_frame, risk_signals, scheme_free
from .ngram import NgramHybrid

TEXT_MODELS = {"N-gram Hybrid"}  # models that also read the raw URL text


def model_input(urls, text: bool = False) -> pd.DataFrame:
    """Feature table; adds the raw `url` column for models that use character n-grams."""
    urls = list(urls)
    X = features_frame(urls)
    if text:
        X["url"] = [scheme_free(u) for u in urls]
    return X


def build_models(seed: int = SEED) -> dict:
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed),
    }
    models["N-gram Hybrid"] = NgramHybrid()
    try:  # optional dependency
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1)
    except ImportError:
        pass
    return models


def metrics_at(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) == 2:
        out["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        out["PR-AUC"] = average_precision_score(y_true, y_prob)
    return out


def pick_threshold(y_true, y_prob, min_recall: float = 0.95) -> float:
    """Highest threshold that still keeps recall >= min_recall (fewest false alarms for that recall)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    ok = [(p, t) for p, r, t in zip(prec[:-1], rec[:-1], thr) if r >= min_recall]
    return float(max(ok, key=lambda x: (x[0], x[1]))[1]) if ok else 0.5


def train_and_compare(train_df: pd.DataFrame, test_df: pd.DataFrame, calibrate: bool = True):
    """Fit every model, score on the held-out test set, return (results_df, fitted, probs, y_test)."""
    full_tr, y_tr = model_input(train_df["url"], True), train_df["label"].to_numpy()
    full_te, y_te = model_input(test_df["url"], True), test_df["label"].to_numpy()
    n_splits = 5 if len(y_tr) >= 100 else 3
    # Group CV by domain, like the train/test split, so CV scores are not inflated by same-site leakage.
    groups = train_df["url"].map(registered_domain).to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    rows, fitted, probs = [], {}, {}
    for name, model in build_models().items():
        X_tr, X_te = (full_tr, full_te) if name in TEXT_MODELS else (full_tr[FEATURE_NAMES], full_te[FEATURE_NAMES])
        cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, groups=groups, scoring="f1")
        model.fit(X_tr, y_tr)
        if calibrate and len(y_tr) >= 200:  # probabilities you can trust more
            model = CalibratedClassifierCV(model, cv=3, method="isotonic").fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        fitted[name], probs[name] = model, prob
        rows.append({"Model": name, **metrics_at(y_te, prob),
                     "CV F1 mean": cv_f1.mean(), "CV F1 std": cv_f1.std()})
    results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return results, fitted, probs, y_te


def save_bundle(model, name: str, threshold: float, metrics: dict, path: str | Path, demo: bool = False,
                baseline: dict | None = None, text_input: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "name": name, "features": FEATURE_NAMES,
                 "threshold": threshold, "metrics": metrics, "demo": demo,
                 "baseline": baseline or {}, "text_input": text_input}, path)
    return path


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if bundle.get("features") != FEATURE_NAMES:
        raise ValueError("Model was trained with a different feature set. Re-train it.")
    return bundle


def predict_urls(bundle: dict, urls) -> pd.DataFrame:
    """Score URLs. Returns url, probability, verdict, risk level."""
    urls = list(urls)
    prob = bundle["model"].predict_proba(model_input(urls, bundle.get("text_input", False)))[:, 1]
    thr = bundle["threshold"]
    level = np.select([prob >= max(thr, 0.8), prob >= thr, prob >= thr * 0.5], ["HIGH", "MEDIUM", "LOW"], "MINIMAL")
    return pd.DataFrame({
        "url": urls, "malicious_probability": prob.round(4),
        "verdict": np.where(prob >= thr, "malicious", "legitimate"), "risk_level": level,
    })


def local_explanation(bundle: dict, url: str, top: int = 5) -> list[dict]:
    """Model-agnostic 'why this score?': replace each feature with a typical (median) value
    and measure how much the malicious probability drops. Positive = pushes towards malicious.
    Cheap alternative to SHAP with no extra dependency."""
    x = model_input([url], bundle.get("text_input", False))
    base = bundle.get("baseline") or {}
    p0 = float(bundle["model"].predict_proba(x)[0, 1])
    variants = []
    for f in FEATURE_NAMES:
        v = x.copy()
        v[f] = base.get(f, 0.0)
        variants.append(v)
    probs = bundle["model"].predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    rows = [{"feature": f, "value": float(x[f].iloc[0]), "typical": float(base.get(f, 0.0)),
             "impact": round(p0 - float(p), 4)} for f, p in zip(FEATURE_NAMES, probs)]
    rows = [r for r in rows if r["impact"] != 0]
    return sorted(rows, key=lambda r: abs(r["impact"]), reverse=True)[:top]


def explain(bundle: dict, url: str) -> dict:
    row = predict_urls(bundle, [url]).iloc[0].to_dict()
    row["signals"] = risk_signals(url)
    row["top_features"] = local_explanation(bundle, url)
    return row


def dump_report(results: pd.DataFrame, extra: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"results": results.round(4).to_dict("records"), **extra}, indent=2))


def error_analysis(test_df: pd.DataFrame, y_prob, threshold: float, top: int = 10) -> pd.DataFrame:
    """Most confident mistakes: false positives (legit flagged) and false negatives (threats missed)."""
    df = test_df[["url", "label"]].copy()
    df["malicious_probability"] = np.asarray(y_prob).round(4)
    df["predicted"] = (df["malicious_probability"] >= threshold).astype(int)
    wrong = df[df["predicted"] != df["label"]].copy()
    wrong["error_type"] = np.where(wrong["label"] == 0, "false_positive", "false_negative")
    wrong["confidence"] = (wrong["malicious_probability"] - threshold).abs()
    wrong = wrong.sort_values("confidence", ascending=False).groupby("error_type").head(top)
    wrong["red_flags"] = wrong["url"].map(lambda u: "; ".join(risk_signals(u)))
    return wrong.drop(columns=["confidence", "predicted"]).reset_index(drop=True)
