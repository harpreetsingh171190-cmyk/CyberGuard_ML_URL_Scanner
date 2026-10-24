"""Matplotlib helpers that save report images."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from .features import FEATURE_NAMES


def _save(fig, out_dir, name):
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / name, dpi=150, bbox_inches="tight")


def confusion(y_true, y_prob, threshold, title, out_dir=None):
    cm = confusion_matrix(y_true, (np.asarray(y_prob) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Legit", "Malicious"],
                yticklabels=["Legit", "Malicious"])
    ax.set(title=f"{title} (threshold {threshold:.2f})", xlabel="Predicted", ylabel="Actual")
    _save(fig, out_dir, "confusion_matrix.png")
    return fig


def roc_and_pr(y_true, probs: dict, out_dir=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        a.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_true, p)
        b.plot(rc, pr, lw=2, label=name)
    a.plot([0, 1], [0, 1], "--", color="gray")
    a.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate"); a.legend(loc="lower right")
    b.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision"); b.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, out_dir, "roc_pr_curves.png")
    return fig


def feature_importance(model, out_dir=None, top: int = 20):
    est = getattr(model, "calibrated_classifiers_", [None])[0]
    est = getattr(est, "estimator", model)
    if not hasattr(est, "feature_importances_"):
        return None
    s = (__import__("pandas").Series(est.feature_importances_, index=FEATURE_NAMES)
         .sort_values().tail(top))
    fig, ax = plt.subplots(figsize=(7, 6))
    s.plot(kind="barh", color="#e07a5f", ax=ax)
    ax.set(title="Top feature importances", xlabel="Importance")
    fig.tight_layout()
    _save(fig, out_dir, "feature_importance.png")
    return fig
