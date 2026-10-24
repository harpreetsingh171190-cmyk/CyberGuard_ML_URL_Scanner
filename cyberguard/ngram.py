"""Hybrid model: character n-gram TF-IDF on the raw URL + scaled lexical features."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES


class NgramHybrid(ClassifierMixin, BaseEstimator):
    """Expects a DataFrame with the lexical feature columns plus a raw `url` text column."""

    def __init__(self, C: float = 1.0, ngram_range=(3, 5), max_features: int = 50_000):
        self.C = C
        self.ngram_range = ngram_range
        self.max_features = max_features

    def _matrix(self, X, fit: bool = False):
        text, num = X["url"].astype(str), X[FEATURE_NAMES].to_numpy(dtype=float)
        if fit:
            self.tfidf_ = TfidfVectorizer(analyzer="char", ngram_range=self.ngram_range,
                                          max_features=self.max_features, sublinear_tf=True)
            self.scaler_ = StandardScaler()
            return hstack([self.tfidf_.fit_transform(text), csr_matrix(self.scaler_.fit_transform(num))]).tocsr()
        return hstack([self.tfidf_.transform(text), csr_matrix(self.scaler_.transform(num))]).tocsr()

    def fit(self, X, y):
        self.clf_ = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.clf_.fit(self._matrix(X, fit=True), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._matrix(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
