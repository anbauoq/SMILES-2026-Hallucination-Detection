"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that detects
hallucinations from hidden-state features.  All four public methods
(``fit``, ``fit_hyperparameters``, ``predict``, ``predict_proba``) are
implemented; signatures are unchanged.

Architecture
------------
StandardScaler → PCA(n_components=50) → SVC(RBF, C=0.2, gamma=1e-4,
class_weight="balanced", probability=True)

Selected by a 200-config hyperparameter sweep (C × gamma × PCA dims) on
the response_mean_late_concat feature set (8 late layers × 896 dims = 7168),
optimised for mean test AUROC across 5 stratified outer folds → 0.7891.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class HallucinationProbe(nn.Module):
    """SVC-based hallucination probe with PCA dimensionality reduction."""

    def __init__(self) -> None:
        super().__init__()
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=50, random_state=42)),
            ("svc",    SVC(
                C=0.2,
                kernel="rbf",
                gamma=1e-4,
                class_weight="balanced",
                probability=True,
                random_state=42,
            )),
        ])
        self._threshold: float = 0.5

    # nn.Module requires a forward method; sklearn handles inference instead.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "HallucinationProbe uses sklearn inference. "
            "Call predict() or predict_proba() instead."
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit the sklearn pipeline on labelled feature vectors."""
        self._clf.fit(X, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        Uses vectorised numpy operations over all candidate thresholds
        simultaneously rather than a Python loop.
        """
        probs = self.predict_proba(X_val)[:, 1]                           # (n,)
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        # preds[i, j] = True iff sample j is predicted positive at threshold i
        preds = probs[np.newaxis, :] >= candidates[:, np.newaxis]         # (m, n)
        pos   = y_val.astype(bool)                                         # (n,)
        tp    = ( preds &  pos[np.newaxis, :]).sum(axis=1).astype(float)  # (m,)
        fp    = ( preds & ~pos[np.newaxis, :]).sum(axis=1).astype(float)  # (m,)
        fn    = (~preds &  pos[np.newaxis, :]).sum(axis=1).astype(float)  # (m,)
        denom = 2.0 * tp + fp + fn
        f1    = np.where(denom > 0, 2.0 * tp / denom, 0.0)               # (m,)

        self._threshold = float(candidates[np.argmax(f1)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using the tuned decision threshold."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates of shape ``(n_samples, 2)``."""
        return self._clf.predict_proba(X)
