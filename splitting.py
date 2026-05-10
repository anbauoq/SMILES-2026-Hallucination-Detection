"""
splitting.py — Train / validation / test split utilities (student-implemented).

Returns a 5-fold stratified cross-validation setup.  For each outer fold,
80 % of the data forms train+val; the remaining 20 % is the held-out test
fold.  The train+val portion is further split (85 / 15) into train and val,
so the val set is used for decision-threshold tuning inside the probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

_N_SPLITS: int = 5

# Type alias for one (train, val, test) index triple.
_Split = tuple[np.ndarray, np.ndarray | None, np.ndarray]


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[_Split]:
    """5-fold stratified cross-validation split.

    Each fold yields a non-overlapping test set (≈20 % of the data) and
    further splits the remaining 80 % into train (≈68 %) and val (≈12 %)
    for threshold tuning.

    Args:
        y:            Label array of shape ``(N,)``.
        df:           Not used. Kept for interface compatibility with the fixed caller.
        test_size:    Not used. Outer fold size is fixed at 1/_N_SPLITS (≈20 %).
                      Kept for interface compatibility with the fixed caller.
        val_size:     Fraction of the train+val portion reserved for val
                      (default 0.15 → ≈12 % of total).
        random_state: Random seed for reproducibility.

    Returns:
        List of ``_N_SPLITS`` ``(idx_train, idx_val, idx_test)`` tuples.
    """
    idx = np.arange(len(y))
    outer = StratifiedKFold(n_splits=_N_SPLITS, shuffle=True, random_state=random_state)
    splits = []
    for fold_idx, (idx_train_val, idx_test) in enumerate(outer.split(idx, y)):
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_size,
            random_state=random_state + fold_idx + 1,
            stratify=y[idx_train_val],
        )
        splits.append((idx_train, idx_val, idx_test))
    return splits
