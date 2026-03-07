"""Sequence builders for time-series models."""

from __future__ import annotations

import numpy as np


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) windows from feature and target arrays."""
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")
    if len(features) <= lookback:
        raise ValueError("Not enough rows to build sequences for the given lookback")

    x_seq, y_seq = [], []
    for i in range(lookback, len(features)):
        x_seq.append(features[i - lookback : i])
        y_seq.append(targets[i, 0])

    return np.array(x_seq), np.array(y_seq)


def create_sequences_with_past_context(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    past_features: np.ndarray,
    past_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sequences for val/test splits using only prior split history as context.

    This allows the first prediction in a split to still have a full lookback window,
    without leaking any future observations.
    """
    if len(past_features) < lookback or len(past_targets) < lookback:
        raise ValueError("past context must have at least `lookback` rows")

    combined_x = np.vstack([past_features[-lookback:], features])
    combined_y = np.vstack([past_targets[-lookback:], targets])

    x_seq, y_seq = [], []
    for i in range(lookback, len(combined_x)):
        x_seq.append(combined_x[i - lookback : i])
        y_seq.append(combined_y[i, 0])

    return np.array(x_seq), np.array(y_seq)
