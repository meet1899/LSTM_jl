from __future__ import annotations

import numpy as np
import pytest

from src.sequence import create_sequences, create_sequences_with_past_context


def test_create_sequences_shape() -> None:
    features = np.arange(20).reshape(10, 2)
    targets = np.arange(10).reshape(10, 1)
    x_seq, y_seq = create_sequences(features, targets, lookback=3)
    assert x_seq.shape == (7, 3, 2)
    assert y_seq.shape == (7,)


def test_create_sequences_requires_matching_lengths() -> None:
    features = np.arange(20).reshape(10, 2)
    targets = np.arange(9).reshape(9, 1)
    with pytest.raises(ValueError):
        create_sequences(features, targets, lookback=3)


def test_create_sequences_with_past_context_uses_history() -> None:
    past_features = np.arange(12).reshape(6, 2)
    past_targets = np.arange(6).reshape(6, 1)
    features = np.arange(8).reshape(4, 2)
    targets = np.arange(4).reshape(4, 1)
    x_seq, y_seq = create_sequences_with_past_context(
        features=features,
        targets=targets,
        lookback=3,
        past_features=past_features,
        past_targets=past_targets,
    )
    assert x_seq.shape == (4, 3, 2)
    assert y_seq.shape == (4,)
