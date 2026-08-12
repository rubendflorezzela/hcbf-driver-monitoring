import numpy as np

from hcbf.metrics import compute_binary_metrics


def test_directional_errors_and_basic_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    metrics = compute_binary_metrics(y, p)
    assert metrics.accuracy == 0.5
    assert metrics.class0_to_class1_error == 0.5
    assert metrics.class1_to_class0_error == 0.5
    assert 0 <= metrics.brier <= 1
    assert 0 <= metrics.ece <= 1
