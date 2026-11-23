import numpy as np

from monitoring.drift import population_stability_index


def test_psi_zero_for_identical_distributions():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    assert population_stability_index(values, values) == 0.0


def test_psi_detects_large_shift():
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    actual = np.array([0.90, 0.05, 0.03, 0.02])
    assert population_stability_index(expected, actual) > 0.25
