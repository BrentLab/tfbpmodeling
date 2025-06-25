import numpy as np
import pandas as pd
import pytest

from tfbpmodeling.stratification_classification import stratification_classification


@pytest.fixture
def test_data():
    binding = pd.Series([0.1, 2.5, 1.2, 3.0, 0.8])
    perturbation = pd.Series([1.1, 0.4, 2.2, 3.0, 0.5])
    return binding, perturbation


def test_binding_only_labels_are_correct(test_data):
    binding, perturbation = test_data
    bins = [0, 1, 2, 4, np.inf]

    labels = stratification_classification(
        binding, perturbation, bins=bins, bin_by_binding_only=True
    )

    expected = np.array([4, 2, 3, 1, 3])
    assert np.array_equal(labels, expected)


def test_combined_stratification_labels_are_correct(test_data):
    binding, perturbation = test_data
    bins = [0, 1, 2, 4, np.inf]

    labels = stratification_classification(
        binding, perturbation, bins=bins, bin_by_binding_only=False
    )

    # Input:
    # binding =       [0.1, 2.5, 1.2, 3.0, 0.8]
    # perturbation =  [1.1, 0.4, 2.2, 3.0, 0.5]
    #
    # Step 1: Compute descending ranks (method="min")
    # binding ranks:      [5.0, 2.0, 3.0, 1.0, 4.0]
    # perturbation ranks: [3.0, 5.0, 2.0, 1.0, 4.0]
    #
    # Step 2: Bin ranks using pd.cut with bins = [0, 1, 2, 4, np.inf], right=True
    # binding_bin:        [4, 2, 3, 1, 3]
    # perturbation_bin:   [3, 4, 2, 1, 3]
    #
    # Step 3: Compute combined stratification labels:
    #   (binding_bin - 1) * 4 + perturbation_bin
    # = [(4-1)*4 + 3, (2-1)*4 + 4, (3-1)*4 + 2, (1-1)*4 + 1, (3-1)*4 + 3]
    # = [15, 8, 10, 1, 11]
    expected = np.array([15, 8, 10, 1, 11])
    assert np.array_equal(labels, expected)


def test_invalid_length_raises():
    binding = pd.Series([0.1, 0.2])
    perturbation = pd.Series([0.1])
    with pytest.raises(
        ValueError, match="length of the binding and perturbation vectors must be equal"
    ):
        stratification_classification(binding, perturbation)


def test_non_series_input_raises():
    binding = [0.1, 0.2]
    perturbation = pd.Series([0.1, 0.2])
    with pytest.raises(ValueError, match="binding vector must be a pandas Series"):
        stratification_classification(binding, perturbation)


def test_non_numeric_series_raises():
    binding = pd.Series(["a", "b", "c"])
    perturbation = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="binding vector must be numeric"):
        stratification_classification(binding, perturbation)


def test_too_few_bins_returns_ones(test_data, caplog):
    binding, perturbation = test_data
    with caplog.at_level("WARNING"):
        result = stratification_classification(binding, perturbation, bins=[0])
    assert np.all(result == 1)
    assert "The number of bins is less than 2" in caplog.text
