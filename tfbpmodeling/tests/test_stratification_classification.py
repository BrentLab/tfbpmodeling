import numpy as np
import pandas as pd
import pytest

from tfbpmodeling.stratification_classification import stratification_classification


@pytest.fixture
def test_data():
    binding = pd.Series([0.1, 2.5, 1.2, 3.0, 0.8])
    return binding


def test_binding_only_labels_are_correct(test_data):
    binding = test_data
    bins = [0, 1, 2, 4, np.inf]

    labels = stratification_classification(binding, bins=bins)

    expected = np.array([4, 2, 3, 1, 3])
    assert np.array_equal(labels, expected)


def test_non_series_input_raises():
    binding = [0.1, 0.2]
    with pytest.raises(ValueError, match="binding vector must be a pandas Series"):
        stratification_classification(binding)


def test_non_numeric_series_raises():
    binding = pd.Series(["a", "b", "c"])
    with pytest.raises(ValueError, match="binding vector must be numeric"):
        stratification_classification(binding)


def test_too_few_bins_returns_ones(test_data, caplog):
    binding = test_data
    with caplog.at_level("WARNING"):
        result = stratification_classification(binding, bins=[0])
    assert np.all(result == 1)
    assert "The number of bins is less than 2" in caplog.text
