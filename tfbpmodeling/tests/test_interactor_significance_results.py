import json
import tempfile

import pandas as pd
import pytest

from tfbpmodeling.interactor_significance_results import InteractorSignificanceResults


def test_interactor_significance_results_init(sample_evaluations):
    """Test object initialization with sample data."""
    results = InteractorSignificanceResults(sample_evaluations)

    assert isinstance(results, InteractorSignificanceResults)
    assert len(results.evaluations) == 3
    assert isinstance(results.to_dataframe(), pd.DataFrame)
    assert results.to_dataframe().shape == (3, 5)  # Ensure correct column count


def test_serialize_deserialize(sample_evaluations):
    """Test saving to and loading from JSON."""
    results = InteractorSignificanceResults(sample_evaluations)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_filepath = temp_file.name

    try:
        # Serialize
        results.serialize(temp_filepath)
        assert temp_filepath is not None
        assert isinstance(temp_filepath, str)

        # Check file exists and is non-empty
        with open(temp_filepath) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3

        # Deserialize
        loaded_results = InteractorSignificanceResults.deserialize(temp_filepath)
        assert isinstance(loaded_results, InteractorSignificanceResults)
        assert len(loaded_results.evaluations) == 3
        assert loaded_results.to_dataframe().equals(results.to_dataframe())

    finally:
        # Cleanup
        import os

        os.remove(temp_filepath)


def test_final_model(sample_evaluations):
    """Test the final_model method, ensuring correct selection of model terms."""
    results = InteractorSignificanceResults(sample_evaluations)
    final_terms = results.final_model()

    # Expected outcome:
    # - "TF1:TF2" (since 0.85 > 0.82)
    # - "TF4" (since 0.81 > 0.78)
    # - "TF5:TF6" (since tie, keeping interactor)
    expected = ["TF1:TF2", "TF4", "TF5:TF6"]
    assert sorted(final_terms) == sorted(expected)


def test_empty_results():
    """Test behavior when initialized with empty data."""
    results = InteractorSignificanceResults([])
    assert results.to_dataframe().empty
    assert results.final_model() == []


def test_invalid_deserialize():
    """Test handling of invalid JSON file structure."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_filepath = temp_file.name

    try:
        # Write incorrect JSON format (dict instead of list)
        with open(temp_filepath, "w") as f:
            json.dump({"interactor": "Invalid"}, f)

        with pytest.raises(ValueError, match="Invalid JSON format"):
            InteractorSignificanceResults.deserialize(temp_filepath)

    finally:
        import os

        os.remove(temp_filepath)
