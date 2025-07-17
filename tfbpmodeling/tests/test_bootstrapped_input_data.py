import os

import numpy as np
import pandas as pd
import pytest

from tfbpmodeling.bootstrapped_input_data import BootstrappedModelingInputData
from tfbpmodeling.modeling_input_data import ModelingInputData


def test_initialization_random_state_none(sample_data):
    """Ensure proper initialization with valid inputs."""
    response_df, predictors_df = sample_data
    input_data = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")
    model_df = input_data.get_modeling_data(formula="TF1 + TF1:TF2 + TF1:TF3 - 1")

    boot_data = BootstrappedModelingInputData(
        input_data.response_df,
        model_df,
        n_bootstraps=5,
    )

    assert isinstance(boot_data.response_df, pd.DataFrame)
    assert isinstance(boot_data.model_df, pd.DataFrame)
    assert boot_data.n_bootstraps == 5
    assert boot_data.response_df.index.equals(boot_data.model_df.index)


def test_initialization_random_state(sample_data):
    """Ensure proper initialization with valid inputs."""
    response_df, predictors_df = sample_data
    input_data = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")
    model_df = input_data.get_modeling_data(formula="TF1 + TF1:TF2 + TF1:TF3 - 1")

    data1 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5, random_state=42
    )

    data2 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5, random_state=42
    )

    # data1 and data2 should have the same bootstrap indices
    assert all(
        np.array_equal(i1, i2)
        for i1, i2 in zip(data1.bootstrap_indices, data2.bootstrap_indices)
    )

    # data3 should have different bootstrap indices because of different random_state
    data3 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5, random_state=100
    )

    assert not all(
        np.array_equal(i1, i2)
        for i1, i2 in zip(data3.bootstrap_indices, data2.bootstrap_indices)
    )


def test_invalid_inputs():
    """Ensure errors are raised for invalid inputs."""

    # Require that response_df and model_df are DataFrames
    with pytest.raises(TypeError):
        BootstrappedModelingInputData("not a df", pd.DataFrame(), 5)

    with pytest.raises(TypeError):
        BootstrappedModelingInputData(
            pd.DataFrame(index=["a", "b", "c"]), "not a df", 5
        )

    # require that the response_df and model_df have the same index in the same order
    with pytest.raises(IndexError):
        BootstrappedModelingInputData(
            pd.DataFrame(index=["a", "b", "c"]), pd.DataFrame(index=["a", "c", "b"]), 5
        )

    # require that n_bootstraps is a positive integer
    with pytest.raises(TypeError):
        BootstrappedModelingInputData(
            pd.DataFrame(index=["a", "b", "c"]), pd.DataFrame(index=["a", "b", "c"]), -1
        )


def test_bootstrap_sample_shape(bootstrapped_sample_data):
    """Ensure bootstrap samples maintain the correct shape."""
    sample_indices, sample_weights = bootstrapped_sample_data.get_bootstrap_sample(0)

    assert isinstance(sample_indices, np.ndarray)
    assert isinstance(sample_weights, np.ndarray)
    assert (
        len(sample_indices) == bootstrapped_sample_data.response_df.shape[0]
    )  # Same number of rows


def test_bootstrap_deterministic(sample_data):
    """Ensure setting a seed makes bootstrapping deterministic."""
    response_df, predictors_df = sample_data
    input_data = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")
    model_df = input_data.get_modeling_data(formula="TF1 + TF1:TF2 + TF1:TF3 - 1")

    np.random.seed(42)
    boot_data_1 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5
    )
    np.random.seed(42)
    boot_data_2 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5
    )

    for i in range(5):
        sample_indices1, sample_weights1 = boot_data_1.get_bootstrap_sample(i)
        sample_indices2, sample_weights2 = boot_data_2.get_bootstrap_sample(i)
        assert all(x == y for x, y in zip(sample_indices1, sample_indices2))
        assert all(x == y for x, y in zip(sample_weights1, sample_weights2))


def test_invalid_bootstrap_index(bootstrapped_sample_data):
    """Ensure index errors are raised for out-of-range bootstrap indices."""
    with pytest.raises(IndexError):
        bootstrapped_sample_data.get_bootstrap_sample(-1)

    with pytest.raises(IndexError):
        bootstrapped_sample_data.get_bootstrap_sample(
            bootstrapped_sample_data.n_bootstraps
        )


def test_sample_weights_sum(bootstrapped_sample_data):
    """Ensure sample weights sum to 1 for each bootstrap sample."""
    for i in range(bootstrapped_sample_data.n_bootstraps):
        weights = bootstrapped_sample_data.get_sample_weight(i)
        assert np.isclose(weights.sum(), 1, atol=1e-6)


def test_sample_weights_index_error(bootstrapped_sample_data):
    """Ensure index errors are raised for invalid weight indices."""
    with pytest.raises(IndexError):
        bootstrapped_sample_data.get_sample_weight(-1)

    with pytest.raises(IndexError):
        bootstrapped_sample_data.get_sample_weight(
            bootstrapped_sample_data.n_bootstraps
        )


def test_bootstrap_iteration(bootstrapped_sample_data):
    """Ensure iteration over bootstrap samples works as expected."""
    iterator = iter(bootstrapped_sample_data)

    for _ in range(bootstrapped_sample_data.n_bootstraps):
        sample_indices, sample_weights = next(iterator)
        assert isinstance(sample_indices, np.ndarray)
        assert isinstance(sample_weights, np.ndarray)

    with pytest.raises(StopIteration):
        next(iterator)


def test_bootstrap_regenerate(bootstrapped_sample_data):
    """Ensure reset_bootstrap_samples generates new samples."""
    old_bootstrap_indices = bootstrapped_sample_data.bootstrap_indices.copy()
    bootstrapped_sample_data.regenerate()
    new_bootstrap_indices = bootstrapped_sample_data.bootstrap_indices

    assert len(old_bootstrap_indices) == len(new_bootstrap_indices)
    assert not all(
        np.array_equal(a, b)
        for a, b in zip(old_bootstrap_indices, new_bootstrap_indices)
    )


def test_bootstrapinputmodeldata_serialize_deserialize(
    bootstrapped_random_sample_data, tmp_path
):
    """Tests serialization and deserialization of the BootstrappedModelingInputData
    class."""

    # Create instance
    model_data = bootstrapped_random_sample_data

    # Define a temporary file path
    json_file = tmp_path / "test_bootstrap.json"

    # Serialize the object
    model_data.serialize(json_file)

    # Ensure the file was created
    assert os.path.exists(json_file)

    # Deserialize the object
    loaded_data = BootstrappedModelingInputData.deserialize(json_file)

    # Check that the restored object has the same properties
    pd.testing.assert_frame_equal(model_data.response_df, loaded_data.response_df)
    pd.testing.assert_frame_equal(model_data.model_df, loaded_data.model_df)
    assert model_data.n_bootstraps == loaded_data.n_bootstraps

    # Verify bootstrap indices
    assert len(model_data.bootstrap_indices) == len(loaded_data.bootstrap_indices)
    for orig, restored in zip(
        model_data.bootstrap_indices, loaded_data.bootstrap_indices
    ):
        np.testing.assert_array_equal(orig, restored)

    # Verify sample weights
    assert len(model_data.sample_weights) == len(loaded_data.sample_weights)
    for key in model_data.sample_weights:
        np.testing.assert_array_equal(
            model_data.sample_weights[key], loaded_data.sample_weights[key]
        )
