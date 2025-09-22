import numpy as np
import pandas as pd
import pytest

from tfbpmodeling.modeling_input_data import ModelingInputData


def test_init_valid_data(sample_data):
    """Test successful initialization with valid data."""
    response_df, predictors_df = sample_data
    instance = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")

    assert isinstance(instance, ModelingInputData)
    assert instance.perturbed_tf == "TF1"
    assert instance.feature_col == "target_symbol"
    assert isinstance(instance.response_df, pd.DataFrame)
    assert isinstance(instance.predictors_df, pd.DataFrame)


def test_init_missing_feature_column():
    """Test initialization failure when feature_col is missing."""
    response_df = pd.DataFrame({"expression": [2.5, 3.2, 1.8, 4.1, 2.9]})
    predictors_df = pd.DataFrame({"TF1": [0.5, 0.8, 0.2, 0.9, 0.7]})

    with pytest.raises(KeyError):
        ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")


def test_init_invalid_response_columns(sample_data):
    """Test error when response_df has incorrect column count."""
    response_df, predictors_df = sample_data
    response_df["extra_col"] = [1, 2, 3, 4, 5]  # Adding an extra column

    with pytest.raises(
        ValueError, match="Response DataFrame must have exactly one numeric column"
    ):
        ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")


def test_init_perturbed_tf_not_in_predictors(sample_data):
    """Test error if perturbed TF is not in predictors."""
    response_df, predictors_df = sample_data

    with pytest.raises(
        KeyError, match="Perturbed TF 'TFX' not found in predictor index"
    ):
        ModelingInputData(response_df, predictors_df, perturbed_tf="TFX")


def test_blacklist_masking(sample_data):
    """Test that feature_blacklist correctly removes specified features."""
    response_df, predictors_df = sample_data
    blacklist = ["gene1", "gene3"]

    instance = ModelingInputData(
        response_df, predictors_df, perturbed_tf="TF1", feature_blacklist=blacklist
    )

    assert instance.blacklist_masked is True
    assert all(gene not in instance.predictors_df.index for gene in blacklist)


def test_perturbed_tf_automatically_blacklisted(sample_data):
    """Ensure the perturbed TF is automatically blacklisted if not explicitly in
    blacklist."""
    response_df, predictors_df = sample_data

    instance = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")

    assert "TF1" in instance.feature_blacklist
    assert instance.blacklist_masked is True


def test_top_n_feature_selection(sample_data):
    """Ensure top_n feature selection ranks features correctly."""
    response_df, predictors_df = sample_data

    instance = ModelingInputData(
        response_df, predictors_df, perturbed_tf="TF1", top_n=3
    )

    assert instance.top_n_masked is True
    assert len(instance.top_n_features) == 3


def test_top_n_invalid_values(sample_data):
    """Ensure invalid top_n values raise errors."""
    response_df, predictors_df = sample_data

    with pytest.raises(ValueError):
        ModelingInputData(response_df, predictors_df, perturbed_tf="TF1", top_n=-5)

    with pytest.raises(ValueError):
        ModelingInputData(
            response_df, predictors_df, perturbed_tf="TF1", top_n="abc"  # type: ignore
        )


def test_get_model_data_no_masking(modeling_input_instance):
    """Ensure get_model_data returns correct unmasked data."""
    data = modeling_input_instance.get_modeling_data(
        formula="TF1 + TF1:TF2 + TF1:TF3 - 1"
    )

    assert isinstance(data, pd.DataFrame)
    assert data.shape[1] == 3
    # assert that the names are "TF1", "TF1:TF2", "TF1:TF3"
    assert all(
        col in data.columns for col in ["TF1", "TF1:TF2", "TF1:TF3"]
    ), "Predictor columns are not as expected."


def test_scale_by_std(modeling_input_instance):
    """Test whether get_modeling_data correctly scales (but does not center)
    predictors."""
    formula = "TF1 + TF1:TF2 + TF1:TF3 - 1"

    data_scaled = modeling_input_instance.get_modeling_data(
        formula=formula, scale_by_std=True, drop_intercept=True
    )

    data_unscaled = modeling_input_instance.get_modeling_data(
        formula=formula, drop_intercept=True
    )

    assert isinstance(data_scaled, pd.DataFrame), "Returned object is not a DataFrame"
    assert data_scaled.shape[1] == 3, f"Expected 3 columns, got {data_scaled.shape[1]}"
    assert all(
        col in data_scaled.columns for col in ["TF1", "TF1:TF2", "TF1:TF3"]
    ), "Predictor columns are not as expected"

    stds = data_scaled.std(axis=0, ddof=0)
    means = data_scaled.mean(axis=0)

    stds_unscaled = data_unscaled.std(axis=0, ddof=0)

    assert np.allclose(stds, 1, atol=1e-6), f"Columns not scaled: stds = {stds}"
    assert not np.allclose(
        means, 0, atol=0.1
    ), f"Columns unexpectedly centered: means = {means}"

    assert not np.allclose(
        stds_unscaled, 1, atol=0.1
    ), f"Expected unscaled data, but stds close to 1: {stds_unscaled}"


def test_get_model_data_with_top_n_masking(sample_data):
    """Ensure get_model_data applies top_n masking correctly."""
    response_df, predictors_df = sample_data

    instance = ModelingInputData(
        response_df, predictors_df, perturbed_tf="TF1", top_n=2
    )
    data = instance.get_modeling_data(formula="TF1 + TF1:TF2 + TF1:TF3 -1")

    assert len(data) == 2
    assert len(data.columns) == 3


def test_get_model_data_with_blacklist(sample_data):
    """Ensure get_model_data applies blacklist masking correctly."""
    response_df, predictors_df = sample_data
    blacklist = ["gene1", "gene3"]

    instance = ModelingInputData(
        response_df, predictors_df, perturbed_tf="TF1", feature_blacklist=blacklist
    )
    data = instance.get_modeling_data(formula="TF1 + TF2 + TF3")

    assert "gene1" not in data.index
    assert "gene3" not in data.index


def test_get_model_data_invalid_formula(sample_data):
    """Ensure an invalid formula raises an error."""
    response_df, predictors_df = sample_data

    instance = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")

    with pytest.raises(ValueError):
        instance.get_modeling_data(formula="")


def test_from_files(monkeypatch, tmp_path):
    """Ensure from_files correctly loads data."""
    response_path = tmp_path / "response.csv"
    predictors_path = tmp_path / "predictors.csv"

    response_data = pd.DataFrame(
        {
            "target_symbol": ["gene1", "gene2"],
            "expression": [2.5, 3.2],
        }
    )
    predictors_data = pd.DataFrame(
        {
            "target_symbol": ["gene1", "gene2"],
            "TF1": [0.5, 0.8],
            "TF2": [1.2, 1.5],
        }
    )

    response_data.to_csv(response_path, index=False)
    predictors_data.to_csv(predictors_path, index=False)

    instance = ModelingInputData.from_files(
        response_path=str(response_path),
        predictors_path=str(predictors_path),
        perturbed_tf="TF1",
    )

    assert isinstance(instance, ModelingInputData)
    assert instance.response_df.shape[0] == 2
    assert instance.predictors_df.shape[0] == 2
