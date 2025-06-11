import numpy as np
from sklearn.linear_model import LassoCV

from tfbpmodeling.lasso_modeling import (
    BootstrapModelResults,
)
from tfbpmodeling.loop_modeling import bootstrap_stratified_cv_loop


def test_bootstrap_stratified_cv_loop(
    random_sample_data,
    bootstrapped_random_sample_data,
    tmp_path,
):
    """Test that bootstrap_stratified_cv_loop runs and saves files to tmp_path."""
    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]
    estimator = LassoCV()
    results = bootstrap_stratified_cv_loop(
        bootstrapped_data=bootstrapped_random_sample_data,
        perturbed_tf_series=perturbed_tf_series,
        estimator=estimator,
        ci_percentile=20.0,
        stabilization_ci_start=10.0,
        num_samples_for_stabilization=10,
        output_dir=str(tmp_path),
        bins=[0, 8, 64, 512, np.inf],
    )

    # Check result object
    assert isinstance(results, BootstrapModelResults)

    # Check that a file was saved in tmp_path
    saved_files = list(tmp_path.glob("selected_variables_ci_*.txt"))
    assert len(saved_files) >= 1

    # Optional: check that saved file contains valid variable names
    with saved_files[-1].open("r") as f:
        contents = f.read().strip()
        assert len(contents) > 0


def test_bootstrap_stratified_cv_loop_no_variables_selected(
    random_sample_data_no_signal,
    bootstrapped_random_sample_data_no_signal,
    tmp_path,
):
    """Ensure that when predictors have no signal, no variables are selected at high
    CI."""
    perturbed_tf_series = random_sample_data_no_signal.predictors_df[
        random_sample_data_no_signal.perturbed_tf
    ]
    ci_percentile = 98.0
    stabilization_ci_start = 50.0
    estimator = LassoCV()
    results = bootstrap_stratified_cv_loop(
        bootstrapped_data=bootstrapped_random_sample_data_no_signal,
        perturbed_tf_series=perturbed_tf_series,
        estimator=estimator,
        ci_percentile=ci_percentile,
        stabilization_ci_start=stabilization_ci_start,
        num_samples_for_stabilization=10,
        output_dir=str(tmp_path),
        bins=[0, 8, 64, 512, np.inf],
    )

    assert isinstance(results, BootstrapModelResults)

    selected_variables = results.extract_significant_coefficients(
        ci_level=str(stabilization_ci_start),
    )
    assert (
        len(selected_variables) == 0
    ), f"Variables incorrectly selected: {selected_variables}"
