import pandas as pd
from sklearn.linear_model import LassoCV

from tfbpmodeling.bootstrap_model_results import BootstrapModelResults
from tfbpmodeling.bootstrap_stratified_cv import bootstrap_stratified_cv_modeling


def test_bootstrap_stratified_cv_modeling(
    random_sample_data, bootstrapped_random_sample_data
):
    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]
    estimator = LassoCV()
    """Tests bootstrap confidence interval estimation."""
    results = bootstrap_stratified_cv_modeling(
        bootstrapped_random_sample_data,
        perturbed_tf_series,
        estimator=estimator,
        ci_percentiles=[95.0, 99.0],
        use_sample_weight_in_cv=False,
    )

    # Ensure result type
    assert isinstance(results, BootstrapModelResults)

    # Validate confidence intervals
    assert isinstance(results.ci_dict, dict)
    assert all(isinstance(v, dict) for v in results.ci_dict.values())

    # Validate bootstrap coefficients
    assert isinstance(results.bootstrap_coefs_df, pd.DataFrame)
    assert (
        results.bootstrap_coefs_df.shape[0]
        == bootstrapped_random_sample_data.n_bootstraps
    )

    # Validate alpha values
    assert isinstance(results.alpha_list, list)
    assert len(results.alpha_list) == bootstrapped_random_sample_data.n_bootstraps


def test_bootstrap_stratified_cv_modeling_with_weights(
    random_sample_data, bootstrapped_random_sample_data
):
    """Tests bootstrap confidence interval estimation."""
    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]
    estimator = LassoCV()
    results = bootstrap_stratified_cv_modeling(
        bootstrapped_random_sample_data,
        perturbed_tf_series,
        estimator=estimator,
        ci_percentiles=[95.0, 99.0],
        use_sample_weight_in_cv=True,
    )

    # Ensure result type
    assert isinstance(results, BootstrapModelResults)

    # Validate confidence intervals
    assert isinstance(results.ci_dict, dict)
    assert all(isinstance(v, dict) for v in results.ci_dict.values())

    # Validate bootstrap coefficients
    assert isinstance(results.bootstrap_coefs_df, pd.DataFrame)
    assert (
        results.bootstrap_coefs_df.shape[0]
        == bootstrapped_random_sample_data.n_bootstraps
    )

    # Validate alpha values
    assert isinstance(results.alpha_list, list)
    assert len(results.alpha_list) == bootstrapped_random_sample_data.n_bootstraps
