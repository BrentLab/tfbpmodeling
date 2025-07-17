import pytest
from sklearn.linear_model import LassoCV

from tfbpmodeling.bootstrap_stratified_cv import bootstrap_stratified_cv_modeling
from tfbpmodeling.evaluate_interactor_significance_linear import (
    evaluate_interactor_significance_linear,
)
from tfbpmodeling.interactor_significance_results import InteractorSignificanceResults
from tfbpmodeling.stratification_classification import stratification_classification


# Testing `evaluate_interactor_significance()`
@pytest.mark.parametrize("top_n_masked", [True, False])
def test_evaluate_interactor_significance_linear(
    random_sample_data, bootstrapped_random_sample_data_factory, top_n_masked, caplog
):
    """Tests evaluation of interactor significance."""
    random_sample_data.top_n_masked = top_n_masked
    bootstrapped_random_sample_data = bootstrapped_random_sample_data_factory(
        random_sample_data
    )

    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]
    estimator = LassoCV()

    init_results = bootstrap_stratified_cv_modeling(
        bootstrapped_random_sample_data,
        perturbed_tf_series,
        estimator=estimator,
        ci_percentiles=[95.0, 99.0],
    )

    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]

    classes = stratification_classification(
        perturbed_tf_series.loc[random_sample_data.response_df.index].squeeze(),
    )

    with caplog.at_level("INFO"):

        results = evaluate_interactor_significance_linear(
            random_sample_data,
            stratification_classes=classes,
            model_variables=list(
                init_results.extract_significant_coefficients().keys()
            ),
        )
        assert (
            "Using 'row_max' in model variables for "
            "evaluate_interactor_significance: False" in caplog.text
        )

    # Ensure the results contain expected keys
    assert isinstance(results, InteractorSignificanceResults)
    # assert all("interactor" in entry and "avg_r2" in entry for entry in results)

    # # Validate interactor match
    # assert results[0]["interactor"] == interactor

    # Ensure R² value is within valid range
    # assert -1.0 <= results[0]["avg_r2"] <= 1.0
