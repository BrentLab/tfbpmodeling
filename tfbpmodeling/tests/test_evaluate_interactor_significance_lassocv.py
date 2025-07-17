import pytest
from sklearn.linear_model import LassoCV

from tfbpmodeling.bootstrap_stratified_cv import bootstrap_stratified_cv_modeling
from tfbpmodeling.evaluate_interactor_significance_lassocv import (
    evaluate_interactor_significance_lassocv,
)
from tfbpmodeling.interactor_significance_results import InteractorSignificanceResults
from tfbpmodeling.stratification_classification import stratification_classification


# Testing `evaluate_interactor_significance_lassocv()`
@pytest.mark.parametrize("top_n_masked", [True, False])
def test_evaluate_interactor_significance_lassocv(
    random_sample_data, bootstrapped_random_sample_data, caplog
):
    """Tests LassoCV-based evaluation of interactor significance."""

    # Step 1: Run initial bootstrap to extract significant coefficients
    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]
    estimator = LassoCV(random_state=42, max_iter=5000)

    init_results = bootstrap_stratified_cv_modeling(
        bootstrapped_random_sample_data,
        perturbed_tf_series,
        estimator=estimator,
        ci_percentiles=[95.0],
        use_sample_weight_in_cv=True,
    )

    # Step 2: Create stratification classes
    classes = stratification_classification(
        perturbed_tf_series.loc[random_sample_data.response_df.index].squeeze(),
    )

    # Step 3: Evaluate LassoCV interactor significance
    with caplog.at_level("INFO"):
        results = evaluate_interactor_significance_lassocv(
            random_sample_data,
            stratification_classes=classes,
            model_variables=list(
                init_results.extract_significant_coefficients().keys()
            ),
            estimator=estimator,
        )
        assert (
            "Using 'row_max' in model variables for "
            "evaluate_interactor_significance: False" in caplog.text
        )

    # Step 4: Validate results
    assert isinstance(results, InteractorSignificanceResults)
    df = results.to_dataframe()
    assert not df.empty
    assert "interactor" in df.columns
    assert "coef_interactor" in df.columns
    assert "coef_main_effect" in df.columns
    assert "r2_lasso_model" in df.columns
