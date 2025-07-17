from sklearn.linear_model import LinearRegression

from tfbpmodeling.stratification_classification import stratification_classification
from tfbpmodeling.stratified_cv_r2 import stratified_cv_r2


#  Testing `stratified_cv_r2()`
def test_stratified_cv_r2(random_sample_data, bootstrapped_random_sample_data):
    """Tests stratified cross-validation R^2 calculation."""
    response_df = bootstrapped_random_sample_data.response_df
    model_df = bootstrapped_random_sample_data.model_df

    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]

    classes = stratification_classification(
        perturbed_tf_series.loc[response_df.index].squeeze(),
    )

    r2_value = stratified_cv_r2(
        response_df,
        model_df,
        classes,
        estimator=LinearRegression(),
    )

    # Ensure valid R^2 output
    assert isinstance(r2_value, float)
    assert -1.0 <= r2_value <= 1.0  # R² should be within a reasonable range
