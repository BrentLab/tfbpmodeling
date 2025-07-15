import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from tfbpmodeling.interactor_significance_results import InteractorSignificanceResults
from tfbpmodeling.modeling_input_data import ModelingInputData
from tfbpmodeling.stratified_cv_r2 import stratified_cv_r2

logger = logging.getLogger("main")


def evaluate_interactor_significance_linear(
    input_data: ModelingInputData,
    stratification_classes: np.ndarray,
    model_variables: list[str],
    estimator: BaseEstimator = LinearRegression(fit_intercept=True),
) -> "InteractorSignificanceResults":
    """
    Compare predictive performance of interaction terms vs. their main effects.

    This function performs a stratified cross-validation comparison between:
    - The original model containing interaction terms (e.g., TF1:TF2)
    - A reduced model where each interactor is replaced by its corresponding
      main effect (e.g., TF2)

    R² scores are computed for both models using stratified CV. The delta in R²
    informs whether the interaction term adds predictive value.

    :param input_data: A `ModelingInputData` instance containing predictors
        and response.
    :param stratification_classes: Array of stratification labels for CV.
    :param model_variables: List of model terms, including interaction terms.
    :param estimator: A scikit-learn estimator to use for modeling. Default is
        `LinearRegression(fit_intercept=True)`.

    :return: An `InteractorSignificanceResults` instance with evaluation results.

    :raises KeyError: If a main effect is missing from the input data.

    """
    logger.info("Interactor significance evaluation method: Linear")

    output = []

    response_df = input_data.response_df

    # Identify interaction terms (those with ":")
    interactors = [var for var in model_variables if ":" in var]

    logger.info(f"Testing the following interaction variables: {interactors}")

    # NOTE: add_row_max is set to True such that IF the formula includes row_max,
    # the column is present. However, if the formula doesn't not include row_max,
    # then that column will not be present in the model matrix.
    add_row_max = "row_max" in model_variables
    logger.info(
        "Using 'row_max' in model variables "
        "for evaluate_interactor_significance: %s",
        add_row_max,
    )
    # Get the average R² of the original model
    avg_r2_original_model = stratified_cv_r2(
        response_df,
        input_data.get_modeling_data(
            " + ".join(model_variables), add_row_max=add_row_max
        ),
        stratification_classes,
        estimator=estimator,
    )

    for interactor in interactors:
        # Extract main effect from interactor
        main_effect = interactor.split(":")[1]

        logger.debug(f"Testing interactor '{interactor}' with variant '{main_effect}'.")

        # Ensure main effect exists in predictors
        if main_effect not in input_data.predictors_df.columns:
            raise KeyError(f"Main effect '{main_effect}' not found in predictors.")

        # Define predictor sets for comparison
        predictors_with_main_effect = [
            var for var in model_variables if var != interactor
        ] + [
            main_effect
        ]  # Replace interactor with main effect

        # Get the average R² of the model with the main effect replacing one of the
        # interaction terms
        avg_r2_main_effect = stratified_cv_r2(
            response_df,
            input_data.get_modeling_data(
                " + ".join(predictors_with_main_effect), add_row_max=add_row_max
            ),
            stratification_classes,
            estimator=estimator,
        )

        # Store results
        output.append(
            {
                "interactor": interactor,
                "variant": main_effect,
                "avg_r2_interactor": avg_r2_original_model,
                "avg_r2_main_effect": avg_r2_main_effect,
                "delta_r2": avg_r2_main_effect - avg_r2_original_model,
            }
        )

    return InteractorSignificanceResults(output)
