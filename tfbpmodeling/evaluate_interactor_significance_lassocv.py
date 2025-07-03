import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

from tfbpmodeling.interactor_significance_results import InteractorSignificanceResults
from tfbpmodeling.modeling_input_data import ModelingInputData
from tfbpmodeling.stratified_cv import stratified_cv_modeling

logger = logging.getLogger("main")


def evaluate_interactor_significance_lassocv(
    input_data: ModelingInputData,
    stratification_classes: np.ndarray,
    model_variables: list[str],
    estimator: BaseEstimator = LassoCV(
        fit_intercept=True,
        max_iter=10000,
        selection="random",
        random_state=42,
        n_jobs=4,
    ),
) -> "InteractorSignificanceResults":
    """
    Evaluate which interaction terms survive LassoCV when main effects are included.

    :return:
        - List of retained interaction terms
        - pd.Series of all model coefficients (indexed by term name)
        - Selected alpha value from LassoCV

    """
    logger.info("Interactor significance evaluation method: LassoCV")

    interactors = [v for v in model_variables if ":" in v]
    modifier_main_effects = {i.split(":")[1] for i in interactors}

    augmented_vars = list(set(model_variables + list(modifier_main_effects)))
    logger.info(
        f"Model includes interaction terms and their main effects: {augmented_vars}"
    )
    add_row_max = "row_max" in augmented_vars
    logger.info(
        "Using 'row_max' in model variables "
        "for evaluate_interactor_significance: %s",
        add_row_max,
    )

    X = input_data.get_modeling_data(
        " + ".join(augmented_vars),
        add_row_max=add_row_max,
        drop_intercept=True,
    )
    y = input_data.response_df

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    model_i = stratified_cv_modeling(
        y,
        X,
        classes=stratification_classes,
        estimator=estimator,
        skf=skf,
    )

    coefs = pd.Series(model_i.coef_, index=X.columns)
    retained_vars = coefs[coefs != 0].index.tolist()
    retained_interactors = [v for v in retained_vars if ":" in v]

    logger.info(f"Retained interaction terms: {retained_interactors}")
    y_pred = model_i.predict(X)
    r2_full_model = r2_score(y, y_pred)

    output = []
    for interactor in interactors:
        main_effect = interactor.split(":")[1]
        output.append(
            {
                "interactor": interactor,
                "variant": main_effect,
                "r2_lasso_model": r2_full_model,
                "coef_interactor": coefs.get(interactor, 0.0),
                "coef_main_effect": coefs.get(main_effect, 0.0),
            }
        )

    return InteractorSignificanceResults(output)
