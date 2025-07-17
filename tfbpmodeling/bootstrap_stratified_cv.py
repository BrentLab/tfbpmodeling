import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold

from tfbpmodeling.bootstrap_model_results import BootstrapModelResults
from tfbpmodeling.bootstrapped_input_data import BootstrappedModelingInputData
from tfbpmodeling.stratification_classification import stratification_classification
from tfbpmodeling.stratified_cv import stratified_cv_modeling

logger = logging.getLogger("main")


def bootstrap_stratified_cv_modeling(
    bootstrapped_data: BootstrappedModelingInputData,
    perturbed_tf_series: pd.Series,
    estimator: BaseEstimator = LassoCV(
        fit_intercept=True,
        max_iter=10000,
        selection="random",
        random_state=42,
        n_jobs=4,
    ),
    ci_percentiles: list[int | float] = [95.0, 99.0],
    **kwargs,
) -> BootstrapModelResults:
    """
    Perform bootstrapped stratified CV modeling and estimate confidence intervals for
    model coefficients.

    This function fits a model (e.g., LassoCV) to multiple bootstrap samples drawn
    from `BootstrappedModelingInputData`, using stratified cross-validation to
    select regularization parameters. Confidence intervals are computed across the
    bootstrapped coefficient estimates.

    :param bootstrapped_data: Bootstrapped samples of predictors and response data.
    :param perturbed_tf_series: Series of TF binding values for stratification.
    :param estimator: scikit-learn estimator. Must support `.fit()` with `sample_weight`
        and allow setting `.cv`. Default is `LassoCV`.
    :param ci_percentiles: List of confidence intervals (e.g., [95.0, 99.0]).
    :params kwargs: Additional keyword arguments. The following are supported:
        - bins: Default is `[0, 8, 64, 512, np.inf]`. List of bin edges
            for stratification

    :return: A BootstrapModelResults object containing:
        - `ci_dict`: Dict of CI bounds per feature per percentile
        - `bootstrap_coefs_df`: DataFrame of coefficients across bootstrap samples
        - `alpha_list`: List of selected regularization parameters per model

    :raises ValueError: If inputs are incompatible or misformatted.
    :raises KeyError: If class assignment for stratification fails.

    """
    n_samples = bootstrapped_data.model_df.shape[0]
    if n_samples <= 100:
        raise ValueError(f"Need >100 target_symbols for stratified CV, got {n_samples}")

    if not isinstance(bootstrapped_data, BootstrappedModelingInputData):
        raise ValueError(
            "bootstrapped_data must be an instance of BootstrappedModelingInputData."
        )

    if estimator is None:
        raise ValueError("An estimator must be provided. Default is LassoCV().")

    # Validate ci_percentiles
    if not isinstance(ci_percentiles, list) or not all(
        isinstance(x, (int, float)) and 0 < x < 100 for x in ci_percentiles
    ):
        raise ValueError(
            "ci_percentiles must be a list of integers or floats between 0 and 100."
        )

    # validate that the index of the response_df and model_df match
    if not bootstrapped_data.response_df.index.equals(bootstrapped_data.model_df.index):
        raise ValueError(
            "The index of the response_df must match the index of the model_df."
        )

    # validate that the index of the perturbed_tf_series matches the index of the
    # model_df
    if not bootstrapped_data.model_df.index.equals(perturbed_tf_series.index):
        raise ValueError(
            "The index of the perturbed TF series must match the index of the model_df."
        )

    # log the dimension of the response and model frames
    logger.info(f"Response frame shape: {bootstrapped_data.response_df.shape}")
    logger.info(f"Model frame shape: {bootstrapped_data.model_df.shape}")

    # log the columns of the model_df
    logger.info(f"Model frame columns: {bootstrapped_data.model_df.columns}")

    # set the bin splits
    bins = kwargs.pop("bins", [0, 8, 64, 512, np.inf])
    # log the bins used for stratification
    logger.info(f"Using the following stratification bins: {bins}.")

    # initialize lists to store bootstrap results
    bootstrap_coefs = []
    alpha_list = []

    # shuffle = True means that the partitioning is random.
    # NOTE: In each iteration, the random state is updated to the current
    # bootstrap iteration index. This ensures that the randomization is
    # reproducible across different runs of the function, while still allowing
    # for variability in how each bootstrap sample is partitioned into train/test
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Bootstrap iterations
    logger.info("Starting bootstrap modeling iterations...")
    for index, (_, sample_weight) in enumerate(bootstrapped_data):
        logger.debug("Bootstrap iteration index: %d", index)

        # Set the random state for StratifiedKFold to the current index
        skf.random_state = index

        # the random_state for the estimator is used to choose among equally good
        # variables. I'm not sure how much this affects results -- we are making
        # a distribution of coefficients rather than letting sklearn choose a
        # model for us -- but it is, similar to StratifiedKFold above, randomized
        # but reproducible by setting random_state to the bootstrap iteration
        try:
            estimator.random_state = index
        except AttributeError:
            logger.warning("Estimator does not have a random_state attribute.")
            pass

        classes = stratification_classification(
            perturbed_tf_series.loc[bootstrapped_data.response_df.index].squeeze(),
            bins=bins,
        )

        model_i = stratified_cv_modeling(
            bootstrapped_data.response_df,
            bootstrapped_data.model_df,
            classes=classes,
            estimator=estimator,
            skf=skf,
            sample_weight=sample_weight,
        )

        alpha_list.append(model_i.alpha_)
        bootstrap_coefs.append(model_i.coef_)
    # Convert bootstrap coefficients to DataFrame
    bootstrap_coefs_df = pd.DataFrame(
        bootstrap_coefs, columns=bootstrapped_data.model_df.columns
    )

    # Compute confidence intervals
    ci_dict = {
        f"{ci}": {
            colname: (
                np.percentile(bootstrap_coefs_df[colname], (100 - ci) / 2),
                np.percentile(bootstrap_coefs_df[colname], 100 - (100 - ci) / 2),
            )
            for colname in bootstrap_coefs_df.columns
        }
        for ci in ci_percentiles
    }

    return BootstrapModelResults(
        ci_dict=ci_dict,
        bootstrap_coefs_df=bootstrap_coefs_df,
        alpha_list=alpha_list,
    )
