import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("main")


def stratification_classification(
    binding_series: pd.Series,
    bins: list = [0, 8, 64, 512, np.inf],
) -> np.ndarray:
    """
    Bin the binding and perturbation data and create groups for stratified k folds.

    :param binding_series: The binding vector to use for stratification
    :param bins: The bins to use for stratification.
        The default is [0, 8, 64, 512, np.inf]

    :return: A numpy array of the stratified classes

    :raises ValueError: If the length of `bins` is less than 2

    """
    # validate the binding data
    if not isinstance(binding_series, pd.Series):
        raise ValueError("The binding vector must be a pandas Series.")
    if binding_series.dtype.kind not in "biufc":
        raise ValueError("The binding vector must be numeric.")

    # if the number of bins is less than 2, return a pd.Series of ones
    if len(bins) < 2:
        logger.warning(
            "The number of bins is less than 2. Returning a series of ones."
            "This will not provide meaningful stratification."
        )
        return np.ones(len(binding_series), dtype=int)

    # create labels for the bins, eg 1, 2, 3, 4 if the bins are [0, 8, 64, 512, np.inf]
    labels = list(range(1, len(bins)))

    # bin the binding data by ranking, and then using the bins and labels to
    # assign rows to bins
    binding_rank = binding_series.rank(method="min", ascending=False).values
    binding_bin = pd.cut(binding_rank, bins=bins, labels=labels, right=True).astype(int)

    # Transform the binding_bin by multiplying it by a scale factor
    # (the number of labels). This ensures that combining
    # (binding_bin - 1) * len(labels) + perturbation_bin
    # produces a unique label for each (binding_bin, perturbation_bin) pair.
    # For example, if binding_bin = 2 and perturbation_bin = 3, and there are 4 labels,
    # then the final label will be (2 - 1) * 4 + 3 = 7.
    # This allows for stratification across both variables simultaneously.
    return binding_bin
