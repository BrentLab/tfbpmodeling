import json
import logging

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state, resample

logger = logging.getLogger("main")


class BootstrappedModelingInputData:
    """
    This class handles bootstrapped resampling of a response vector and model matrix.

    This class supports both on-the-fly generation and externally provided bootstrap
    indices. For each bootstrap sample, it maintains sample weights derived from
    frequency counts of resampled instances.

    """

    def __init__(
        self,
        response_df: pd.DataFrame,
        model_df: pd.DataFrame,
        n_bootstraps: int,
        normalize_sample_weights: bool = True,
        random_state: int | None = None,
    ) -> None:
        """
        Initialize bootstrapped modeling input.

        Either `n_bootstraps` or `bootstrap_indices` must be provided.

        :param response_df: Response variable.
        :param model_df: Predictor matrix.
        :param n_bootstraps: Number of bootstrap replicates to generate.
        :param random_state: Random state for reproducibility. Can be an integer or a
            numpy RandomState object, or None. If None (default), then a random
            random state is chosen.

        :raises ValueError: if the response_df and model_df do not have the same index
            or if arguments are not correct datatype.

        """

        self.response_df: pd.DataFrame = response_df
        self.model_df: pd.DataFrame = model_df
        if not response_df.index.equals(model_df.index):
            raise IndexError("response_df and model_df must have the same index order.")
        self.normalize_sample_weights = normalize_sample_weights

        # If bootstrap_indices is provided, set n_bootstraps based on its length
        self.n_bootstraps = n_bootstraps

        # set the random number generator attribute
        self.random_state = random_state
        self._rng = check_random_state(self.random_state)
        logger.info(
            f"Using random state: {self.random_state}"
            if self.random_state is not None
            else "No explicit random state set."
        )

        # Initialize attributes
        self._bootstrap_indices: list[np.ndarray] = []
        self._sample_weights: dict[int, np.ndarray] = {}

        self._generate_bootstrap_indices()

    @property
    def response_df(self) -> pd.DataFrame:
        """
        Get the response DataFrame.

        :return: The response DataFrame.

        """
        return self._response_df

    @response_df.setter
    def response_df(self, value: pd.DataFrame) -> None:
        """
        Set the response DataFrame and validate it.

        :param value: The response DataFrame to set.
        :raises TypeError: if the input is not a dataframe
        :raises IndexError: if the input dataframe has an empty index.

        """
        if not isinstance(value, pd.DataFrame):
            raise TypeError("response_df must be a DataFrame.")
        if value.index.empty:
            raise IndexError("response_df must have a non-empty index.")
        self._response_df = value

    @property
    def model_df(self) -> pd.DataFrame:
        """
        Get the model DataFrame.

        :return: The model DataFrame.

        """
        return self._model_df

    @model_df.setter
    def model_df(self, value: pd.DataFrame) -> None:
        """
        Set the model DataFrame and validate it.

        :param value: The model DataFrame to set.
        :raises TypeError: if the input is not a dataframe
        :raises IndexError: if the input dataframe has an empty index.

        """
        if not isinstance(value, pd.DataFrame):
            raise TypeError("model_df must be a DataFrame.")
        if value.index.empty:
            raise IndexError("model_df must have a non-empty index.")
        self._model_df = value

    @property
    def random_state(self) -> int | None:
        """
        An integer used to set the random state when generating the bootstrap samples.

        Set this explicitly for reproducibility

        """
        return self._random_state

    @random_state.setter
    def random_state(self, value: int | None) -> None:
        if not isinstance(value, (int, type(None))):
            raise TypeError("random state must be an integer or None")
        self._random_state = value

    @property
    def n_bootstraps(self) -> int:
        """
        Get the number of bootstrap samples.

        :return: The number of bootstrap samples.

        """
        return self._n_bootstraps

    @n_bootstraps.setter
    def n_bootstraps(self, value: int) -> None:
        """
        Set the number of bootstrap samples.

        :param value: The number of bootstrap samples to generate.
        :raises TypeError: If the value is not a positive integer.

        """
        if not isinstance(value, int) or value <= 0:
            raise TypeError("n_bootstraps must be a positive integer.")
        logger.info(f"Number of bootstrap samples set to: {value}")
        self._n_bootstraps = value

    @property
    def bootstrap_indices(self) -> list[np.ndarray]:
        """A list of arrays representing bootstrap sample indices."""
        return self._bootstrap_indices

    @bootstrap_indices.setter
    def bootstrap_indices(self, value: list[np.ndarray]) -> None:
        """
        Set bootstrap sample indices and compute sample weights.

        :param value: A list of arrays containing valid index values from `response_df`.

        :raises ValueError: If indices are invalid or contain out-of-range values.

        """
        if not isinstance(value, list) or not all(
            isinstance(indices, np.ndarray) for indices in value
        ):
            raise TypeError("bootstrap_indices must be a list of numpy arrays.")

        valid_indices = set(self.response_df.index)
        for i, indices in enumerate(value):
            if not set(indices).issubset(valid_indices):
                raise IndexError(
                    f"Bootstrap sample {i} contains invalid "
                    "indices not found in response_df."
                )

        self._bootstrap_indices = value
        self._compute_sample_weights()

    @property
    def normalize_sample_weights(self) -> bool:
        """
        Get the normalization status for sample weights.

        :return: True if sample weights are normalized, False otherwise.

        """
        return self._normalize_sample_weights

    @normalize_sample_weights.setter
    def normalize_sample_weights(self, value: bool) -> None:
        """
        Set the normalization status for sample weights.

        :param value: Boolean indicating whether to normalize sample weights.
        :raises TypeError: If the input is not a boolean.

        """
        if not isinstance(value, bool):
            raise TypeError("normalize_sample_weights must be a boolean.")
        logger.info(f"Sample weights normalization set to: {value}")
        self._normalize_sample_weights = value

    @property
    def sample_weights(self) -> dict[int, np.ndarray]:
        """
        Normalized sample weights corresponding to bootstrap samples.

        :return: A dictionary mapping bootstrap index to sample weights.

        """
        return self._sample_weights

    @sample_weights.setter
    def sample_weights(self, value: dict[int, np.ndarray]) -> None:
        """
        Set the sample weights directly.

        :param value: Dictionary mapping index to numpy arrays of weights.
        :raises TypeError: If the input is not a dictionary of int to ndarray.

        """
        if not isinstance(value, dict) or not all(
            isinstance(k, int) and isinstance(v, np.ndarray) for k, v in value.items()
        ):
            raise TypeError("sample_weights must be a dictionary of numpy arrays.")
        self._sample_weights = value

    def _generate_bootstrap_indices(self) -> None:
        """Generates bootstrap sample indices and sample weights."""
        y_indices: np.ndarray = self.response_df.index.to_numpy()

        self._bootstrap_indices = [
            resample(y_indices, replace=True, random_state=self._rng)
            for _ in range(self.n_bootstraps)
        ]
        self._compute_sample_weights()

    def _compute_sample_weights(self) -> None:
        """
        Computes sample weights from existing bootstrap indices.

        :param integer: If False, the sample weights are divided by the number of
            observations in the response_df. If True, the sample weights are not divided
            by the number of observations.

        """
        y_indices: np.ndarray = self.response_df.index.to_numpy()
        sample_weights: dict[int, np.ndarray] = {}

        logger.info(
            f"Sample weights normalization method: {self.normalize_sample_weights}"
        )

        for i, sample in enumerate(self._bootstrap_indices):
            index_mapping = {label: idx for idx, label in enumerate(y_indices)}
            integer_indices = np.array([index_mapping[label] for label in sample])
            sample_counts = np.bincount(integer_indices, minlength=len(y_indices))

            if self.normalize_sample_weights:
                # note sample_counts.sum() == len(y_indices) in this case, but
                # sample_counts.sum() seems to be more canonical
                sample_weights[i] = sample_counts / sample_counts.sum()
            else:
                sample_weights[i] = sample_counts

        self._sample_weights = sample_weights

    def get_bootstrap_sample(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves a bootstrap sample by index.

        :param i: Bootstrap sample index.
        :return: Tuple of (sample_indices, sample_weights).
        :raises IndexError: If the index exceeds the number of bootstraps.

        """
        if i >= self.n_bootstraps or i < 0:
            raise IndexError(
                f"Bootstrap index {i} out of range. Max: {self.n_bootstraps - 1}"
            )

        sampled_indices = self.bootstrap_indices[i]
        sample_weights = self.get_sample_weight(i)

        return (
            sampled_indices,
            sample_weights,
        )

    def get_sample_weight(self, i: int) -> np.ndarray:
        """
        Retrieves sample weights for a bootstrap sample.

        :param i: Bootstrap sample index.
        :return: Array of sample weights.

        """
        if i >= self.n_bootstraps or i < 0:
            raise IndexError(f"Sample weight index {i} out of range.")
        return self.sample_weights[i]

    def regenerate(self) -> None:
        """
        Re-generate, randomly, bootstrap samples and sample weights.

        This should be called if the response or predictors change.

        """
        self._generate_bootstrap_indices()

    def save_indices(self, filename: str) -> None:
        """
        Saves only the bootstrap indices to a JSON file.

        Saves the bootstrap indices to a JSON file. This can be used to persist the
        bootstrap indices for later use, allowing for reproducibility in analyses.

        :param filename: Path to the JSON file where the bootstrap indices will be
            saved. This will overwrite the file if it exists.

        """
        data = {
            "n_bootstraps": self.n_bootstraps,
            "bootstrap_indices": [
                indices.tolist() for indices in self._bootstrap_indices
            ],
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    def serialize(self, filename: str) -> None:
        """
        Saves the object as a JSON file.

        Serializes the current state of the BootstrappedModelingInputData object to a
        JSON file, including the response and model DataFrames, number of bootstraps,
        bootstrap indices, and sample weights.

        :param filename: Path to the JSON file where the object will be saved.
        :raises ValueError: If the filename is not a valid path or if the object cannot
            be serialized. This method will overwrite the file if it exists.

        """
        data = {
            "response_df": self.response_df.to_dict(orient="split"),
            "index_name": self.response_df.index.name,
            "model_df": self.model_df.to_dict(orient="split"),
            "n_bootstraps": self.n_bootstraps,
            "normalize_sample_weights": self.normalize_sample_weights,
            "random_state": self.random_state,
        }

        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def deserialize(cls, filename: str):
        """
        Loads the object from a JSON file.

        :param filename: Path to the BootstrapModelingData JSON file.

        """
        with open(filename) as f:
            data = json.load(f)

        response_df = pd.DataFrame(**data["response_df"]).rename_axis(
            index=data["index_name"]
        )
        model_df = pd.DataFrame(**data["model_df"]).rename_axis(
            index=data["index_name"]
        )
        n_bootstraps = data["n_bootstraps"]

        normalize_sample_weights = data["normalize_sample_weights"]

        random_state = data["random_state"]

        instance = cls(
            response_df,
            model_df,
            n_bootstraps,
            normalize_sample_weights=normalize_sample_weights,
            random_state=random_state,
        )

        return instance

    def __iter__(self):
        """Resets the iterator and returns itself."""
        self._current_index = 0
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Provides the next bootstrap sample for iteration.

        :return: Tuple of (sample_indices, sample_weights).
        :raises StopIteration: When all bootstrap samples are exhausted.

        """
        if self._current_index >= self.n_bootstraps:
            raise StopIteration

        sample_indices, sample_weights = self.get_bootstrap_sample(self._current_index)

        self._current_index += 1
        return sample_indices, sample_weights
