import json
from typing import Any

import numpy as np
import pandas as pd


class InteractorSignificanceResults:
    """
    Container for storing and analyzing the results of interactor significance testing.

    This class holds evaluations comparing the predictive power of interaction terms
    versus their corresponding main effects in a model, based on cross-validated R².

    Provides methods to:
    - Convert results to a DataFrame.
    - Serialize and deserialize results from disk.
    - Select final model terms by comparing interaction and main effect performance.

    """

    def __init__(self, evaluations: list[dict[str, Any]]):
        """
        Initialize the evaluations object.

        :param evaluations: A list of dictionaries containing significance test results.

        """
        self.evaluations = evaluations

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return evaluations as a Pandas DataFrame.

        :return: DataFrame containing the significance test results.

        """
        return pd.DataFrame(self.evaluations)

    def serialize(self, filepath: str) -> None:
        """
        Save the evaluations to a JSON file.

        :param filepath: Path to the output JSON file.

        """
        with open(filepath, "w") as f:
            json.dump(self.evaluations, f, indent=4)

    @classmethod
    def deserialize(cls, filepath: str) -> "InteractorSignificanceResults":
        """
        Load evaluations from a JSON file.

        :param filepath: Path to the JSON file containing evaluation results.
        :return: An instance of `InteractorSignificanceResults`.

        :raises ValueError: If the JSON content is not a list.

        """
        with open(filepath) as f:
            evaluations = json.load(f)

        if not isinstance(evaluations, list):
            raise ValueError(
                f"Invalid JSON format: Expected a list, got {type(evaluations)}"
            )

        return cls(evaluations)

    def final_model(self) -> list[str]:
        """
        Select the preferred model terms based on R² comparison.

        For each interactor, compares R² of the full model (with interaction term) to
        that of a model where the interactor is replaced by its main effect. Whichever
        yields higher R² is retained.

        :return: List of selected model terms (interactor or main effect).

        """
        df = self.to_dataframe()

        if df.empty:
            return []

        # Select either the interactor or the variant based on max R²
        df["selected"] = np.where(
            df["avg_r2_interactor"] >= df["avg_r2_main_effect"],
            df["interactor"],
            df["variant"],
        )

        return df["selected"].tolist()
