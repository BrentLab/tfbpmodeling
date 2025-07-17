import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("main")


class BootstrapModelResults:
    """
    Encapsulates the results from bootstrapped stratified cross-validation modeling.

    This includes:
    - Confidence intervals for model coefficients across bootstrap iterations
    - Raw coefficient estimates from each iteration
    - Alpha values (regularization strengths) selected during each iteration
    - Methods for extracting statistically significant coefficients
    - Visualization utilities
    - Serialization and deserialization support

    """

    def __init__(
        self,
        ci_dict: dict[str, dict[str, tuple[float, float]]],
        bootstrap_coefs_df: pd.DataFrame,
        alpha_list: list[float],
        alpha_df: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Initialize BootstrapModelResults.

        :param ci_dict: Nested dictionary mapping confidence levels to (low, high)
            confidence intervals for each coefficient.
        :param bootstrap_coefs_df: DataFrame of shape (n_bootstraps, n_features)
            containing coefficient values from each bootstrap sample.
        :param alpha_list: List of alpha values (regularization strength) selected
            during each bootstrap iteration.
        :param alpha_df: a dataframe with the columns 'bootstrap_idx', 'alpha', 'fold',
            and 'mse'

        """
        self.ci_dict = ci_dict
        self.bootstrap_coefs_df = bootstrap_coefs_df
        self.alpha_list = alpha_list
        self.alpha_df = alpha_df

    def extract_significant_coefficients(
        self, ci_level: str = "95.0", threshold: float = 1e-15
    ) -> dict[str, tuple[float, float]]:
        """
        Extract coefficients that are statistically significant based on their bootstrap
        confidence intervals.

        A coefficient is considered significant if its entire confidence interval
            lies above `threshold` or below `-threshold`.

        :param ci_level: Confidence interval level (e.g., "95.0").
        :param threshold: Minimum effect size for significance.
        :return: Dictionary mapping coefficient names to their (low, high) CI bounds.

        """
        ci_dict_local = self.ci_dict.copy()

        # If CI level is not precomputed, calculate it
        if ci_level not in ci_dict_local:
            ci_level_numeric = float(ci_level)
            # log that the ci_level is not in the ci_dict
            logger.debug(
                f"Generating confidence intervals for ci level: {ci_level_numeric}"
            )
            ci_dict_local[ci_level] = {
                colname: (
                    np.percentile(
                        self.bootstrap_coefs_df[colname], (100 - ci_level_numeric) / 2
                    ),
                    np.percentile(
                        self.bootstrap_coefs_df[colname],
                        100 - (100 - ci_level_numeric) / 2,
                    ),
                )
                for colname in self.bootstrap_coefs_df.columns
            }

        # Select significant coefficients based on the confidence interval threshold
        significant_coefs_dict = {
            coef: bounds
            for coef, bounds in ci_dict_local[ci_level].items()
            if bounds[0] > threshold or bounds[1] < -threshold
        }

        # remove the following terms from ci_dict:
        keys_to_remove = [
            "bootstrap_idx",
            "final_training_score",
            "alpha",
            "left_asymptote",
            "right_asymptote",
            "Intercept",
        ]
        for key in keys_to_remove:
            significant_coefs_dict.pop(key, None)

        return significant_coefs_dict

    def visualize_significant_coefficients(
        self, ci_level: str = "95.0", threshold: float = 0.0
    ) -> plt.Figure | None:
        """
        Visualize the distribution of coefficients that are significant at the specified
        confidence level.

        :param ci_level: Confidence interval level (e.g., "95.0").
        :param threshold: Minimum absolute value for significance.
        :return: Matplotlib figure, or None if no significant coefficients are found.

        """
        significant_coefs = self.extract_significant_coefficients(ci_level, threshold)

        if not significant_coefs:
            print(
                f"No significant coefficients found for CI {ci_level} "
                "at threshold {threshold}."
            )
            return None

        # Extract relevant coefficients for plotting
        df_extracted = self.bootstrap_coefs_df[list(significant_coefs.keys())]

        # Create the boxplot
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_extracted, orient="h")
        plt.axvline(x=0, linestyle="--", color="black")
        plt.xlabel("Coefficient Values")
        plt.title(f"Coefficients with {ci_level}% CI outside ±{threshold}")

        return fig

    @staticmethod
    def truncate_at_threshold(val: float, threshold: float = 1e-15) -> float:
        return 0.0 if abs(val) < threshold else val

    def serialize(self, filename: str, output_dir: str | None = None) -> None:
        """
        Save the results to disk.

        Creates two files:
        - `{filename}.json`: confidence intervals
        - `{filename}.pkl`: tuple of (bootstrap_coefs_df, alpha_list)

        :param filename: Base filename (without extension).
        :param output_dir: Optional directory to write files into. Uses current
            directory if not specified.

        :raises FileNotFoundError: If the specified directory does not exist.

        """
        # Validate that the output directory exists
        if output_dir:
            if not os.path.isdir(output_dir):
                raise FileNotFoundError(
                    f"The output directory '{output_dir}' does not exist. "
                    "Please create it before saving."
                )
            filepath_json = os.path.join(output_dir, f"{filename}.json")
            filepath_pkl = os.path.join(output_dir, f"{filename}.pkl")
        else:
            filepath_json = f"{filename}.json"
            filepath_pkl = f"{filename}.pkl"

        # Save confidence intervals as JSON
        ci_dict_trunc = {}
        for level, intervals in self.ci_dict.items():
            ci_dict_trunc[level] = {
                coef: (
                    self.truncate_at_threshold(bounds[0]),
                    self.truncate_at_threshold(bounds[1]),
                )
                for coef, bounds in intervals.items()
            }
        with open(filepath_json, "w") as f:
            json.dump(ci_dict_trunc, f, indent=4)

        # Save DataFrame and alpha_list as a Pickle file
        with open(filepath_pkl, "wb") as f:
            pickle.dump((self.bootstrap_coefs_df, self.alpha_list), f)

    @classmethod
    def deserialize(
        cls, ci_dict_json: str, coefs_alphas_pkl: str
    ) -> "BootstrapModelResults":
        """
        Load model results from disk.

        :param ci_dict_json: Path to the JSON file with confidence intervals.
        :param coefs_alphas_pkl: Path to the Pickle file with coefficient matrix and
            alpha list.
        :return: A new BootstrapModelResults instance.
        :raises FileNotFoundError: If either file is missing.
        :raises ValueError: If the pickle file contents are invalid.

        """
        # Ensure both files exist before proceeding
        if not os.path.exists(ci_dict_json):
            raise FileNotFoundError(
                f"Confidence intervals file '{ci_dict_json}' not found."
            )
        if not os.path.exists(coefs_alphas_pkl):
            raise FileNotFoundError(f"Pickle file '{coefs_alphas_pkl}' not found.")

        # Load confidence intervals from JSON
        with open(ci_dict_json) as f:
            ci_dict = json.load(f)

        # Load DataFrame and alpha_list from Pickle
        with open(coefs_alphas_pkl, "rb") as f:
            loaded_data = pickle.load(f)

        # Validate loaded data
        if not isinstance(loaded_data, tuple) or len(loaded_data) != 2:
            raise ValueError(
                "Pickle file does not contain expected (DataFrame, list) format."
            )

        bootstrap_coefs_df, alpha_list = loaded_data

        return cls(ci_dict, bootstrap_coefs_df, alpha_list)

    @classmethod
    def from_jsonl(
        cls,
        db_path: str,
        bootstrap_results_table_name: str = "bootstrap_results",
        mse_table_name: str = "mse_path",
    ) -> "BootstrapModelResults":
        """
        Load bootstrap results from JSONL files. This is intended to be used with the
        sigmoid bootstrap results.

        :param db_path: Path to the directory containing the JSONL files for a given
            regulator
        :param bootstrap_results_table_name: Name of the JSONL file containing bootstrap
            coefficient/final model results
        :param mse_table_name: Name of the JSONL file containing fold-wise MSE results
            by bootstrap_idx/alpha
        :return: An instance of BootstrapModelResults
        :raises FileNotFoundError: If the JSONL files do not exist.

        """
        bootstrap_coef_results_path = os.path.join(
            db_path, f"{bootstrap_results_table_name}.jsonl"
        )
        mse_path = os.path.join(db_path, f"{mse_table_name}.jsonl")

        if not os.path.isfile(bootstrap_coef_results_path):
            raise FileNotFoundError(
                f"Results file not found: {bootstrap_coef_results_path}"
            )
        if not os.path.isfile(mse_path):
            raise FileNotFoundError(f"Results file not found: {mse_path}")

        results_rows = []
        with open(bootstrap_coef_results_path) as f:
            for line in f:
                try:
                    results_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not results_rows:
            raise ValueError("No valid records found in the results JSONL file.")

        bootstrap_coef_results_df = pd.DataFrame(results_rows)

        # Handle optional MSE file
        mse_rows = []
        with open(mse_path) as f:
            for line in f:
                try:
                    mse_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        alpha_df = pd.DataFrame(mse_rows) if mse_rows else pd.DataFrame()

        return cls(
            ci_dict={},
            bootstrap_coefs_df=bootstrap_coef_results_df,
            alpha_list=[],
            alpha_df=alpha_df,
        )
