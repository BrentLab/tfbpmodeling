import json

import pandas as pd

from tfbpmodeling.bootstrap_model_results import BootstrapModelResults


def test_bootstrapmodelresult_from_jsonl(tmp_path):

    # Create dummy bootstrap results JSONL
    bootstrap_results = [
        {
            "bootstrap_idx": 0,
            "alpha": 0.1,
            "final_training_score": 0.95,
            "left_asymptote": 0.0,
            "right_asymptote": 1.0,
            "Intercept": 0.5,
            "TF1": 0.2,
            "TF2": -0.3,
        }
    ]
    mse_results = [
        {"bootstrap_idx": 0, "alpha": 0.1, "fold": 0, "mse": 0.25},
        {"bootstrap_idx": 0, "alpha": 0.1, "fold": 1, "mse": 0.20},
    ]

    # Write to temp JSONL files
    bootstrap_path = tmp_path / "bootstrap_results.jsonl"
    mse_path = tmp_path / "mse_path.jsonl"

    with open(bootstrap_path, "w") as f:
        for entry in bootstrap_results:
            f.write(json.dumps(entry) + "\n")

    with open(mse_path, "w") as f:
        for entry in mse_results:
            f.write(json.dumps(entry) + "\n")

    # Run from_jsonl
    result = BootstrapModelResults.from_jsonl(str(tmp_path))

    # Assertions
    assert isinstance(result.bootstrap_coefs_df, pd.DataFrame)
    assert not result.bootstrap_coefs_df.empty
    assert isinstance(result.alpha_df, pd.DataFrame)
    assert not result.alpha_df.empty
    assert result.alpha_list == []
    assert "TF1" in result.bootstrap_coefs_df.columns
    assert "mse" in result.alpha_df.columns
