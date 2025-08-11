import json

import pandas as pd

from tfbpmodeling.generate_stage_result_table import generate_stage_result_table


def test_generate_stage_result_table(tmp_path):
    # Create dummy tf_dir structure
    tf_dir = tmp_path / "TF1"
    tf_dir.mkdir()

    # all_data_result_object/result_obj.json
    all_data_dir = tf_dir / "all_data_result_object"
    all_data_dir.mkdir()
    all_data_result = {
        "stage_results": {
            "TF1:VAR1": [0.1, 0.2],
            "TF1:VAR2": [-0.2, -0.1],
        }
    }
    with open(all_data_dir / "result_obj.json", "w") as f:
        json.dump(all_data_result, f)

    # topn_result_object/result_obj.json
    topn_dir = tf_dir / "topn_result_object"
    topn_dir.mkdir()
    topn_result = {
        "stage_results": {
            "TF1:VAR1": [0.0, 0.0],  # zero
        }
    }
    with open(topn_dir / "result_obj.json", "w") as f:
        json.dump(topn_result, f)

    # interactor_vs_main_result.json
    main_effect_data = [
        {
            "interactor": "TF1:VAR1",
            "variant": "VAR1",
            "coef_main_effect": 0.5,
            "coef_interactor": -0.3,
        },
        {
            "interactor": "TF1:VAR2",
            "variant": "VAR2",
            "coef_main_effect": -0.1,
            "coef_interactor": 0.05,
        },
    ]
    with open(tf_dir / "interactor_vs_main_result.json", "w") as f:
        json.dump(main_effect_data, f)

    # Run function
    generate_stage_result_table(str(tf_dir))

    # Read generated CSV
    csv_path = tf_dir / "stage_result_table.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert "predictor" in df.columns
    assert "all_data" in df.columns
    assert "topn" in df.columns
    assert "main_effect" in df.columns
    assert "mTF" in df.columns

    # Check specific values
    row_var1 = df[df["predictor"] == "TF1:VAR1"].iloc[0]
    assert row_var1["all_data"] == "positive"
    assert row_var1["topn"] == "zero"
    assert row_var1["main_effect"] == "positive"
    assert row_var1["mTF"] == "negative"

    row_var2 = df[df["predictor"] == "TF1:VAR2"].iloc[0]
    assert row_var2["all_data"] == "negative"
    assert row_var2["mTF"] == "positive"
