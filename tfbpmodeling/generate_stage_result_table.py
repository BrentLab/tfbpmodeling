import json
import os

import pandas as pd


def get_stage_result(interval):
    """
    Map a coefficient interval to a STAGE_RESULT label.

    This helper converts a coefficient's confidence interval (or a point
    estimate represented as a degenerate interval) into one of the categorical
    stage results used in the summary table.

    - "positive": lower > 0
    - "negative": upper < 0
    - "zero": the interval contains 0 (e.g., lower <= 0 <= upper) or both
      bounds are exactly 0
    - "none": the predictor/stage is absent (interval is None) or unrecognized

    :param interval: A 2-element tuple/list (lower, upper) of floats representing
        the coefficient interval for a predictor at a given stage. Use the same
        value for lower and upper to represent a point estimate. Pass None if the
        predictor/stage is absent.

    :return: A string in {"positive", "negative", "zero", "none"} indicating the
        STAGE_RESULT classification.

    :raises: None

    """

    if interval is None:
        return "none"
    lower, upper = interval
    if lower > 0:
        return "positive"
    elif upper < 0:
        return "negative"
    elif lower <= 0 <= upper or (lower == 0 and upper == 0):
        return "zero"
    return "none"


def generate_stage_result_table(tf_dir):
    """
    Build and write `stage_result_table.csv` for a single TF modeling run.

    This function collates stage-level outcomes from three pipeline stages—
    "all data", "topn", and "main effect/interactor"—and emits a unified table
    with one row per predictor. Each stage’s numeric interval (or point estimate)
    is mapped to a categorical STAGE_RESULT via `get_stage_result`, producing the
    columns:
      - predictor
      - all_data
      - topn
      - main_effect
      - mTF

    :param tf_dir: Directory containing stage outputs for a single TF run.

    :return: None (writes CSV to disk)

    :raises: None. If the required all-data file is missing, prints a message and
             returns early without writing a CSV.

    """

    all_data_path = os.path.join(tf_dir, "all_data_result_object", "result_obj.json")
    topn_path = os.path.join(tf_dir, "topn_result_object", "result_obj.json")
    main_effect_path = os.path.join(tf_dir, "interactor_vs_main_result.json")
    output_csv_path = os.path.join(tf_dir, "stage_result_table.csv")

    if not os.path.exists(all_data_path):
        print(f"Skipping：{tf_dir}，Not find all_data_result_object")
        return

    with open(all_data_path) as f:
        all_data_json = json.load(f)
    all_data_stage = list(all_data_json.values())[0]

    topn_stage = {}
    if os.path.exists(topn_path):
        with open(topn_path) as f:
            topn_json = json.load(f)
            topn_stage = list(topn_json.values())[0]

    main_effect_dict = {}
    mtf_dict = {}
    if os.path.exists(main_effect_path):
        with open(main_effect_path) as f:
            main_effect_list = json.load(f)
        for entry in main_effect_list:
            tf, variant = entry["interactor"].split(":")
            predictor = f"{tf}:{entry['variant']}"
            interval_main = [entry["coef_main_effect"]] * 2
            interval_interactor = [entry["coef_interactor"]] * 2
            main_effect_dict[predictor] = get_stage_result(interval_main)
            mtf_dict[predictor] = get_stage_result(interval_interactor)

    predictors = sorted(all_data_stage.keys())

    rows = []
    for predictor in predictors:
        rows.append(
            {
                "predictor": predictor,
                "all_data": get_stage_result(all_data_stage.get(predictor)),
                "topn": (
                    get_stage_result(topn_stage.get(predictor))
                    if topn_stage
                    else "none"
                ),
                "main_effect": main_effect_dict.get(predictor, "none"),
                "mTF": mtf_dict.get(predictor, "none"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
