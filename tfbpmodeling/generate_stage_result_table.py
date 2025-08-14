import json
import os

import pandas as pd


def get_stage_result(interval):
    """
    Map a coefficient interval to a STAGE_RESULT string.

    Parameters
    ----------
    interval : tuple[float, float] | list[float] | None
        The (lower, upper) bounds for a predictor's coefficient interval.
        Pass None if the predictor/stage is absent.

    Returns
    -------
    str
        One of:
        - "positive": lower > 0 (strictly positive interval)
        - "negative": upper < 0 (strictly negative interval)
        - "zero": the interval contains 0 (e.g., lower <= 0 <= upper) or both
          bounds are exactly 0
        - "none": the interval is None or unrecognized (fallback)

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
