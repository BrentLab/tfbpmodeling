import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from tfbpmodeling.bootstrap_stratified_cv import bootstrap_stratified_cv_modeling
from tfbpmodeling.bootstrap_stratified_cv_loop import bootstrap_stratified_cv_loop
from tfbpmodeling.bootstrapped_input_data import BootstrappedModelingInputData
from tfbpmodeling.evaluate_interactor_significance_lassocv import (
    evaluate_interactor_significance_lassocv,
)
from tfbpmodeling.evaluate_interactor_significance_linear import (
    evaluate_interactor_significance_linear,
)
from tfbpmodeling.modeling_input_data import ModelingInputData
from tfbpmodeling.stratification_classification import stratification_classification
from tfbpmodeling.utils.exclude_predictor_variables import exclude_predictor_variables

logger = logging.getLogger("main")


def generate_stage_result_table(tf_dir):
    def get_stage_result(interval):
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
            interval_main = [entry["avg_r2_main_effect"]] * 2
            interval_interactor = [entry["avg_r2_interactor"]] * 2
            main_effect_dict[predictor] = get_stage_result(interval_main)
            mtf_dict[predictor] = get_stage_result(interval_interactor)

    predictors = sorted(
        set(all_data_stage.keys())
        | set(topn_stage.keys())
        | set(main_effect_dict.keys())
    )

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


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    This could be used to customize the help message formatting for the argparse parser.

    Left as a placeholder.

    """


def linear_perturbation_binding_modeling(args):
    """
    :param args: Command-line arguments containing input file paths and parameters.
    """
    if not isinstance(args.max_iter, int) or args.max_iter < 1:
        raise ValueError("The `max_iter` parameter must be a positive integer.")

    max_iter = int(args.max_iter)

    logger.info(f"estimator max_iter: {max_iter}.")

    logger.info("Step 1: Preprocessing")

    # validate input files/dirs
    if not os.path.exists(args.response_file):
        raise FileNotFoundError(f"File {args.response_file} does not exist.")
    if not os.path.exists(args.predictors_file):
        raise FileNotFoundError(f"File {args.predictors_file} does not exist.")
    if os.path.exists(args.output_dir):
        logger.warning(f"Output directory {args.output_dir} already exists.")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory created at {args.output_dir}")

    # the output subdir is where the output of this modeling run will be saved
    output_subdir = os.path.join(
        args.output_dir, os.path.join(args.perturbed_tf + args.output_suffix)
    )
    if os.path.exists(output_subdir):
        raise FileExistsError(
            f"Directory {output_subdir} already exists. "
            "Please specify a different `output_dir`."
        )
    else:
        os.makedirs(output_subdir, exist_ok=True)
        logger.info(f"Output subdirectory created at {output_subdir}")

    # instantiate a estimator
    # `fit_intercept` is set opposite of `scale_by_std`. If `scale_by_std` is `False`,
    # the default, then `fit_intercept` is set to True and the estimator will fit the
    # intercept. If `scale_by_std` is True, then the estimator will not fit the
    # intercept, meaning it assumes the data is centered.
    estimator = LassoCV(
        fit_intercept=True,
        selection="random",
        n_alphas=100,
        random_state=42,
        n_jobs=args.n_cpus,
        max_iter=max_iter,
    )

    input_data = ModelingInputData.from_files(
        response_path=args.response_file,
        predictors_path=args.predictors_file,
        perturbed_tf=args.perturbed_tf,
        feature_blacklist_path=args.blacklist_file,
        top_n=args.top_n,
    )

    logger.info("Step 2: Bootstrap LassoCV on all data, full interactor model")

    # Unset the top n masking -- we want to use all the data for the first round
    # modeling
    input_data.top_n_masked = False

    # extract a list of predictor variables, which are the columns of the predictors_df
    predictor_variables = input_data.predictors_df.columns.drop(input_data.perturbed_tf)

    # drop any variables which are in args.exclude_interactor_variables
    predictor_variables = exclude_predictor_variables(
        list(predictor_variables), args.exclude_interactor_variables
    )

    # create a list of interactor terms with the perturbed_tf as the first term
    interaction_terms = [
        f"{input_data.perturbed_tf}:{var}" for var in predictor_variables
    ]

    # Construct the full interaction formula, ie perturbed_tf + perturbed_tf:other_tf1 +
    # perturbed_tf:other_tf2 + ... . perturbed_tf main effect only added if
    # --ptf_main_effect is passed.
    if args.ptf_main_effect:
        logger.info("adding pTF main effect to `all_data_formula`")
        all_data_formula = (
            f"{input_data.perturbed_tf} + {' + '.join(interaction_terms)}"
        )
    else:
        all_data_formula = " + ".join(interaction_terms)

    if args.squared_pTF:
        # if --squared_pTF is passed, then add the squared perturbed TF to the formula
        squared_term = f"I({input_data.perturbed_tf} ** 2)"
        logger.info(f"Adding squared term to model formula: {squared_term}")
        all_data_formula += f" + {squared_term}"

    if args.cubic_pTF:
        # if --cubic_pTF is passed, then add the cubic perturbed TF to the formula
        cubic_term = f"I({input_data.perturbed_tf} ** 3)"
        logger.info(f"Add cubic term to model formula: {cubic_term}")
        all_data_formula += f" + {cubic_term}"

    # if --row_max is passed, then add "row_max" to the formula
    if args.row_max:
        logger.info("Adding `row_max` to the all data model formula")
        all_data_formula += " + row_max"

    # if --add_model_variables is passed, then add the variables to the formula
    if args.add_model_variables:
        logger.info(
            f"Adding model variables to the all data model "
            f"formula: {args.add_model_variables}"
        )
        all_data_formula += " + " + " + ".join(args.add_model_variables)

    logger.debug(f"All data formula: {all_data_formula}")

    # create the bootstrapped data.
    bootstrapped_data_all = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=input_data.get_modeling_data(
            all_data_formula,
            add_row_max=args.row_max,
            drop_intercept=True,
            scale_by_std=args.scale_by_std,
        ),
        n_bootstraps=args.n_bootstraps,
        normalize_sample_weights=args.normalize_sample_weights,
        random_state=args.random_state,
    )

    logger.info(
        f"Running bootstrap LassoCV on all data with {args.n_bootstraps} bootstraps"
    )
    if args.iterative_dropout:
        logger.info("Using iterative dropout modeling for all data results.")
        all_data_results = bootstrap_stratified_cv_loop(
            bootstrapped_data=bootstrapped_data_all,
            perturbed_tf_series=input_data.predictors_df[input_data.perturbed_tf],
            estimator=estimator,
            ci_percentile=float(args.all_data_ci_level),
            stabilization_ci_start=args.stabilization_ci_start,
            bins=args.bins,
            output_dir=output_subdir,
        )
    else:
        logger.info("Using standard bootstrap modeling for all data results.")
        all_data_results = bootstrap_stratified_cv_modeling(
            bootstrapped_data=bootstrapped_data_all,
            perturbed_tf_series=input_data.predictors_df[input_data.perturbed_tf],
            estimator=estimator,
            ci_percentiles=[float(args.all_data_ci_level)],
            bins=args.bins,
        )
    # create the all data object output subdir
    all_data_output = os.path.join(output_subdir, "all_data_result_object")
    os.makedirs(all_data_output, exist_ok=True)

    logger.info(f"Serializing all data results to {all_data_output}")
    all_data_results.serialize("result_obj", all_data_output)

    # Extract the coefficients that are significant at the specified confidence level
    all_data_sig_coefs = all_data_results.extract_significant_coefficients(
        ci_level=args.all_data_ci_level,
    )

    logger.info(f"all_data_sig_coefs: {all_data_sig_coefs}")

    if not all_data_sig_coefs:
        logger.warning(
            f"No significant coefficients found at {args.all_data_ci_level}% "
            "confidence level. Exiting."
        )
        return

    # write all_data_sig_coefs to a json file
    all_data_ci_str = str(args.all_data_ci_level).replace(".", "-")
    all_data_output_file = os.path.join(
        output_subdir, f"all_data_significant_{all_data_ci_str}.json"
    )
    logger.info(f"Writing the all data significant results to {all_data_output_file}")
    with open(
        all_data_output_file,
        "w",
    ) as f:
        json.dump(all_data_sig_coefs, f, indent=4)

    logger.info(
        "Step 3: Running LassoCV on topn data with significant coefficients "
        "from the all data model"
    )

    # Create the formula for the topn modeling from the significant coefficients
    # NOTE: to remove the intercept, we need to add " -1 "
    topn_formula = f"{' + '.join(all_data_sig_coefs.keys())}"
    logger.debug(f"Topn formula: {topn_formula}")

    # apply the top_n masking
    input_data.top_n_masked = True

    # Create the bootstrapped data for the topn modeling
    bootstrapped_data_top_n = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=input_data.get_modeling_data(
            topn_formula,
            add_row_max=args.row_max,
            drop_intercept=True,
            scale_by_std=args.scale_by_std,
        ),
        n_bootstraps=args.n_bootstraps,
        normalize_sample_weights=args.normalize_sample_weights,
        random_state=(
            args.random_state + 10 if args.random_state else args.random_state
        ),
    )

    logger.debug(
        f"Running bootstrap LassoCV on topn data with {args.n_bootstraps} bootstraps"
    )
    topn_results = bootstrap_stratified_cv_modeling(
        bootstrapped_data_top_n,
        input_data.predictors_df[input_data.perturbed_tf],
        estimator=estimator,
        ci_percentiles=[float(args.topn_ci_level)],
    )

    # create the topn data object output subdir
    topn_output = os.path.join(output_subdir, "topn_result_object")
    os.makedirs(topn_output, exist_ok=True)

    logger.info(f"Serializing topn results to {topn_output}")
    topn_results.serialize("result_obj", topn_output)

    # extract the topn_results at the specified confidence level
    topn_output_res = topn_results.extract_significant_coefficients(
        ci_level=args.topn_ci_level
    )

    logger.info(f"topn_output_res: {topn_output_res}")

    if not topn_output_res:
        logger.warning(
            f"No significant coefficients found at {args.topn_ci_level}% "
            "confidence level. Exiting."
        )
        return

    # write topn_output_res to a json file
    topn_ci_str = str(args.topn_ci_level).replace(".", "-")
    topn_output_file = os.path.join(
        output_subdir, f"topn_significant_{topn_ci_str}.json"
    )
    logger.info(f"Writing the topn significant results to {topn_output_file}")
    with open(topn_output_file, "w") as f:
        json.dump(topn_output_res, f, indent=4)

    logger.info(
        "Step 4: Test the significance of the interactor terms that survive "
        "against the corresoponding main effect"
    )

    if args.stage4_topn:
        logger.info("Stage 4 will use top-n masked input data.")
        input_data.top_n_masked = True
    else:
        logger.info("Stage 4 will use full input data.")

    # calculate the statification classes for the perturbed TF (all data)
    stage4_classes = stratification_classification(
        input_data.predictors_df[input_data.perturbed_tf].squeeze(),
        bins=args.bins,
    )

    # Test the significance of the interactor terms
    evaluate_interactor_significance = (
        evaluate_interactor_significance_lassocv
        if args.stage4_lasso
        else evaluate_interactor_significance_linear
    )

    results = evaluate_interactor_significance(
        input_data,
        stratification_classes=stage4_classes,
        model_variables=list(
            topn_results.extract_significant_coefficients(
                ci_level=args.topn_ci_level
            ).keys()
        ),
        estimator=estimator,
    )

    output_significance_file = os.path.join(
        output_subdir, "interactor_vs_main_result.json"
    )
    logger.info(
        "Writing the final interactor significance "
        f"results to {output_significance_file}"
    )
    results.serialize(output_significance_file)

    generate_stage_result_table(output_subdir)


def parse_bins(s):
    try:
        return [np.inf if x == "np.inf" else int(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid bin value in '{s}'")


def parse_comma_separated_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_json_dict(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")


# Allowed keys for method='L-BFGS-B' (excluding deprecated options)
LBFGSB_ALLOWED_KEYS = {
    "maxcor",  # int
    "ftol",  # float
    "gtol",  # float
    "eps",  # float or ndarray
    "maxfun",  # int
    "maxiter",  # int
    "maxls",  # int
    "finite_diff_rel_step",  # float or array-like or None
}


def parse_lbfgsb_options(s):
    try:
        opts = json.loads(s)
        if not isinstance(opts, dict):
            raise ValueError("Options must be a JSON object")

        unexpected_keys = set(opts) - LBFGSB_ALLOWED_KEYS
        if unexpected_keys:
            raise argparse.ArgumentTypeError(
                f"Unexpected keys in --minimize_options: {unexpected_keys}"
            )
        return opts
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def add_general_arguments_to_subparsers(subparsers, general_arguments):
    for subparser in subparsers.choices.values():
        for arg in general_arguments:
            subparser._add_action(arg)


def common_modeling_binning_arguments(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument(
        "--bins",
        type=parse_bins,
        default="0,8,64,512,np.inf",
        help=(
            "Comma-separated list of bin edges (integers or 'np.inf'). "
            "Default is --bins 0,8,12,np.inf"
        ),
    )


def common_modeling_input_arguments(
    parser: argparse._ArgumentGroup, top_n_default: int | None = 600
) -> None:
    """Add common input arguments for modeling commands."""
    parser.add_argument(
        "--response_file",
        type=str,
        required=True,
        help=(
            "Path to the response CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), matching the index "
            "format in both response and predictor files. The perturbed gene will "
            "be removed from the model data only if its column names match the "
            "index format."
        ),
    )
    parser.add_argument(
        "--predictors_file",
        type=str,
        required=True,
        help=(
            "Path to the predictors CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), ensuring consistency "
            "between response and predictor files."
        ),
    )
    parser.add_argument(
        "--perturbed_tf",
        type=str,
        required=True,
        help=(
            "Name of the perturbed transcription factor (TF) used as the "
            "response variable. It must match a column in the response file."
        ),
    )
    parser.add_argument(
        "--blacklist_file",
        type=str,
        default="",
        help=(
            "Optional file containing a list of features (one per line) to be excluded "
            "from the analysis."
        ),
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap samples to generate for resampling. Default is 1000",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Set this to an integer to make the bootstrap sampling reproducible. "
        "Default is None (no fixed seed) and each call will produce different "
        "bootstrap indices. Note that if this is set, the `top_n` random_state will "
        "be +10 in order to make the top_n indices different from the `all_data` step",
    )
    parser.add_argument(
        "--normalize_sample_weights",
        action="store_true",
        help=(
            "Set this to normalize the sample weights to sum to 1. " "Default is False."
        ),
    )
    parser.add_argument(
        "--scale_by_std",
        action="store_true",
        help=(
            "Set this to center and scale the model matrix. Note that setting this "
            "will set the `fit_intercept` parameter of the LassoCV estimator to "
            "False."
        ),
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=top_n_default,
        help=(
            "Number of features to retain in the second round of modeling. "
            f"Default is {top_n_default}"
        ),
    )


def common_modeling_feature_options(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument(
        "--row_max",
        action="store_true",
        help=(
            "Include the row max as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )
    parser.add_argument(
        "--squared_pTF",
        action="store_true",
        help=(
            "Include the squared pTF as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )
    parser.add_argument(
        "--cubic_pTF",
        action="store_true",
        help=(
            "Include the cubic pTF as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )
    parser.add_argument(
        "--exclude_interactor_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to exclude from the interactor terms. "
            "E.g. red_median,green_median. To exclude all variables, use 'exclude_all'"
        ),
    )
    parser.add_argument(
        "--add_model_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to add to the all_data model. "
            "E.g., red_median,green_median would be added as ... + red_median + "
            "green_median"
        ),
    )
    parser.add_argument(
        "--scale_center",
        action="store_true",
        help=("Scale and center the model matrix by the mean and standard deviation. "),
    )
    parser.add_argument(
        "--ptf_main_effect",
        action="store_true",
        help=(
            "Include the perturbed transcription factor (pTF) main effect in the "
            "modeling formula. This is added to the all_data model formula."
        ),
    )
