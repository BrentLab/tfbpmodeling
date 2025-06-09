import pytest

from tfbpmodeling.utils.exclude_predictor_variables import exclude_predictor_variables


@pytest.mark.parametrize(
    ("predictors", "exclude", "expected"),
    [
        # Nothing excluded  ➜ list unchanged
        (["A", "B", "C"], [], ["A", "B", "C"]),
        # Specific exclusions ➜ only those removed
        (["A", "B", "C"], ["B"], ["A", "C"]),
        # Sentinel term present ➜ everything removed
        (["A", "B"], ["exclude_all"], []),
        # Mixed case: sentinel + other names ➜ still empty
        (["X", "Y"], ["exclude_all", "X"], []),
    ],
)
def test_exclude_predictor_variables_happy_path(predictors, exclude, expected):
    assert exclude_predictor_variables(predictors, exclude) == expected


@pytest.mark.parametrize(
    "bad_predictors, bad_exclude, bad_term, expected_message",
    [
        # Non-list predictors
        ("not_a_list", [], "exclude_all", "predictor_variables must be a list"),
        # Non-list exclude list
        (["A", "B"], "not_a_list", "exclude_all", "exclude_list must be a list"),
        # Non-string sentinel
        (["A"], [], 123, "exclude_all_term must be a string"),
    ],
)
def test_exclude_predictor_variables_type_errors(
    bad_predictors, bad_exclude, bad_term, expected_message
):
    with pytest.raises(TypeError, match=expected_message):
        exclude_predictor_variables(bad_predictors, bad_exclude, bad_term)
