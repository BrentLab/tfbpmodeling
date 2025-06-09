def exclude_predictor_variables(
    predictor_variables: list,
    exclude_list: list,
    exclude_all_term: str = "exclude_all",
) -> list:
    """
    Given a set of predictor variables, return a list of variables that are not in the
    exclude_list.

    :param predictor_variables: list of predictor variables
    :param exclude_list: list of variables to exclude
    :param exclude_all_term: a term that, if present in the exclude_list, will exclude
        all predictor variables and return an empty list
    :return: list of predictor variables that are not in the exclude_list
    :raises TypeError: if predictor_variables or exclude_list is not a list
    :raises TypeError: if exclude_all_term is not a string

    """
    if not isinstance(predictor_variables, list):
        raise TypeError("predictor_variables must be a list")
    if not isinstance(exclude_list, list):
        raise TypeError("exclude_list must be a list")
    if not isinstance(exclude_all_term, str):
        raise TypeError("exclude_all_term must be a string")

    exclusions: set[str] = set(exclude_list)
    if exclude_all_term in exclusions:
        return []
    return [v for v in predictor_variables if v not in exclusions]
