#!/usr/bin/env python
# feature_generator/feature_generator/compare_scores.py

import numpy as np

def compare_scores(dependent_vars_list, prediction_scores, optimal_features):
    """Compare the predictive accuracy for each dependent variable between each algorithm

    :param dependent_vars_list  -- the list of dependent variables
    :param prediction_scores    -- dictionary of dictionaries of prediction scores for each algorithm in scope
    :param optimal_features     -- dictionary of dictionaries of optimal features for each algorithm in scope
    :return prediction_set      -- the dict of which algorithm along with which features to utilize
    """

    # Explanation of nested list / nested dict/ nested list expression:
    # [key for key, val in prediction_scores.items() if prediction_scores[key][dep_var] == np.max(list({v[dep_var] for v in prediction_scores.itervalues()}))]
    # Handling a dictionary of dictionaries, with each sub-dictionary having the same key set, being
    # the dependent_vars_list. Find the best accuracy, identify the algorithm, extract optimal feature set
    # best prediction score/accuracy -> np.max(list({v[dep_var] for v in prediction_scores.itervalues()}))
    # key for the key, value pair such that prediction_scores[key][value] == best prediction score/accuracy
    # Output format:  dict('dependent_var_name' : {'algorithm': [optimal feature set list]})
    prediction_set = dict()
    for dep_var in dependent_vars_list:
        optimal_algo = [key for key, val in prediction_scores.items() if prediction_scores[key][dep_var] == np.max(list({v[dep_var] for v in prediction_scores.itervalues()}))]
        prediction_set[dep_var] = {optimal_algo: optimal_features[dep_var]}

    return prediction_set
