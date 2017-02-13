#!/usr/bin/env python
# feature_generator/feature_generator/feature_generator_classes.py

import pandas as pd
import numpy as np
import abc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.grid_search import GridSearchCV
from data_handling.data_cleaner_fg import DataCleanerFg


class FeatureGenerator(object):
    """Feature Generator Abstract Base Class"""

    def __init__(self, parsed_input):
        self.output_path = parsed_input['output_file_path']
        self.algorithm_name = "Feature Generator abstract base class"
        # Seeding for simulations, can adjust default bounds [1, 10]
        # seeds = np.random.randomint(1, high=10000, size=seeding_size)
        if 1 <= parsed_input['simulation_runs'] <= 10:
            self.seeding_size = parsed_input['simulation_runs']
        else:
            self.seeding_size = 3

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def optimization_routine(self, x_training_data, y_training_data, cross_val=5):
        print("Do nothing with this abstract base class/method")
        pass

    @abc.abstractmethod
    def feature_generator_routine(self, dependent_vars, independent_vars, dependent_vars_columns_list, independent_vars_columns_list):
        print("Do nothing with this abstract base class/method")
        pass

    def output_handler(decorated_fcn):
        """Deocrator definition
        :param decorated_fcn     -- function to be decorated
        """

        def writer(model_name, optimal_features, prediction_scores, dropped_records):
            """ Post processing file writing operations
            :param model_name               -- the name of the model to prepend to output
            :param optimal_features                -- the path location where to write output
            :param prediction_scores           -- the blind input dataframe that was predicted against
            :param dropped_records          -- dropped records output dataframe
            """
            # Output csv to hdfs
            # NOTE: dropped_records is a pandas dataframe
            hdfs_path = '/usr/home/feature_generator/routine_results'
            dropped_records_filename = model_name + '_dropped_records.csv'
            csv_predicted = '/'.join([hdfs_path, dropped_records_filename])
            os.system('echo "%s" | hadoop fs -put - %s' % (dropped_records.to_csv(encoding='utf-8'), csv_predicted))
            print('csv file saved to hdfs, in ', hdfs_path)

        @functools.wraps(decorated_fcn)
        def inner(*args, **kwargs):
            model_name, model_name, optimal_features, prediction_scores = decorated_fcn(*args, **kwargs)
            writer(model_name, model_name, optimal_features, prediction_scores)

        return inner

    @output_handler
    def write_to_hdfs(self, optimal_features, prediction_scores, dropped_records):
        """ Run when there are multiple optimization runs
        NOTE, this function gets decorated so its purpose is to prepare for decoration
        :param optimal_features             -- optimal feature set from ml model
        :param prediction_scores            -- prediction scores from ml model
        :returns model_name,                -- the model name to prepend to output
                 prediction_scores          -- the pandas dataframe of predictions
        """
        model_name = self.algorithm_name

        return model_name, optimal_features, prediction_scores, dropped_records


class RfFeatureGenerator_Pipeline(FeatureGenerator):
    """Random Forest Feature Generator Sub-Class"""

    def __init__(self, parsed_input):
        FeatureGenerator.__init__(self, parsed_input)
        self.algorithm_name = "random forest"

    def optimization_routine(self, x_training_data, y_training_data, cross_val=5):
        """Optimization routine for random forest
        :param x_training_data             -- independent variable(s) training data
        :param y_training_data             -- dependent variable training data
        :param cross_val                   -- number of cross validation folds
        :return rf_model                   -- the optimized random forest prediction model object
        """
        rf_grid = GridSearchCV(RandomForestClassifier(),
                               {'n_estimators': [42, 120, 200, 335, 565],
                                'criterion': ['gini', 'entropy'],
                                'max_depth': [None, 3, 5, 8, 10],
                                'max_features': ['auto', 'log2']}, cv=cross_val)
        rf_grid.fit(x_training_data, y_training_data)
        rf_model = rf_grid.best_estimator_
        rf_model.fit(x_training_data, y_training_data)

        return rf_model

    def encode_target(self, df, target_column):
        """Add column to df with integers for the target.

        :param df               -- pandas DataFrame.
        :param target_column    -- column to map to int, producing a new Target column.
        :returns df_mod,        -- modified DataFrame
                 targets        -- list of target names.
        """
        targets = df[target_column].unique()
        map_to_int = {name: n for n, name in enumerate(targets)}
        df["Target"] = df[target_column].replace(map_to_int)

        return (df, targets)

    def random_forest(self, x_training_data, y_training_data):
        """ Random Forest model, native-python, NOT optimized per input text test data
        :param x_training_data             -- independent variable(s) training data
        :param y_training_data             -- dependent variable training data
        :return text_clf                   -- the random forest prediction model object
        """
        # ngram_range=(1, 2), token_pattern=u'(?u)\\b\\w+\\b')
        text_clf = skPipeline([('vect', CountVectorizer(stop_words='english')),
                               ('tfidf', TfidfTransformer(use_idf=True,
                                                          sublinear_tf=True,
                                                          smooth_idf=False)),
                               ('clf', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))])
        x_training_data = x_training_data.transpose()
        _ = text_clf.fit(x_training_data, y_training_data)

        return text_clf

    def feature_generator_routine(self, dependent_vars, independent_vars, dependent_vars_columns_list, independent_vars_columns_list):
        """ RF feature generator routine
        :param dependent_vars                   -- dataframe of dependent variables to analyze
        :param independent_vars                 -- dataframe of independent variables to utilize in analysis
        :param dependent_vars_columns_list      -- list of dependent variables
        :param independent_vars_columns_list    -- list of independent variables
        :returns rf_optimal_features,           -- optimal features dictionary w.r.t. random forest predictor
                 rf_prediction_scores           -- accuracy score for each dependent variable
        """
        X = independent_vars.copy()
        rf_optimal_features = dict()
        rf_prediction_scores = dict()
        data_cleaner = DataCleanerFg()
        discarded_records = pd.DataFrame()
        dirty_training_records = pd.DataFrame()
        base_threshold = .0001

        # RandomForest (RF) Routine
        for k in range(len(dependent_vars_columns_list)):
            dependent_variable = dependent_vars_columns_list[k]
            y = dependent_vars[dependent_variable]
            training_df = pd.concat([X, y], axis=1)
            cleaned_training_df, dirty_training_records = data_cleaner.data_handler(training_df)
            base_accuracy_first_loop = .00001
            sorted_full_rf_feature_list = list()
            for i in range(self.seeding_size):
                test_split_sz = round(np.random.uniform(.1, .5), 2)
                #cleaned_training_df = cleaned_training_df.apply(LabelEncoder().fit_transform)
                X_train, X_test, y_train, y_test, discarded_records = data_cleaner.tt_split(cleaned_training_df,
                                                                                            dependent_variable,
                                                                                            test_split_size=test_split_sz)
                print("Shape of X_train is ", X_train.shape)
                print("Shape of X_test is ", X_test.shape)
                print("Shape of y_train is ", y_train.shape)
                print("Shape of y_test is ", y_test.shape)

                rf_clf = self.random_forest(X_train, y_train)
                rf_importances = rf_clf.feature_importances_
                rf_model_score = rf_clf.score(X_test, y_test)

                # Compare RF score
                if rf_model_score > base_accuracy_first_loop:
                    base_accuracy_first_loop = rf_model_score
                    features_index = X_train.columns
                    importances_dict = dict(zip(features_index, rf_importances))
                    sorted_full_rf_feature_list = sorted(importances_dict, key=importances_dict.__getitem__, reverse=True)

                if len(sorted_full_rf_feature_list):
                    pass
                else:
                    print('ERROR: ZERO INDEPENDENT VARIABLES FOUND FOR DEPENDENT VAR: %s' % dependent_variable)
                    break

            # Find applicable features for random forest algorithm per accuracy threshold (base_threshold)
            base_features = []
            base_accuracy_second_loop = .00001
            best_rf_accuracy = .0001
            for p in range(len(sorted_full_rf_feature_list)):
                base_features.append(sorted_full_rf_feature_list[p])
                X_per_feature_list = cleaned_training_df[base_features]
                filtered_training_df = pd.concat([X_per_feature_list, y], axis=1)

                for z in range(self.seeding_size):
                    test_split_sz = round(np.random.uniform(.1, .5), 2)
                    # cleaned_training_df = cleaned_training_df.apply(LabelEncoder().fit_transform)
                    X_train, X_test, y_train, y_test, discarded_records = data_cleaner.tt_split(filtered_training_df,
                                                                                                dependent_variable,
                                                                                                test_split_size=test_split_sz)

                    rf_clf = self.random_forest(X_train, y_train)
                    rf_model_score = rf_clf.score(X_test, y_test)

                    # Compare RF score
                    if rf_model_score > base_accuracy_second_loop:
                        base_accuracy_second_loop = rf_model_score

                if p == 0:
                    best_rf_accuracy = base_accuracy_second_loop
                    rf_optimal_features[dependent_variable] = base_features
                    rf_prediction_scores[dependent_variable] = best_rf_accuracy
                else:
                    if (base_accuracy_second_loop - best_rf_accuracy) > base_threshold:
                        best_rf_accuracy = base_accuracy_second_loop
                        rf_optimal_features[dependent_variable] = base_features
                        rf_prediction_scores[dependent_variable] = best_rf_accuracy
                    else:
                        break

        # Finalize records to discard
        discarded_records = pd.concat([dirty_training_records, discarded_records], axis=1)

        return rf_optimal_features, rf_prediction_scores, discarded_records


class RfFeatureGenerator(FeatureGenerator):
    """Random Forest Feature Generator Sub-Class"""

    def __init__(self, parsed_input):
        FeatureGenerator.__init__(self, parsed_input)
        self.algorithm_name = "random forest"

    def optimization_routine(self, x_training_data, y_training_data, cross_val=5):
        """Optimization routine for random forest
        :param x_training_data             -- independent variable(s) training data
        :param y_training_data             -- dependent variable training data
        :param cross_val                   -- number of cross validation folds
        :return rf_model                   -- the optimized random forest prediction model object
        """
        rf_grid = GridSearchCV(RandomForestClassifier(),
                               {'n_estimators': [42, 120, 200, 335, 565],
                                'criterion': ['gini', 'entropy'],
                                'max_depth': [None, 3, 5, 8, 10],
                                'max_features': ['auto', 'log2']}, cv=cross_val)
        rf_grid.fit(x_training_data, y_training_data)
        rf_model = rf_grid.best_estimator_
        rf_model.fit(x_training_data, y_training_data)

        return rf_model

    def encode_target(self, df, target_column):
        """Add column to df with integers for the target.

        :param df               -- pandas DataFrame.
        :param target_column    -- column to map to int, producing a new Target column.
        :returns df_mod,        -- modified DataFrame
                 targets        -- list of target names.
        """
        targets = df[target_column].unique()
        map_to_int = {name: n for n, name in enumerate(targets)}
        df["Target"] = df[target_column].replace(map_to_int)

        return (df, targets)

    def random_forest(self, x_training_data, y_training_data):
        """Random Forest model, native-python, NOT optimized per input text test data
        :param x_training_data             -- independent variable(s) training data
        :param y_training_data             -- dependent variable training data
        :return text_clf                   -- the random forest prediction model object
        """
        text_clf = skPipeline([('vect', CountVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=u'(?u)\\b\\w+\\b')),
                               ('tfidf', TfidfTransformer(use_idf=True,
                                                          sublinear_tf=True,
                                                          smooth_idf=False)),
                               ('clf', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))])

        x_training_data = x_training_data.transpose()
        _ = text_clf.fit(x_training_data, y_training_data)

        return text_clf

    def feature_generator_routine(self, dependent_vars, independent_vars, dependent_vars_columns_list, independent_vars_columns_list):
        """RF feature generator routine
        :param dependent_vars                   -- dataframe of dependent variables to analyze
        :param independent_vars                 -- dataframe of independent variables to utilize in analysis
        :param dependent_vars_columns_list      -- list of dependent variables
        :param independent_vars_columns_list    -- list of independent variables
        :returns rf_optimal_features,           -- optimal features dictionary w.r.t. random forest predictor
                 rf_prediction_scores           -- accuracy score for each dependent variable
        """
        X = independent_vars.copy()
        rf_optimal_features = dict()
        rf_prediction_scores = dict()
        data_cleaner = DataCleanerFg()
        discarded_records = pd.DataFrame()
        dirty_training_records = pd.DataFrame()
        svm_independent_variable_score = dict()

        for k in range(len(dependent_vars_columns_list)):
            dependent_variable = dependent_vars_columns_list[k]
            y = dependent_vars[dependent_variable]
            training_df = pd.concat([X, y], axis=1)
            cleaned_training_df, dirty_training_records = data_cleaner.data_handler(training_df)

            for v in range(len(independent_vars_columns_list)):
                independent_variable = independent_vars_columns_list[k]
                X_single = cleaned_training_df[independent_variable]
                X_single = pd.concat([X_single, y], axis=1)
                base_accuracy = .00001

                for i in range(self.seeding_size):
                    test_split = round(np.random.uniform(.1, .5), 2)
                    X_train, X_test, y_train, y_test, discarded_records = data_cleaner.tt_split(X_single,
                                                                                                dependent_variable,
                                                                                                test_split_size=test_split)
                    pipeline_clf = self.random_forest(X_train, y_train)
                    pipeline_predictions = pipeline_clf.predict(X_test, y_test)
                    pipeline_accuracy = accuracy_score(pipeline_predictions, y_test)

                    if pipeline_accuracy > base_accuracy:
                        base_accuracy = pipeline_accuracy
                        svm_independent_variable_score[independent_variable] = base_accuracy

            # After going through all the independent variables for the particular dependent variable, pick the top 2
            independent_var_list_sorted_by_value = sorted(svm_independent_variable_score, key=svm_independent_variable_score.__getitem__)
            top_2_svm_list = independent_var_list_sorted_by_value[-2:]
            rf_optimal_features[dependent_variable] = top_2_svm_list
            top_accuracy = list()

            # Rerun pipeline with the two top variables to get final accuracy metric for the dependent variable
            X_double = cleaned_training_df[top_2_svm_list]
            X_double = pd.concat([X_double, y], axis=1)
            for i in range(self.seeding_size):
                test_split = round(np.random.uniform(.1, .5), 2)
                X_train, X_test, y_train, y_test, discarded_records = data_cleaner.tt_split(X_double,
                                                                                            dependent_variable,
                                                                                            test_split_size=test_split)
                pipeline_clf = self.random_forest(X_train, y_train)
                pipeline_predictions = pipeline_clf.predict(X_test, y_test)
                pipeline_accuracy = accuracy_score(pipeline_predictions, y_test)
                top_accuracy.append(pipeline_accuracy)

            rf_prediction_scores[dependent_variable] = np.max(top_accuracy)

        # Finalize records to discard
        discarded_records = pd.concat([dirty_training_records, discarded_records], axis=1)

        return rf_optimal_features, rf_prediction_scores, discarded_records


class GbmFeatureGenerator(FeatureGenerator):
    """Random Forest Feature Generator Sub-Class"""

    def __init__(self, parsed_input):
        FeatureGenerator.__init__(self, parsed_input)
        self.algorithm_name = "gradient boosted machine"

    def optimization_routine(self, x_training_data, y_training_data, cross_val=5):
        """Optimization routine for gbm
        :param x_training_data             -- independent variable(s) training data
        :param y_training_data             -- dependent variable training data
        :param cross_val                   -- number of cross validation folds
        :return rf_model                   -- the optimized random forest prediction model object
        """
        gbm_grid = GridSearchCV(GradientBoostingClassifier(),
                                {'n_estimators': [10, 30, 135, 380, 640, 1075],
                                 'loss': ['deviance', 'exponential'],
                                 'max_depth': [3, 5, 8, 14],
                                 'learning_rate': [.1, .05, .01]}, cv=cross_val)
        gbm_grid.fit(x_training_data, y_training_data)
        gbm_model = gbm_grid.best_estimator_
        gbm_model.fit(x_training_data, y_training_data)

        return gbm_model

    def gbm(self, x_training_data, y_training_data):
        """GBM model, native-python, NOT optimized per input text test data
        :param x_training_data             -- independent variable(s) training data
        :param y_training_data             -- dependent variable training data
        :return text_clf                   -- the random forest prediction model object
        """
        text_clf = GradientBoostingClassifier(n_estimators=100,
                                              max_depth=5,
                                              learning_rate=.1)
        _ = text_clf.fit(x_training_data, y_training_data)

        return text_clf

    def feature_generator_routine(self, dependent_vars, independent_vars, dependent_vars_columns_list, independent_vars_columns_list):
        """GBM feature generator routine
        :param dependent_vars                   -- dataframe of dependent variables to analyze
        :param independent_vars                 -- dataframe of independent variables to utilize in analysis
        :param dependent_vars_columns_list      -- list of dependent variables
        :param independent_vars_columns_list    -- list of independent variables
        :returns gbm_optimal_features,          -- optimal features dictionary w.r.t. random forest predictor
                 gbm_prediction_scores          -- accuracy score for each dependent variable
        """
        X = independent_vars.copy()
        gbm_optimal_features = dict()
        gbm_prediction_scores = dict()
        data_cleaner = DataCleanerFg()
        discarded_records = pd.DataFrame()
        dirty_training_records = pd.DataFrame()
        base_threshold = .0001

        # Gradient Boosted Machine (GBM) Routine
        for k in range(len(dependent_vars_columns_list)):
            dependent_variable = dependent_vars_columns_list[k]
            y = dependent_vars[dependent_variable]
            training_df = pd.concat([X, y], axis=1)
            cleaned_training_df, dirty_training_records = data_cleaner.data_handler(training_df)
            base_accuracy_first_loop = .00001
            sorted_full_gbm_feature_list = list()

            for i in range(self.seeding_size):
                test_split_sz = round(np.random.uniform(.1, .5), 2)
                X_train, X_test, y_train, y_test, discarded_records = data_cleaner.tt_split(cleaned_training_df,
                                                                                            dependent_variable,
                                                                                            test_split_size=test_split_sz)
                gbm_clf = self.gbm(X_train, y_train)
                gbm_importances = gbm_clf.feature_importances_
                gbm_model_score = gbm_clf.score(X_test, y_test)

                # Compare GBM score
                if gbm_model_score > base_accuracy_first_loop:
                    base_accuracy_first_loop = gbm_model_score
                    features_index = X_train.columns
                    importances_dict = dict(zip(features_index, gbm_importances))
                    sorted_full_gbm_feature_list = sorted(importances_dict, key=importances_dict.__getitem__, reverse=True)

            if len(sorted_full_gbm_feature_list):
                pass
            else:
                print('ERROR: ZERO INDEPENDENT VARIABLES FOUND FOR DEPENDENT VAR: %s' % dependent_variable)
                break

            # Find applicable features for random forest algorithm per accuracy threshold (base_threshold)
            base_features = []
            base_accuracy_second_loop = .00001
            best_gbm_accuracy = .0001
            for p in range(len(sorted_full_gbm_feature_list)):
                base_features.append(sorted_full_gbm_feature_list[p])
                X_per_feature_list = cleaned_training_df[base_features]
                filtered_training_df = pd.concat([X_per_feature_list, y], axis=1)

                for z in range(self.seeding_size):
                    test_split_sz = round(np.random.uniform(.1, .5), 2)
                    X_train, X_test, y_train, y_test, discarded_records = data_cleaner.tt_split(filtered_training_df,
                                                                                                dependent_variable,
                                                                                                test_split_size=test_split_sz)
                    gbm_clf = self.gbm(X_train, y_train)
                    gbm_model_score = gbm_clf.score(X_test, y_test)

                    # Compare RF score
                    if gbm_model_score > base_accuracy_second_loop:
                        base_accuracy_second_loop = gbm_model_score

                if p == 0:
                    best_gbm_accuracy = base_accuracy_second_loop
                    gbm_optimal_features[dependent_variable] = base_features
                    gbm_prediction_scores[dependent_variable] = best_gbm_accuracy
                else:
                    if (base_accuracy_second_loop - best_gbm_accuracy) > base_threshold:
                        best_gbm_accuracy = base_accuracy_second_loop
                        gbm_optimal_features[dependent_variable] = base_features
                        gbm_prediction_scores[dependent_variable] = best_gbm_accuracy
                    else:
                        break

        # Finalize records to discard
        discarded_records = pd.concat([dirty_training_records, discarded_records], axis=1)

        return gbm_optimal_features, gbm_prediction_scores, discarded_records


class SvmFeatureGenerator(FeatureGenerator):
    """SVM_Pipeline Feature Generator Sub-Class"""

    def __init__(self, parsed_input):
        FeatureGenerator.__init__(self, parsed_input)
        self.algorithm_name = "svm pipeline"

    @classmethod
    def optimized_svm_pipeline(self, x_training_data, y_training_data):
        """ML Pipeline model, native-python, pre-optimized per input text test data
        :param x_training_data             -- independent variable(s) training data
        :param y_training_data             -- dependent variable training data
        :return text_clf                   -- the pipeline prediction model object
        """
        text_clf = skPipeline([('vect', CountVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=u'(?u)\\b\\w+\\b')),
                               ('tfidf', TfidfTransformer(use_idf=True,
                                                          sublinear_tf=True,
                                                          smooth_idf=False)),
                               ('clf', SGDClassifier(loss='perceptron', penalty='l2',
                                                     alpha=1e-3, n_iter=5, random_state=42))])
        _ = text_clf.fit(x_training_data, y_training_data)

        return text_clf

    def optimization_routine(self, x_training_data, y_training_data, cross_val=5):
        pass

    def feature_generator_routine(self, dependent_vars, independent_vars, dependent_vars_columns_list, independent_vars_columns_list):
        """SVM feature generator routine
        :param dependent_vars                   -- dataframe of dependent variables to analyze
        :param independent_vars                 -- dataframe of independent variables to utilize in analysis
        :param dependent_vars_columns_list      -- list of dependent variables
        :param independent_vars_columns_list    -- list of independent variables
        :returns rf_optimal_features,           -- optimal features dictionary w.r.t. random forest predictor
                 rf_prediction_scores           -- accuracy score for each dependent variable
        """
        X = independent_vars.copy()
        svm_optimal_features = dict()
        svm_prediction_scores = dict()
        data_cleaner = DataCleanerFg()
        discarded_records = pd.DataFrame()
        dirty_training_records = pd.DataFrame()
        svm_independent_variable_score = dict()

        for k in range(len(dependent_vars_columns_list)):
            dependent_variable = dependent_vars_columns_list[k]
            y = dependent_vars[dependent_variable]
            training_df = pd.concat([X, y], axis=1)
            print("Shape of training_df = ", training_df.shape)
            cleaned_training_df, dirty_training_records = data_cleaner.data_handler(training_df)
            print("Shape of cleaned_training_df = ", cleaned_training_df.shape)

            for v in range(len(independent_vars_columns_list)):
                independent_variable = independent_vars_columns_list[v]
                X_single = cleaned_training_df[independent_variable]
                print("Showing independent var v from list = ", independent_variable)
                print("Type of cleaned_training_df before concat = ", type(cleaned_training_df))
                print("Type of X_single before concat = ", type(X_single))
                print("Type of y before concat = ", type(y))
                X_single = pd.concat([X_single, y], axis=1)
                base_accuracy = .00001

                for i in range(self.seeding_size):
                    test_split = round(np.random.uniform(.1, .5), 2)
                    X_train, X_test, y_train, y_test, discarded_records = data_cleaner.tt_split(X_single,
                                                                                                dependent_variable,
                                                                                                test_split_size=test_split)
                    print("X_train shape is ", X_train.shape)
                    print("X_test shape is ", X_test.shape)
                    print("y_train shape is ", y_train.shape)
                    print("y_test shape is ", y_test.shape)
                    print("First 5 values of X_train", X_train[:5])
                    print("Type of values of X_train", type(X_train.iloc[1]))

                    pipeline_clf = self.optimized_svm_pipeline(X_train, y_train)
                    pipeline_predictions = pipeline_clf.predict(X_test, y_test)
                    pipeline_accuracy = accuracy_score(pipeline_predictions, y_test)

                    if pipeline_accuracy > base_accuracy:
                        base_accuracy = pipeline_accuracy
                        svm_independent_variable_score[independent_variable] = base_accuracy

            # After going through all the independent variables for the particular dependent variable, pick the top 2
            independent_var_list_sorted_by_value = sorted(svm_independent_variable_score, key=svm_independent_variable_score.__getitem__)
            top_2_svm_list = independent_var_list_sorted_by_value[-2:]
            svm_optimal_features[dependent_variable] = top_2_svm_list
            top_accuracy = list()

            # Rerun pipeline with the two top variables to get final accuracy metric for the dependent variable
            X_double = cleaned_training_df[top_2_svm_list]
            X_double = pd.concat([X_double, y], axis=1)
            for i in range(self.seeding_size):
                test_split = round(np.random.uniform(.1, .5), 2)
                X_train, X_test, y_train, y_test, discarded_records = data_cleaner.tt_split(X_double,
                                                                                            dependent_variable,
                                                                                            test_split_size=test_split)
                pipeline_clf = self.optimized_svm_pipeline(X_train, y_train)
                pipeline_predictions = pipeline_clf.predict(X_test, y_test)
                pipeline_accuracy = accuracy_score(pipeline_predictions, y_test)
                top_accuracy.append(pipeline_accuracy)

            svm_prediction_scores[dependent_variable] = np.max(top_accuracy)

        # Finalize records to discard
        discarded_records = pd.concat([dirty_training_records, discarded_records], axis=1)

        return svm_optimal_features, svm_prediction_scores, discarded_records
