#!/usr/bin/env python
# feature_generator/feature_generator_spark_main.py
# Analytics Pipeline
# Paul Wormuth, Data Scientist,
""" Spark on Hadoop, Data Feature Generator """


import sys
import pandas as pd
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructField, StringType, StructType

# ---- Instantiate SparkContext & SQLContext ---- #
sc = SparkContext()
sc.addPyFile('hdfs://<hostname.domain>:<port>/path/to/hdfs/location/of/feature_generator_egg_file.egg')
sqlsc = SQLContext(sc)

# ---- Function Definitions & Main ---- #

def base_spark_dataframe(parsed_input):
    """Read file from HDFS and instantiate base Spark dataframe
    :param parsed_input:        -- the parsed input that contains the parameters/paths w.r.t. the data in hdfs
    :return: spark_df_data      -- a spark dataframe of the entire data set
    """
    category_data = sc.textFile(parsed_input['input_file_path'])
    category_headers = category_data.take(1)[0]
    category_data = category_data\
        .filter(lambda line: line != category_headers)\
        .map(lambda x: x.encode('utf-8'))\
        .map(lambda l: l.split(","))
    extracted_fields = [StructField(field_name, StringType(), True) for field_name in category_headers.split(',')]
    extracted_schema = StructType(extracted_fields)
    spark_df_data = sqlsc.createDataFrame(category_data, extracted_schema)

    return spark_df_data


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
    :param prediction_scores               -- the input test file to generate predictions for
    :returns model_name,                -- the model name to prepend to output
             prediction_scores          -- the pandas dataframe of predictions
    """
    model_name = self.algorithm_name

    return model_name, optimal_features, prediction_scores

def main():
    """ Main program to run in pyspark
        In order to utilize the sci-kit learn libraries and take advantage of their feature ranking
        it's required to convert the spark dataframe into a pandas dataframe. This will bring the
        memory overhead entirely onto the spark master node's JVM, therefore be congnizant of RAM
        availability on that node.
    """
    # Import modules associated to feature_generator python package
    from feature_generator.feature_generator_classes import RfFeatureGenerator, SvmFeatureGenerator, GbmFeatureGenerator
    from feature_generator.compare_scores import compare_scores
    from data_handling.input_args_handler_fg import parse_input_data

    # Parse input arguments into dictionary, retrieve base spark dataframe
    parsed_input = parse_input_data(sys.argv[1])
    spark_df = base_spark_dataframe(parsed_input)
    pd_df = spark_df.toPandas()

    # If foreign text, decode to unicode, then encode to utf-8
    # pd_df = pd_df.applymap(lambda x: str(x).decode('latin-1').encode('utf-8'))

    # Fill the multiple na values with zeros, then filter out all zero columns
    pd_df.replace([np.inf, -np.inf, '', 'nan'], np.nan, inplace=True)
    pd_df = pd_df.fillna(0)

    # There are various ways to filter out zeros, this is just one
    pd_df = pd_df.loc[:, (pd_df != 0).any(axis=0)]
    zeros_na_columns_df = pd_df.loc[:, (pd_df == 0).any(axis=0)]

    # Segregate dependent and independent variables (originally passed from input json)
    pd_df.columns = map(lambda txt: txt.lower(), pd_df.columns.tolist())
    column_list = pd_df.columns.tolist()
    independent_var_keyword = parsed_input['independent_var_keyword']
    dependent_var_keyword = parsed_input['dependent_var_keyword']
    dependent_vars_columns_list = [x for x in column_list if dependent_var_keyword in x]
    independent_vars_columns_list = [x for x in column_list if independent_var_keyword in x]
    unlabeled_vars_columns_list = [x for x in column_list if (independent_var_keyword not in x) & (dependent_var_keyword not in x)]
    dependent_vars = pd_df[dependent_vars_columns_list]
    independent_vars = pd_df[independent_vars_columns_list]
    unlabeled_vars = pd_df[unlabeled_vars_columns_list]

    # Sanity check on variable sets
    print("Dependent variables", dependent_vars_columns_list)
    print("Independent variables", independent_vars_columns_list)
    if unlabeled_vars.empty:
        print("All columns contained a keyword, zero columns designated for removal")
        removed_columns_df = pd.DataFrame()
    else:
        if unlabeled_vars.shape[1]:
            print("WARNING: {} Columns designated for removal for not containing a keyword".format(unlabeled_vars.shape[1]))
        else:
            print("WARNING: 1 Column designated for removal for not containing a keyword")
        print unlabeled_vars_columns_list
        removed_columns_df = pd.concat([unlabeled_vars, zeros_na_columns_df], axis=1)

    # Generate Random Forest Optimal Features
    rf_feature_generator = RfFeatureGenerator(parsed_input)
    rf_optimal_features, rf_prediction_scores, dropped_records = rf_feature_generator.feature_generator_routine(dependent_vars,
                                                                                                                independent_vars,
                                                                                                                dependent_vars_columns_list,
                                                                                                                independent_vars_columns_list)

    # Generate SVM Optimal Features
    svm_feature_generator = SvmFeatureGenerator(parsed_input)
    svm_optimal_features, svm_prediction_scores, dropped_records = svm_feature_generator.feature_generator_routine(dependent_vars,
                                                                                                                   independent_vars,
                                                                                                                   dependent_vars_columns_list,
                                                                                                                   independent_vars_columns_list)

    # Generate GBM Optimal Features
    gbm_feature_generator = GbmFeatureGenerator(parsed_input)
    gbm_optimal_features, gbm_prediction_scores, dropped_records = gbm_feature_generator.feature_generator_routine(dependent_vars,
                                                                                                                   independent_vars,
                                                                                                                   dependent_vars_columns_list,
                                                                                                                   independent_vars_columns_list)

    # Compare scores to get optimal prediction algorithm
    algorithm_list = ['random forest', 'svm pipeline', 'gbm']
    prediction_scores = dict(zip(algorithm_list, [rf_prediction_scores, svm_prediction_scores, gbm_prediction_scores]))
    optimal_features = dict(zip(algorithm_list, [rf_optimal_features, svm_optimal_features, gbm_optimal_features]))
    best_prediction_algorithm_set = compare_scores(dependent_vars_columns_list, prediction_scores, optimal_features)

    # Output configuration json to HDFS, as well as filtered/removed data
    # TODO -- Create decorated (base) class method to handle json configuration output



if __name__ == '__main__':
    sys.exit(main() or 0)