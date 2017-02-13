#!/usr/bin/env python
# ml_pipeline_27/tests/test_parse_input_data.py
import unittest
import json
import copy
#from data_handling.input_args_handler import parse_input_data
from data_handling.input_args_handler_fg import parse_input_data


class TestParse_input_data(unittest.TestCase):
    """ Unit Testing /data_handling/input_args_handler:parse_input_data """

    # prepare test
    def setUp(self):
        """ Setup for test, instantiate object / cycle method(s) for test object creation """
        print("TestParse_input_data:setup_:begin")
        # Instantiate instances/values from classes/functions being tested
        test_input = {"training_file": "/ml_pipeline_27/unit_testing/dummy_value1.csv",
                      "file_for_testing": "/ml_pipeline_27/unit_testing/dummy_value2.csv",
                      "model_name": "ml_pipeline_27_unit_testing_dummy_value",
                      "model_path": "/ml_pipeline_27/unit_testing/dummy_path1",
                      "output_json_path": "/ml_pipeline_27/unit_testing/dummy_path2",
                      "csv_accuracy_report_path": "/ml_pipeline_27/unit_testing/dummy_path3",
                      "category": "SHAVING_PRODUCTS",
                      "country": "UK",
                      "target_global_char": "GLOBAL_BRAND_1",
                      "predicted_results_path": "/ml_pipeline_27/unit_testing/dummy_path4",
                      "test_train_split": "50"
                      }
        json_input_raw = json.dumps(test_input)
        self.validate_output = copy.deepcopy(test_input)
        self.test_output = parse_input_data(json_input_raw)
        print("TestParse_input_data:setup_:end")

    # ending test
    def tearDown(self):
        """ Post-test cleanup """
        # Do something based on test complexity/objectives/options
        print("TestParse_input_data:tearDown_")

    def test_parse_input_data(self):
        """ Test parse_input_data function """
        self.assertEqual(first=self.validate_output['training_file'],
                         second=self.test_output['training_file_path'], msg=None)
        self.assertEqual(first=self.validate_output['csv_accuracy_report_path'],
                         second=self.test_output['accuracy_output_path'], msg=None)
        self.assertEqual(first=0.0,
                         second=self.test_output['test_percentage'], msg=None)

    # @unittest.skip("Skip over integrity test")
    def test_parse_input_data_integrity(self):
        """ Testing parse_input_data returns a dictionary """
        self.assertTrue(isinstance(self.test_output, dict))