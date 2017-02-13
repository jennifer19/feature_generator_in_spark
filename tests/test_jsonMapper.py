#!/usr/bin/env python
# ml_pipeline_27/tests/test_json_mapper.py
import unittest
from data_handling.input_args_handler import json_mapper


class TestJson_mapper(unittest.TestCase):
    """ Unit Testing /data_handling/input_args_handler:json_mapper """

    # prepare test
    def setUp(self):
        """ Setup for test, instantiate object / cycle method(s) for test object creation """
        print("TestJson_mapper:setup_:begin")
        # Instantiate instances/values from classes/functions being tested
        category, country, target_global_char = 'SHAVING_PRODUCTS', 'UK', 'GLOBAL_BRAND_1'
        self.validate_output = ['RETAILER_DESC', 'LOCAL_GB_LOC_00004_BRAND']
        self.test_output = json_mapper(category, country, target_global_char)
        print("TestJson_mapper:setup_:end")

    # ending test
    def tearDown(self):
        """ Post-test cleanup """
        # Do something based on test complexity/objectives/options
        print("TestJson_mapper:tearDown_")

    def test_json_mapper(self):
        """ Test json_mapper function """
        self.assertEqual(first=self.validate_output, second=self.test_output, msg=None)

    # @unittest.skip("Skip over integrity test")
    def test_json_mapper_integrity(self):
        """ Testing json_mapper returns a list """
        self.assertTrue(isinstance(self.test_output, list))
