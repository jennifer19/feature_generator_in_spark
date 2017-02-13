#!/usr/bin/env python
# ml_pipeline_27/tests/test_dataCleanerPrediction.py
import unittest
import numpy as np
import pandas as pd
from data_handling.data_cleaner import DataCleanerPrediction


class TestDataCleanerPrediction(unittest.TestCase):
    """ Unit Testing /data_handling/data_cleaner:DataCleanerPrediction """

    # prepare test
    def setUp(self):
        """ Setup for test, instantiate object / cycle method(s) for test object creation """
        print("TestDataCleanerPrediction:setup_:begin")
        # Instantiate instances/values from classes/functions being tested
        description_col = np.asarray(['description one', 'description two', 'description three'])
        brand_col = np.asarray(['brand one', 'UNBRANDED', 'brand three'])
        self.local_char_set = ['RETAILER_DESC', 'LOCAL_GB_LOC_00004_BRAND']
        self.input_df = pd.DataFrame({'RETAILER_DESC': description_col, 'LOCAL_GB_LOC_00004_BRAND': brand_col})
        self.target_global_char = 'GLOBAL_BRAND_1'
        self.validate_data_handler = pd.DataFrame({'RETAILER_DESC': np.asarray(['description one', 'description three']),
                                                   'LOCAL_GB_LOC_00004_BRAND': np.asarray(['brand one', 'brand three'])})
        self.validate_data_preparation = pd.Series(['description one brand one', 'description two brand two', 'description three brand three', ])
        self.test_output = DataCleanerPrediction()
        self.test_output_data_handler = self.test_output.data_handler(self.input_df)
        self.test_output_data_preparation = self.test_output.data_preparation(self.input_df, self.local_char_set, self.target_global_char)
        print("TestDataCleanerPrediction:setup_:end")

    # ending test
    def tearDown(self):
        """ Post-test cleanup """
        # Do something based on test complexity/objectives/options
        print("TestDataCleanerPrediction:tearDown_")

    def test_data_handler(self):
        """ Test DataCleanerPrediction.data_handler method """
        self.assertEqual(first=self.validate_data_handler.shape, second=self.test_output_data_handler.shape, msg=None)

    def test_data_preparation(self):
        """ Test DataCleanerPrediction.data_preparation method """
        self.assertEqual(first=self.validate_data_preparation.shape, second=self.test_output_data_preparation.shape, msg=None)

    #@unittest.skip("Skip over integrity test")
    def test_DataCleanerPrediction_integrity(self):
        """ Testing DataCleanerPrediction returns an object """
        self.assertTrue(isinstance(self.test_output, object))
        self.assertTrue(isinstance(self.test_output_data_handler, pd.DataFrame))
        self.assertTrue(isinstance(self.test_output_data_preparation, pd.Series))