#!/usr/bin/env python
# ml_pipeline_27/tests/test_dataCleanerTraining.py
import unittest
import numpy as np
import pandas as pd
#from data_handling.data_cleaner import DataCleanerTraining
from data_handling.data_cleaner_fg import DataCleanerFg


class TestDataCleanerTraining(unittest.TestCase):
    """ Unit Testing /data_handling/data_cleaner:DataCleanerTraining """

    # prepare test
    def setUp(self):
        """ Setup for test, instantiate object / cycle method(s) for test object creation """
        print("TestDataCleanerTraining:setup_:begin")
        # Instantiate instances/values from classes/functions being tested
        description_col = np.asarray(['description one', 'description two', 'description three'])
        brand_col_local = np.asarray(['brand one', 'UNBRANDED', 'brand three'])
        brand_col_global = np.asarray(['global brand one', 'UNBRANDED', 'global brand three'])
        self.local_char_set = ['RETAILER_DESC', 'LOCAL_GB_LOC_00004_BRAND']
        self.input_df = pd.DataFrame({'RETAILER_DESC': description_col,
                                      'LOCAL_GB_LOC_00004_BRAND': brand_col_local})
        self.pandas_df = pd.DataFrame({'RETAILER_DESC': description_col,
                                       'LOCAL_GB_LOC_00004_BRAND': brand_col_local,
                                       'GLOBAL_BRAND_1': brand_col_global})
        self.target_global_char = 'GLOBAL_BRAND_1'
        self.validate_data_handler_clean = pd.DataFrame({'RETAILER_DESC': np.asarray(['description one', 'description three']),
                                                         'LOCAL_GB_LOC_00004_BRAND': np.asarray(['brand one', 'brand three'])})
        self.validate_data_handler_dirty = pd.DataFrame({'RETAILER_DESC': np.asarray(['description two']),
                                                         'LOCAL_GB_LOC_00004_BRAND': np.asarray(['UNBRANDED'])})
        self.test_output = DataCleanerTraining()
        self.test_output_data_handler_clean, self.test_output_data_handler_dirty = self.test_output.data_handler(self.input_df)
        self.test_output_tt_split = self.test_output.tt_split(self.pandas_df, self.local_char_set, self.target_global_char)
        print("TestDataCleanerTraining:setup_:end")

    # ending test
    def tearDown(self):
        """ Post-test cleanup """
        # Do something based on test complexity/objectives/options
        print("TestDataCleanerTraining:tearDown_")

    def test_data_handler(self):
        """ Test DataCleanerTraining.data_handler method """
        self.assertEqual(first=self.validate_data_handler_clean.shape, second=self.test_output_data_handler_clean.shape, msg=None)
        self.assertEqual(first=self.validate_data_handler_dirty.shape, second=self.test_output_data_handler_dirty.shape, msg=None)

    #@unittest.skip("Skip over integrity test")
    def test_tt_split(self):
        """ Test DataCleanerTraining.data_preparation method """
        self.assertEqual(first=5, second=len(self.test_output_tt_split), msg=None)

    #@unittest.skip("Skip over integrity test")
    def test_DataCleanerPrediction_integrity(self):
        """ Testing DataCleanerTraining returns an object """
        self.assertTrue(isinstance(self.test_output, object))
        self.assertTrue(isinstance(self.test_output_data_handler_clean, pd.DataFrame))
        self.assertTrue(isinstance(self.test_output_data_handler_dirty, pd.DataFrame))
        self.assertTrue(isinstance(self.test_output_tt_split, tuple))