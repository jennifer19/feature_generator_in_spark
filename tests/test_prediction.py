#!/usr/bin/env python
# ml_pipeline_27/ml_pipeline_27/test_prediction.py
import unittest
import numpy as np
import random
import string
from ml_pipeline_27.ml_pipeline_27_classes import Prediction


class TestPrediction(unittest.TestCase):
    """ Unit Testing /ml_pipeline_27/ml_pipeline_27_classes:Prediction """

    # prepare test
    def setUp(self):
        """ Setup for test, instantiate object / cycle method(s) for test object creation """
        print("TestPrediction:setup_:begin")
        # Instantiate instances/values from classes/functions being tested
        chars = []
        for i in range(100):
            chars.append("".join([random.choice(string.letters[:20]) for i in xrange(15)]))

        self.training_df = np.asarray(chars)
        self.testing_df = np.random.random_integers(0, high=1, size=100)
        self.test_output_ml_model = Prediction.optimized_svm_pipeline(self.training_df, self.testing_df)
        print("TestPrediction:setup_:end")

    # ending test
    def tearDown(self):
        """ Post-test cleanup """
        # Do something based on test complexity/objectives/options
        print("TestPrediction:tearDown_")

    def test_prediction_reporting(self):
        """ Test Prediction.prediction_reporting method
            NOTE: this method is decorated, and requires an integration test
        """
        pass

    #@unittest.skip("Skip over integrity test")
    def test_TestPrediction_integrity(self):
        """ Testing TestPrediction returns an object """
        self.assertTrue(isinstance(self.test_output_ml_model, object))