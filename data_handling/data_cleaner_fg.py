#!/usr/bin/env python
# feature_generator/data_handling/data_cleaner_fg.py
# Clean and prepare pandas python dataframe object data

import pandas as pd
import re


class DataCleanerFg(object):
    """Data Cleaner class for feature generator application"""
    def __init__(self):
        self.name = 'Data Cleaner'

    def data_handler(self, input_df):
        """Cycling through the entire pandas dataframe, insure bad entries are not present
        :param input_df             -- the pandas dataframe being cleansed
        :returns clean_pd_df,       -- the cleaned pandas dataframe
                 dirty_pd_df        -- the dirty records pandas dataframe
        """
        bad_list = ['NOT AVAILABLE', 'UNBRANDED', 'Image is Ready for Coding', '                ',
                    'UNKNOWNCODE', 'UNKNOWN', 'NONE', 'MISMATCH', 'UNCODED', 'NOT ON CATALOG',
                    'NO DESCRIPTION AVAILABLE', 'MISMATCH', 'UNIDENTIFIED', 'BOOYAHHHAHA', 'AVAILABLE,DESCRIPTION,NOT,']
        clean_pd_df = input_df.copy()
        num_records, num_variables = clean_pd_df.shape
        threshold = .11 * num_variables
        tmp_df = input_df.copy()
        dirty_pd_df = pd.DataFrame()
        for col in clean_pd_df.columns:
            # Check for nan's and nulls, if it's all bad, remove the column
            percent_null = (clean_pd_df[col].isnull().sum())/float(num_records)
            print("{} column in loop, and record count is {}".format(col, clean_pd_df.shape[0]))
            if percent_null > threshold:
                clean_pd_df = clean_pd_df.drop(col, 1)
                print("{} column removed for having {} percent null values".format(col, percent_null))
            else:
                temp_clean_pd_df = clean_pd_df[~clean_pd_df[col].isin(bad_list)]
                if temp_clean_pd_df.empty:
                    print("ERROR: The filtering of data based on filler code dismantled the entire data set")
                else:
                    clean_pd_df = clean_pd_df[~clean_pd_df[col].isin(bad_list)]
                    del(temp_clean_pd_df)
                tmp_df = tmp_df[tmp_df[col].isin(bad_list)]
                if tmp_df.empty:
                    pass
                else:
                    dirty_pd_df = pd.concat([dirty_pd_df, tmp_df], axis=0)

        def test_list(x):
            if isinstance(x, list):
                return (' '.join(x)).replace('\n', '')
            else:
                return str(x)

        def test_str(x):
            if isinstance(x, str):
                return re.findall(r"[\w']+", x)
            else:
                return x

        def is_str(x):
            if isinstance(x, str):
                return x
            else:
                return str(x)

        print("Shape of clean_pd_df BEFORE regex conditioning", clean_pd_df.shape)
        print("Shape of dirty_pd_df BEFORE regex conditioning", dirty_pd_df.shape)
        clean_pd_df = clean_pd_df.applymap(lambda x: test_str(x))
        clean_pd_df = clean_pd_df.applymap(lambda x: test_list(x))
        if dirty_pd_df.empty:
            pass
        else:
            dirty_pd_df = dirty_pd_df.applymap(lambda x: test_str(x))
            dirty_pd_df = dirty_pd_df.applymap(lambda x: test_list(x))

        for col in clean_pd_df.columns:
            clean_pd_df[col] = clean_pd_df[col].apply(lambda x: is_str(x))
            clean_pd_df[col] = clean_pd_df[col].replace('nan', 'BOOYAHHHAHA')
            clean_pd_df = clean_pd_df[~clean_pd_df[col].isin(['BOOYAHHHAHA'])]
            print("First 3 rows of {0} are {1}".format(col, clean_pd_df[col][:3]))

        print("Shape of clean_pd_df AFTER regex conditioning AND dropna", clean_pd_df.shape)
        print("Shape of dirty_pd_df AFTER regex conditioning", dirty_pd_df.shape)

        return clean_pd_df, dirty_pd_df

    def tt_split(self, pandas_df, target_vector, test_split_size=.35):
        """Custom Test-Train-Split method, insuring set of common values between test and training data sets

        :param pandas_df                            -- input dataframe
        :param target_vector                        -- target series
        :param test_split_size                      -- percentage to designate to test
        :returns X_train, X_test, y_train, y_test,  -- output test-train-split data sets
                 toss_df                            -- discarded records
        """

        # Isolate dependent and independent variables, in this case target_vector is a string
        target_vector = target_vector.map(lambda x: str(x).encode('utf-8'))
        y_value_counts = target_vector.value_counts()
        y_values_list = y_value_counts.index.tolist()
        toss_df = pd.DataFrame()
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        # Perform test/train/split based on count of records in dataset
        if test_split_size == 0:
            X_train = pandas_df.drop(target_vector, 1)
            X_test = pd.DataFrame()
            y_train = pandas_df[target_vector]
            y_test = pd.DataFrame()
        else:
            # MANUALLY SET EVERY SPLIT LEVEL UNTIL COUNT IS OVER 9
            for i in range(len(y_values_list)):
                cnt = y_value_counts[i]
                sub_df = pandas_df[target_vector == y_values_list[i]]
                sub_df = sub_df.sample(frac=1)
                if cnt == 1:
                    toss_df = pd.concat([toss_df, sub_df], axis=0)
                elif cnt == 2:
                    train_df = pd.concat([train_df, sub_df[:1]], axis=0)
                    test_df = pd.concat([test_df, sub_df[1:]], axis=0)
                elif cnt == 3:
                    train_df = pd.concat([train_df, sub_df[:2]], axis=0)
                    test_df = pd.concat([test_df, sub_df[2:]], axis=0)
                elif cnt == 4:
                    train_df = pd.concat([train_df, sub_df[:3]], axis=0)
                    test_df = pd.concat([test_df, sub_df[3:]], axis=0)
                elif cnt == 5:
                    train_df = pd.concat([train_df, sub_df[:3]], axis=0)
                    test_df = pd.concat([test_df, sub_df[3:]], axis=0)
                elif cnt == 6:
                    train_df = pd.concat([train_df, sub_df[:4]], axis=0)
                    test_df = pd.concat([test_df, sub_df[4:]], axis=0)
                elif cnt == 7:
                    train_df = pd.concat([train_df, sub_df[:5]], axis=0)
                    test_df = pd.concat([test_df, sub_df[5:]], axis=0)
                elif cnt == 8:
                    train_df = pd.concat([train_df, sub_df[:5]], axis=0)
                    test_df = pd.concat([test_df, sub_df[5:]], axis=0)
                elif cnt == 9:
                    train_df = pd.concat([train_df, sub_df[:6]], axis=0)
                    test_df = pd.concat([test_df, sub_df[6:]], axis=0)
                else:
                    test_split = cnt-int(cnt*test_split_size)
                    train_df = pd.concat([train_df, sub_df[:test_split]], axis=0)
                    test_df = pd.concat([test_df, sub_df[test_split:]], axis=0)

            # Perform Special Instructions, finalize return data sets
            if train_df.empty:
                print("Quality check found ERROR, empty data containers being passed to data_cleaner, check input data")
                X_train = pd.DataFrame()
                X_test = pd.DataFrame()
                y_train = pd.DataFrame()
                y_test = pd.DataFrame()
            else:
                X_train = pandas_df.loc[train_df.index]
                X_test = pandas_df.loc[test_df.index]
                y_train = pandas_df.loc[train_df.index]
                y_test = pandas_df.loc[test_df.index]

        return X_train, X_test, y_train, y_test, toss_df
