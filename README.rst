This is a custom python module designed to extract the most statistically significant features of a data set

This module is designed for an Apache Spark environment, to be distributed within the Spark cluster
    An *.egg file must be generated and pushed to the cluster.
        To generate the egg file, run this at the command prompt of the pycharm terminal
        ~/PycharmProjects/feature_generator$ python setup.py bdist_egg

    In this example, it's assumed the egg file has been generated from this package and placed on the
    hadoop cluster with its respective path, called in the Spark main program via the 'addPyFile' method:
    (In spark_main.py):
        sc = SparkContext()
        sc.addPyFile('hdfs://<hostname>.<domain>.<org>:<port>/path/to/feature_generator.egg')
        sqlsc = SQLContext(sc)

Compatibility:
- The feature generator package is written for python 2.7 with a best attempt for Python 3.5+ compatibility
 The metaclass definition on the abstract base class(es) for 3.5 wouldn't work in 2.7, and the string handling
 is different (a well-known issue, thus not elaborated on here) between the two versions.



