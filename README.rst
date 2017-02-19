=========================================
Apache Spark Feature Generator via Python
=========================================

----
Overview
----

1. This is a custom python module designed to extract the most statistically significant features of a text-baesd data set.
2. This module is designed for an Apache Spark environment, with the module to be wrapped as an egg file stored in a Hadoop cluster, and consumed within a Spark program
3. The repo therefore has two components:
 * The package to wrap up as the egg file such that it can be run in Spark
 * The spark_main.py file that is the actual pyspark program that runs on the Spark cluster


Setup file, egg file generation, spark_main.py
===========

.. image:: ![fg_package_setup](https://cloud.githubusercontent.com/assets/20135017/23099652/c6155a78-f631-11e6-8965-94a6183a2a00.png)

This module is designed for an Apache Spark environment, to be distributed within the Spark cluster
    An *.egg file must be generated and pushed to the cluster.
        To generate the egg file, run this at the command prompt of the pycharm terminal
        
        **~/PycharmProjects/feature_generator$ python setup.py bdist_egg**
        

    In this example, it's assumed the egg file has been generated from this package and placed on the
    hadoop cluster with its respective path, called in the Spark main program via the 'addPyFile' method:
    (In spark_main.py):
        sc = SparkContext()
        sc.addPyFile('hdfs://<hostname>.<domain>.<org>:<port>/path/to/feature_generator.egg')
        sqlsc = SQLContext(sc)



----
Python 2/3 Compatibility:
----
- The feature generator package is written for python 2.7 with a best attempt for Python 3.5+ compatibility
- The metaclass definition on the abstract base class(es) for 3.5 wouldn't work in 2.7, and the string handling::
 is different (a well-documented condition, thus not elaborated on here) between the two versions.



