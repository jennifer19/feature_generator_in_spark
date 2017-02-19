=========================================
Apache Spark Feature Generator via Python
=========================================

--------
Overview
--------

1. This is a custom python module designed to extract the most statistically significant features of a text-based data set.
2. This module is designed for an Apache Spark environment, with the module to be wrapped as an egg file, saved onto a Hadoop cluster (hdfs), and consumed within a Spark program
3. The repo therefore has two components:
 * The package to wrap up as the egg file such that it can be run in Spark
 * The spark_main.py file that is the actual pyspark program that runs on the Spark cluster


Setup file, egg file generation, spark_main.py
===============================================

Update and inspect the setup.py file to insure it matches your specifications

.. image:: ![fg_package_setup](https://cloud.githubusercontent.com/assets/20135017/23099652/c6155a78-f631-11e6-8965-94a6183a2a00.png)


To generate the egg file, run a standard egg distribution generation in the terminal at the codebase loadpoint
       
        **~/PycharmProjects/feature_generator$ python setup.py bdist_egg**
        

Finally, spark_main.py is the actual Spark program that will get loaded/launched for the Spark job 
    $ spark-submit ..path/to/spark_main.py
    
Note: In spark_main.py the path to the egg file (setup for Hadoop here, otherwise insure the Spark worker nodes have access to the file via the identical path to it)
    Using the 'addPyFile' method (in spark_main.py)::
    
        sc = SparkContext()
        sc.addPyFile('hdfs://<hostname>.<domain>.<org>:<port>/path/to/feature_generator.egg')
        sqlsc = SQLContext(sc)



--------------------------
Python 2/3 Compatibility:
--------------------------
- The feature generator package is written for python 2.7 with a best attempt for Python 3.5+ compatibility
- The metaclass definition on the abstract base class(es) for 3.5 wouldn't work in 2.7, and the string handling is different (a well-documented condition, thus not elaborated on here) between the two versions.



