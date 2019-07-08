
from sklearn import  datasets, metrics
from clustertesters import wine_ExpectationMaximizationTestCluster as emtc
import pandas as pd


if __name__ == "__main__":
    wine_data = datasets.load_wine()
    #print breast_cancer
    X, y = wine_data.data, wine_data.target

    tester = emtc.ExpectationMaximizationTestCluster(X, y, clusters=range(1,11), plot=True, targetcluster=10, stats=True)
    tester.run()

