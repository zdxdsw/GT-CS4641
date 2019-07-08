from sklearn import datasets
import numpy
import pandas
from clustertesters import wine_KMeansTestCluster as kmtc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score

if __name__ == "__main__":
    wine_data = datasets.load_wine()
    #print breast_cancer
    X, y = wine_data.data, wine_data.target
    
    print X

    
    tester = kmtc.KMeansTestCluster(X, y, clusters=range(1,15),
                                    plot=True, targetcluster=10, stats=True)
    tester.run()
    #tester.visualize()
    
    '''
    #########################################################################
    #PCA
    pca = PCA(n_components=13, random_state=10)
    X_r = pca.fit(X).transform(X)
    X_pca = X_r
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ["b","g","r","c","m","y","k"]
    lw = 2

    for color, i in zip(colors, [1,2]):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 3], color=color, alpha=.8, lw=lw, label=i)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Wine Quality dataset')
    '''
    #########################################################################
    # ICA
    ica = FastICA(n_components=10, random_state=10)
    print(X.shape)
    X_r = ica.fit(X).transform(X)
    X_list = numpy.ndarray.tolist(X_r)
    
    for i in range(X_r.shape[0]):
        X_list[i].append(str(y[i]))
    df_ICA = pandas.DataFrame(X_list,
                              columns = ['0','1','2','3','4','5','6','7','8','9','class'])
    df_ICA.to_csv('wine_ICA.csv', index=False, header=True)
    import scipy
    df = pandas.DataFrame(numpy.ndarray.tolist(X_r))
    print(df.values)
 
    print(scipy.stats.kurtosis(df.values, axis=0, fisher=False, bias=False, nan_policy='propagate'))
    
    '''
    plt.figure()
    colors = ["b","g","r","c","m","y","k"]
    lw = 2

    for color, i in zip(colors, [1,2]):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 3], color=color, alpha=.8, lw=lw, label=i)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('ICA of Wine Quality dataset')
    '''
    
    #########################################################################
    # RP
    rca = GaussianRandomProjection(n_components=8, random_state=10)
    X_r = rca.fit_transform(X)
    X_list = numpy.ndarray.tolist(X_r)
    #print(X_r.shape)
    for i in range(X_r.shape[0]):
        X_list[i].append(str(y[i]))
    df_RCA = pandas.DataFrame(X_list,
                              columns = ['0','1','2','3','4','5','6','7','class'])
    df_RCA.to_csv('wine_RP.csv', index=False, header=True)
    import scipy
    df = pandas.DataFrame(numpy.ndarray.tolist(X_r))
    print(df.values)
 
    print('\nrp kurtosis: ',scipy.stats.kurtosis(df.values, axis=0, fisher=False, bias=False, nan_policy='propagate'))
    
    '''
    plt.figure()
    colors = ["b","g","r","c","m","y","k"]
    lw = 2

    for color, i in zip(colors, [1,2]):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 3], color=color, alpha=.8, lw=lw, label=i)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Random Projection of Wine Quality dataset')
    
    #################################################
    #Univariate feature selection (K best)

    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import mutual_info_classif

    X_new = SelectKBest(chi2, k=5).fit_transform(X, y)
    X_fs = X_new

    plt.figure()
    colors = ["b","g","r","c","m","y","k"]
    lw = 2

    for color, i in zip(colors, [1,2]):
        plt.scatter(X_new[y == i, 0], X_new[y == i, 3], color=color, alpha=.8, lw=lw, label=i)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Chi square feature selection of Wine Quality dataset')
    plt.show()
    '''






