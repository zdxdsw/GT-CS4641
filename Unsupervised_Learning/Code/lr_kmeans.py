import pandas

from clustertesters import lr_KMeansTestCluster as kmtc
import numpy
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


def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column].replace(map_to_int, inplace=True)
    return (df_mod, map_to_int)

if __name__ == "__main__":
    letter_recognition = pandas.read_csv("letter.csv")
    
    dft, mapping = encode_target(letter_recognition, "class")
    dft.to_csv('letternew.csv')
    #print dft
    #dft2 = pd.read_csv("phishing.csv")
    X = (dft.ix[:,1:])
    y = dft.ix[:, 0]
    #print X
    print y
    
    tester = kmtc.KMeansTestCluster(X, y, clusters=range(1,31), plot=True, targetcluster=2, stats=True)
    tester.run()
    
    #########################################################################
    # ICA
    ica = FastICA(n_components=13, random_state=10)
    print(X.shape)
    X_r = ica.fit(X).transform(X)
    X_list = numpy.ndarray.tolist(X_r)
    
    for i in range(X_r.shape[0]):
        X_list[i].append(str(y[i]))
    df_ICA = pandas.DataFrame(X_list,
                              columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','class'])
    df_ICA.to_csv('lr_ICA.csv', index=False, header=True)
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
    #Random Projection feature transformation

    rca = GaussianRandomProjection(n_components=13, random_state=10)
    X_r = rca.fit_transform(X)
    X_list = numpy.ndarray.tolist(X_r)
    #print(X_r.shape)
    for i in range(X_r.shape[0]):
        X_list[i].append(str(y[i]))
    df_RCA = pandas.DataFrame(X_list,
                              columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','class'])
    df_RCA.to_csv('lr_RP.csv', index=False, header=True)
    import scipy
    df = pandas.DataFrame(numpy.ndarray.tolist(X_r))
    print(df.values)
    




