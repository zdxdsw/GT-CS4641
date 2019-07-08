import pandas as pd

from clustertesters import lr_KMeansTestCluster as kmtc


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
    letter_recognition = pd.read_csv("lr_IG.csv")
    # You can choose from "lr_PCA.csv", "lr_ICA.csv", "lr_RP.csv", "lr_IG.csv"
    dft, mapping = encode_target(letter_recognition, "class")
    #dft.to_csv('letternew.cvs')
    #dft2 = pd.read_csv("phishing.csv")

    # These two lines might be different. It depend on which .csv file is chosen.
    X = (dft.ix[:,1:])
    y = dft.ix[:, 0]
    print(y)
    #print X
    #print y
    tester = kmtc.KMeansTestCluster(X, y, clusters=range(1,31), plot=True, targetcluster=2, stats=True)
    tester.run()





