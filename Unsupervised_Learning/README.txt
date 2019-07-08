CS 4641 Project3 
Name: CHANG Yingshan
GT Student ID: 903457645
GT Account: ychang363

Run Clustering Algorithms: wine_kmeans.py, wine_em.py, lr_kmeans.py, lr_em.py
Install the required python packages, navigate to the folder containing the .py file and type the following command line: "python XXX.py"

Run Dimensionality Reduction Algorithms: 
1. PCA: PCA is done by Weka under "Select attributes" menu.
2. ICA: ICA is implemented by python using sklearn.decomposition.FastICA. Run wine_kmeans.py and lr_kmeans.py, the resulting transformed datasets will be stored in "wine_ICA.csv" and "lr_ICA.csv".
3. RP: RP is implemented by python using sklearn.random_projection.GaussianRandomProjection. Run wine_kmeans.py and lr_kmeans.py, the resulting transformed datasets will be stored in "wine_RP.csv" and "lr_RP.csv".
4. Info Gain: Info Gain is done by Weka under "select attribute" menu.

Re-run Clustering Algorithms: wine_dr_kmeans.py, wine_dr_em.py, lr_dr_kmeans.py, lr_dr_em.py
Install the required python packages, navigate to the folder containing the .py file and type the following command line: "python XXX.py"

Neural Network and Dimensionality Reduction:
1. Datasets for this part can be found in "Datasets_after_Dimensionality_Reduction" folder.
2. Open Weka GUI, open a dataset under "Preprocess" menu.
3. Convert the "class" column from numeric to nominal on Weka GUI (filters->unsupervised -> attributes -> NumericToNominal) 
4. Choose "MultiLayerPerceptron" as the classifier under "Classify" menu. 

Clustering and Neural Network:
1. Datasets for this part can be found in "Neural_Network_and_Clustering" folder.
2. Open Weka GUI, open a dataset under "Preprocess" menu.
3. Convert the "class" column from numeric to nominal on Weka GUI (filters->unsupervised -> attributes -> NumericToNominal) 
4. Choose "MultiLayerPerceptron" as the classifier under "Classify" menu. 