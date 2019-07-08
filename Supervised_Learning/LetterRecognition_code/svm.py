import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from keras.utils import np_utils
from sklearn.svm import SVC
from sklearn.model_selection import KFold

def main():
    data = []
    lable= []
    with open("letter-recognition.csv") as ifile:
        for line in ifile:
            tmp = line.strip().split(',')
            #print tmp[0]
            tmp1=[]
            for i in range(1,len(tmp)-1):
                tmp1.append(float(tmp[i]))
            data.append(tmp1)
            lable.append(tmp[0])
    # judge the performance as a function of training size
    #data = data[0:12000]
    #lable = lable[0:12000]
    x = np.array(data)
    y = np.array(lable)
    print ('x.shape() = ', x.shape)
    print ('\ny.shape() = ', y.shape)

    test_data = []
    test_lable= []
    with open("testing.csv") as ifile:
        for line in ifile:
            tmp = line.strip().split(',')
            #print tmp[0]
            tmp1=[]
            for i in range(1,len(tmp)-1):
                tmp1.append(float(tmp[i]))
            test_data.append(tmp1)
            test_lable.append(tmp[0])

    test_x = np.array(test_data)
    test_y = np.array(test_lable)
    print ('test_x.shape() = ', test_x.shape)
    print ('\ntest_y.shape() = ', test_y.shape)

    ''''' split the dataset into training and validation set '''
    ''''' split ratio: 80:20 '''
    kf = KFold(n_splits = 5, shuffle = True)
    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=7, gamma='auto', kernel='poly',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=0)
    i = 0
    total = 0.0
    training_total = 0.0
    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(x_train, y_train)
        training_score = model.score(x_train, y_train)
        score = model.score(x_val,y_val)
        print ('score ',i+1,' = ',score)
        total += score
        training_total += training_score
        i+=1
    validation_avg_score = total/i
    training_avg_score = training_total/i
    print('training score = ', training_avg_score)
    print('validation score = ', validation_avg_score)
    '''''precision & recall on validation set'''  

    #precision, recall, thresholds = precision_recall_curve(y_val, predictions)  
    #print(classification_report(y_val, predictions))
    model.fit(x,y)
    predictions = model.predict(test_x)
    print('testing score = ', model.score(test_x, test_y))
    '''''precision & recall on validation set'''  
    #precision, recall, thresholds = precision_recall_curve(y_val, predictions)  
    print(classification_report(test_y, predictions))

if __name__ == '__main__':
    main()


    
