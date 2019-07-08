import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import KFold

import numpy
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
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y)
    # convert integers to dummy variables (one hot encoding)
    y = np_utils.to_categorical(encoded_Y)

    encoded_test_Y = encoder.fit_transform(test_y)
    test_y = np_utils.to_categorical(encoded_test_Y)

    ''''' split the dataset into training and validation set '''
    ''''' split ratio: 80:20 '''
    model = Sequential()
    model.add(Dense(output_dim=20, input_dim=15, init='random_uniform', activation='tanh'))
    #model.add(Dropout(0.1))
    #model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(output_dim=26, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    kf = KFold(n_splits = 5, shuffle = True)
    i = 0
    total = 0.0
    training_total = 0.0
    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(x_train, y_train, epochs=20, batch_size=16,  verbose=0)
        training_score = model.evaluate(x_train, y_train)[1]
        score = model.evaluate(x_val,y_val)[1]
        print ('score ',i+1,' = ',score)
        total += score
        training_total += training_score
        i+=1
    validation_avg_score = total/i
    training_avg_score = training_total/i
    print('training score = ', training_avg_score)
    print('validation score = ', validation_avg_score)
    
    model.fit(x, y, epochs=20, batch_size=16,  verbose=0)
    # calculate predictions
    #predictions = model.predict(test)
    #predictions=predictions.tolist()
    #answer =[i.index(max(i)) for i in predictions]
    
    print('testing score = ', model.evaluate(test_x, test_y))
    '''''precision & recall on validation set'''  
    #precision, recall, thresholds = precision_recall_curve(y_val, predictions)  
    #print(classification_report(test_y, answer))

if __name__ == '__main__':
    main()


    
