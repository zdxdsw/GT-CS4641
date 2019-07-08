import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import mlrose
from sklearn.metrics import accuracy_score
def main():
    data = []
    label= []
    with open("titanic.csv") as ifile:
        for line in ifile:
            tmp = line.strip().split(',')
            tmp1=[]
            for i in [2,3,4,5,6,8,10]:
                tmp1.append(tmp[i])
            data.append(tmp1)
            label.append(tmp[1])
    data.remove(data[0])
    label.remove(label[0])
    
    for i in range(len(data)):
        for j in [0,2,3,4,5]:
            if data[i][j] == '':
                data[i][j] = -1.0
            data[i][j] = float(data[i][j])
        if data[i][1] == 'female':
            data[i][1] = 0
        elif data[i][1] == 'male':
            data[i][1] = 1
        else:
            data[i][1] = -1
        if data[i][6] == 'C':
            data[i][6] = 0
        elif data[i][6] == 'S':
            data[i][6] = 1
        elif data[i][6] == 'Q':
            data[i][6] = 2
        else:
            data[i][6] = -1

    childAge = 18
    def getIdentity(passenger):
        age, sex = passenger
        
        if age < childAge:
            return 'child'
        elif sex == 1.0:
            return 'male_adult'
        else:
            return 'female_adult'
    
    def cleandata(nparray):
        train_df = pd.DataFrame(data = nparray,
                                columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
        train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
        train_df['IsAlone'] = 0
        train_df.loc[train_df.FamilySize == 1, 'IsAlone'] = 1
        #train_df['NameLength'] = train_df.Name.apply(lambda x : len(x))
        train_df = pd.concat([train_df, pd.DataFrame(train_df[['Age', 'Sex']].apply(getIdentity, axis = 1), columns = ['Identity'])], axis = 1)
        train_df = pd.concat([train_df, pd.get_dummies(train_df.Identity)], axis = 1)
        #scaler = MinMaxScaler()
        select_features = ['Pclass', 'Sex', 'child', 'female_adult', 'male_adult',
                           'FamilySize', 'IsAlone', 'Age', 'SibSp', 'Fare', 'Embarked']
        scaler = StandardScaler()
        train_df = scaler.fit_transform(train_df[select_features])
        return train_df.copy()
    x = np.array(data[100:])
    x = cleandata(x)
    y = np.array(label[100:])
    test_x = np.array(data[:100])
    test_x = cleandata(test_x)
    test_y = np.array(label[:100])
    print ('x.shape() = ', x.shape)
    print ('\ny.shape() = ', y.shape)
    print ('x_test.shape() = ', test_x.shape)

    np.random.seed(3)
    model = mlrose.NeuralNetwork(hidden_nodes=[2,2], activation='tanh',
                                 algorithm='random_hill_climb', max_iters=1000,
                                 bias=True, is_classifier=True, learning_rate=0.01,
                                 early_stopping=True, clip_max=5, max_attempts=100)
    model2 = mlrose.NeuralNetwork(hidden_nodes=[2,2], activation='tanh',
                                 algorithm='simulated_annealing', max_iters=10000,
                                 bias=True, is_classifier=True, learning_rate=0.01,
                                 early_stopping=True, clip_max=5, max_attempts=100)
    model3 = mlrose.NeuralNetwork(hidden_nodes=[2,2], activation='tanh',
                                 algorithm='genetic_alg', max_iters=200,
                                 pop_size=500, mutation_prob=0.4,
                                 bias=True, is_classifier=True, learning_rate=0.0001,
                                 early_stopping=True, clip_max=5, max_attempts=10)
    n = 1
    train_acc_list = []
    test_acc_list = []
    for i in range(n):
        model3.fit(x,y)
        train_pred = model3.predict(x)
        train_pred1 = []
        for i in range(len(train_pred)):
            if (train_pred[i][0] == 0):
                train_pred1.append('0')
            else:
                train_pred1.append('1')
        train_accuracy = accuracy_score(y, train_pred1)
        test_pred = model3.predict(test_x)
        test_pred1 = []
        for i in range(len(test_pred)):
            if (test_pred[i][0] == 0):
                test_pred1.append('0')
            else:
                test_pred1.append('1')
        test_accuracy = accuracy_score(test_y, test_pred1)
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)
    print('training accuracy = ', sum(train_acc_list)/len(train_acc_list))
    print('\ntesting accuracy = ', sum(test_acc_list)/len(test_acc_list))


if __name__ == '__main__':
    main()

