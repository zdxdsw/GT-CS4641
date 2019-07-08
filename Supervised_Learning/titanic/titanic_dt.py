from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import scipy as sp
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from keras.utils import np_utils
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def main():
    data = []
    lable= []
    with open("titanic.csv") as ifile:
        for line in ifile:
            tmp = line.strip().split(',')
            tmp1=[]
            for i in [2,3,4,5,6,8,10]:
                tmp1.append(tmp[i])
            data.append(tmp1)
            lable.append(tmp[1])
    data.remove(data[0])
    lable.remove(lable[0])
    
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

    test_data = []
    with open("test.csv") as ifile:
        for line in ifile:
            tmp = line.strip().split(',')
            tmp1=[]
            for i in [1,2,3,4,5,6,7]:
                tmp1.append(tmp[i])
            test_data.append(tmp1)
    test_data.remove(test_data[0])

    for i in range(len(test_data)):
        for j in [0,2,3,4,5]:
            if test_data[i][j] == '':
                test_data[i][j] = -1.0
            test_data[i][j] = float(test_data[i][j])
        if test_data[i][1] == 'female':
            test_data[i][1] = 0
        elif test_data[i][1] == 'male':
            test_data[i][1] = 1
        else:
            test_data[i][1] = -1
        if test_data[i][6] == 'C':
            test_data[i][6] = 0
        elif test_data[i][6] == 'S':
            test_data[i][6] = 1
        elif test_data[i][6] == 'Q':
            test_data[i][6] = 2
        else:
            test_data[i][6] = -1

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
    
    x = np.array(data)
    x = cleandata(x)
    # judge the performance as a function of training size
    x = x[0:400]
    lable = lable[0:400]
    y = np.array(lable)
    test_x = np.array(test_data)
    test_x = cleandata(test_x)
    print ('x.shape() = ', x.shape)
    print ('\ny.shape() = ', y.shape)
    print ('x_test.shape() = ', test_x.shape)

    ''''' split the dataset into training and validation set '''
    ''''' split ratio: 80:20 '''
    kf = KFold(n_splits = 5, shuffle = True)
    dt = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=3, max_depth=50)  
    i = 0
    total = 0.0
    training_total = 0.0
    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        dt.fit(x_train, y_train)
        training_score = dt.score(x_train, y_train)
        score = dt.score(x_val,y_val)
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
    dt.fit(x,y)
    test_df = pd.read_csv('test.csv')
    predictions = dt.predict(test_x)
    test_df['Survived'] = np.array(predictions)
    test_df = test_df.drop(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], axis=1)
    print('shape = ', test_df.shape)
    header = ['PassengerId', 'Survived']
    test_df.to_csv('dt_submission.csv', columns=header, index=False, header=True)

if __name__ == '__main__':
    main()


    
