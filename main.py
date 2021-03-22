# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:25:32 2021

@author: Rahul
"""

# Attempt kNN and then SVM
# compare the accuracies
# if more datapoints or datapoints overlap -> SVM

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import variables

data = pd.read_excel('CustomerData.xlsx', sheet_name='Retrieve CustomerCreditRiskData')

print(data.head())


# Trying to predict if the customer is creditworthy
# As it's already labelled ; we'll use supervised M.L. 
# and since it's a category ('Worthy' or 'Not worthy'), it'll fall under Classification problem

#Unpacking the X, y tuple
X, y = variables.retrieveVariablesManually(data)
# print(X, y)


#Preprocessing step
# Transforming the textual data to numbers
# as we can't feed into the algorithm  
Le = LabelEncoder()

# column wise
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
    
    
# print(X)

# transforming y
label_mapping = {   
    'Not Worthy': 0,
    'Worthy': 1
}

y = y.copy()
y['creditworthy'] = y['creditworthy'].map(label_mapping)
y = np.array(y)

# print(y)

# create model
# k = 25
knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train)
  
prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)

print("predictions :", prediction)

print("accuracy :", accuracy)       

