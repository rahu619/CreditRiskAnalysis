# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:25:32 2021

@author: Rahul
"""

# Attempt kNN and then SVM
# compare the accuracies
# if more datapoints or datapoints overlap -> SVM

import pandas as pd
from sklearn import metrics
from process import Process
from algorithms import Algorithm


# Trying to predict if the customer is creditworthy
# As it's already labelled ; we'll use supervised M.L. 
# and since it's a category ('Worthy' or 'Not worthy'), it'll fall under Classification problem

fileName = 'CustomerData.xlsx'
sheetName = 'Retrieve CustomerCreditRiskData'

# In an ideal scenario, dataplot could happen before choosing the apt algorithm


# Selected features and knn prediction

processObj = Process(fileName, sheetName)

X, y = processObj.RetrieveVariablesManually()

# initializing the algorithm class
algorithmObj = Algorithm(usePersistedModel = True)

# Evaluating prediction score
[X_train, X_test, y_train, y_test], prediction = algorithmObj.KNNApproach(X, y)

# TODO: refactor this later. Maybe introduce a Evaluate class
accuracy = metrics.accuracy_score(y_test, prediction)


print("KNN predictions :", prediction)
print("KNN accuracy :", accuracy)       


# seems like we can't achieve over 75% of accuracy with the kNN approach


# SVM - if many features
# ideal for both classification and regression





