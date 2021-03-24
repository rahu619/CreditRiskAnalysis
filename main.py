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

# In an ideal scenario, dataplot would happen before choosing the apt algorithm

processObj = Process(fileName, sheetName)

X, y = processObj.RetriveVariablesPostManualProcessing()

algorithmObj = Algorithm(True) #not persisting model for now

# print(X)

[X_train, X_test, y_train, y_test], prediction = algorithmObj.KNNApproach(X, y)


accuracy = metrics.accuracy_score(y_test, prediction)

print("kNN predictions :", prediction)

print("kNN accuracy :", accuracy)       

# seems like we can' achieve over 75% of accuracy with the kNN approach

