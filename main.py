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
from importData import ImportData
from classifier import Classifier


# Trying to predict if the customer is creditworthy
# As it's already labelled ; we'll use supervised M.L. 
# and since it's a category ('Worthy' or 'Not worthy'), it'll fall under 
# a Classification problem

# Importing data and retrieving the dataframe
df = ImportData().df

print('--------------Dataframe Head Info--------------')
print(df.head())
print()


# Instantiating classifiers

# TODO: turn usePersistedModel on after testing
classifierObj = Classifier(df, usePersistedModel = False)

classifierObj.plotImportantFeatures()

classifierObj.knnApproach()

classifierObj.svmApproach()

classifierObj.randomForestApproach()


# Analysing Features/ Predictor variables

# processObj.RetrieveVariablesByRFE()

# X, y = processObj.RetrieveVariablesManually()

# # initializing the algorithm class
# algorithmObj = Algorithm(usePersistedModel = True)

# # Executing and evaluating prediction score

# # trying knn prediction
# algorithmObj.KNNApproach(X, y)    


# # seems like we can't achieve over 75% of accuracy with the KNN approach



# SVM - if there are many features
# potentially ideal for both classification and regression
# Trying 5 fold cross validation as well, instead of train,test and split 
# as the samples we have are limited 
# Potential downsides will be more computational power
# algorithmObj.SVMApproach(X, y)








