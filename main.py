# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:25:32 2021

@author: Rahul
"""

# Attempt kNN and then SVM
# compare the accuracies
# if more datapoints or datapoints overlap -> SVM

import pandas as pd
import seaborn as sns
from sklearn import metrics
from importData import ImportData
from classifier import Classifier

# Trying to predict if the customer is creditworthy
# As it's already labelled ; we'll use supervised M.L. 
# and since it's a category ('Worthy' or 'Not worthy'), it'll fall under 
# a Classification problem

# Retrieving the imported, tranformed dataframe 
df = ImportData().df

# print('-------------- Dataframe Head Info -------------\n')
# print(df.head())
# print()

# print('---- Validating Multivariate outliers in n-dimensional space ... \n')
# print(df.describe())
# print()

# seems like we have lot of points for credit amount outside the box of observation    
# sns.boxplot(x=df['creditamount'])
# sns.boxplot(x=df['status'])
# sns.boxplot(x=df['age'])

# Instantiating classifiers
classifierObj = Classifier(df, usePersistedModel = False)

# # Plotting important features.
# Please check the plots windows in Spyder IDE to view it.
classifierObj.plotImportantFeatures()

# Getting k-nearest neighbor accuracy score
classifierObj.knnApproach()

# Getting linear support vector machine accuracy score
classifierObj.svmApproach()

# Getting random forest decision tree accuracy score
classifierObj.randomForestApproach()


# seems like we can't achieve over 75% of accuracy with the 
# Random Forest Decision Tree approach


# SVM - if there are many features
# potentially ideal for both classification and regression
# Trying 5 fold cross validation as well, instead of train,test and split 
# as the samples we have are limited
 
# Potential downsides will be more computational power









