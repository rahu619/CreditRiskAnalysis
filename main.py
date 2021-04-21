# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:25:32 2021

@author: Rahul
"""

import numpy as np;
import pandas as pd
from sklearn import metrics
from processData import ProcessData
from classifier import Classifier
from IPython import get_ipython

# get_ipython().run_line_magic('matplotlib', 'inline')


# Helper method to intialize and return classifier class object
def initializeClassifier(df, features = None):
    classifierObj = Classifier(df, usePersistedModel = False)

    if features is not None:
        classifierObj.initializeVariables(features)
        
    return classifierObj

# The main method
def main():    
    # Trying to predict if the customer is creditworthy
    # As it's already labelled ; we'll use supervised M.L. 
    # and since it's a category ('Worthy' or 'Not worthy'), it'll fall under 
    # a Classification problem
    
    # Retrieving the imported, tranformed dataframe 
    processDataObj = ProcessData()
    df = processDataObj.getdataFrame

    # Instantiating classifiers
    classifierObj = initializeClassifier(df)
    
    # Plotting important features.
    classifierObj.plotImportantFeatures()
    
    # Based on the above Feature Importance bar chart, we have identified the top features
    topFeatures = ['status',         
                    'duration',
                    'creditamount',
                    'credithistory',
                    'savings',
                    'purpose'] 
    
    # Redefining the X, y variables based on our top features
    classifierObj.initializeVariables(topFeatures)
    
    # Step : Identifying data outliers
    
    # Visualizing outliers in our top features
    processDataObj.visualizeOutliers(topFeatures)
    
    # Removed outliers and retrieving new filtered dataframe
    filtered_df = processDataObj.removeOutliers()
    
    # Initializing classifier class again with our filtered dataframe
    initializeClassifier(filtered_df)
    
    print("--------------- Analysing accuracy scores of different algorithms ----------------\n")
    # Step : Predicting using three different algorithms
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



# Just to make sure this class can't be imported but executed alone
# as it's the entry point
if __name__ == '__main__':
    main()
    










