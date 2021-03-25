#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 24 10:36:08 2021

@author: Rahul
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Preprocessing , 
# feature selection,
# transformation


# returns the predictor and target variables.
class Process:

    fileName = ''
    sheetName = ''

    def __init__(self, filename, sheetname):
        self.fileName = filename
        self.sheetName = sheetname
            
    def RetrieveVariablesManually(self):
        data = pd.read_excel(self.fileName, sheet_name = self.sheetName)
        
        # print(data.head(n=2))
        # print(data.dtypes)
       
       # The independent / predictor variables
        X = data[[
            # 'foreignworker',
            'status',            
            # 'credithistory',
            #'savings',
            # 'employmentsince',
            # 'creditamount',
            # 'age',
            'otherinstallments'
            ]].values

       # The dependent / target variable
        y = data[['creditworthy']]
        
               
        # plt.scatter(X, y)
        # plt.show()

       # Preprocessing step
       # Transforming the textual data to numbers
       # as we can't the data directly into the algorithm
        Le = LabelEncoder()

       # transforming X column wise
        for i in range(len(X[0])):
            X[:, i] = Le.fit_transform(X[:, i])

       # transforming y
        label_mapping = {'Not Worthy': 0, 'Worthy': 1}

        y = y.copy()
        y['creditworthy'] = y['creditworthy'].map(label_mapping)
        y = np.array(y)

        return X, y


    # Retrieving variables using
    # Recursive Feature Elimination
    # RFE with the logistic regression algorithm to select the top 3 features.
    def RetrieveVariablesByRFE(self):
        
        df = pd.read_excel(self.fileName, sheet_name = self.sheetName)
        array = df.values
        
        le = LabelEncoder()
        for i in range(21):
            array[:,i] = le.fit_transform(array[:,i])
    
        X = array[:, 0:19]
        y = array[:, 20]
        y = y.astype('int')

    
        # feature extraction
        model = LogisticRegression(solver='lbfgs')
       
        # getting the four most important features
        rfe = RFE(model, 4)
     
        fit = rfe.fit(X, y)
        
        print("Num Features: %d" % fit.n_features_)
        print("Selected Features: %s" % fit.support_)
        print("Feature Ranking: %s" % fit.ranking_)
        
        # Based on RFE , Columns 1, 2, 5, 9 are important
        
        

