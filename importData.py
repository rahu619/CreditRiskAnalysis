#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 24 10:36:08 2021

@author: Rahul
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_predict
from sklearn import metrics

# Imports the dataset and returns the dataframe object
class ImportData:

    fileName = 'CustomerData.xlsx'
    sheetName = 'Retrieve CustomerCreditRiskData'    
    df = None
    
    def __init__(self):
        self.df = pd.read_excel(self.fileName, sheet_name = self.sheetName)       
        self.validatingDataQuality();
        
    # verifying data quality
    def validatingDataQuality(self):
        print("---------------------------------------------------------------")
        print(self.df.info()) # We could verify if there are any null values
        # for col in self.df.columns:
        #     print(col, len(self.df[col].unique()), self.df[col].unique)
        print("---------------------------------------------------------------")        
        print()
        
    # Getter for Dataframe  
    @property
    def dataFrame(self):
        return self.df       
        
    def RetrieveVariablesManually(self):
        # print(data.head(n=2))
        # print(data.dtypes)
       
       # The independent / predictor variables
        X = self.df[[
            'foreignworker',
            'status',            
            'credithistory',
            'savings',
            'employmentsince',
            'creditamount',
            'age',
            'otherinstallments'
            ]].values

       # The dependent / target variable
        y = self.df[['creditworthy']]
        
               
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


