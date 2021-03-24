#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 24 10:36:08 2021

@author: Rahul
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# For preprocessing , 
# feature selection,
# transformation
# returns the predictor and target variables.

class Process:

    fileName = ''
    sheetName = ''

    def __init__(self, filename, sheetname):
        self.fileName = filename
        self.sheetName = sheetname

   # Manually choosing features that I believe
   # which would be apt for the perfect prediction.

    def RetriveManuallyProcessedVariables(self):

        data = pd.read_excel(self.fileName, sheet_name = self.sheetName)
       # print(data.head())
       
        # The independent / predictor variables
        X = data[[
            'foreignworker',
            'status',
            'credithistory',
            'savings',
            'employmentsince',
            'creditamount',
            'age',
            ]].values

       # The dependent / target variable
        y = data[['creditworthy']]

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
