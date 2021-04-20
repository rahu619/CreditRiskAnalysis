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


    # Retrieving variables using
    # Recursive Feature Elimination
    # RFE with the logistic regression algorithm to select the top 3 features.
    def RetrieveVariablesByRFE(self):
        
        df = pd.read_excel(self.fileName, sheet_name = self.sheetName)
        cols = list(df.columns.values) #Make a list of all of the columns in the df
        cols.pop(cols.index('creditworthy')) #Remove b from list
        df = df[cols+['creditworthy']] #Create new dataframe with columns in the order you want
        
        array = df.values

        le = LabelEncoder()
        for i in range(21):
            array[:, i] = le.fit_transform(array[:, i])
    
        X = array[:, 0:20]
        y = array[:, 21]
        
               # The dependent / target variable
        y = self.df[['creditworthy']]
        
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

        # Create decision tree classifer object
        clf = RandomForestClassifier(random_state=0, n_jobs=-1)
        # Train model
        model = clf.fit(X, np.ravel(y, order='C'))
        
        
        # Trying 5 fold cross validation
        predicted = cross_val_predict(clf, X, np.ravel(y, order='C'), cv=5)
        
        accuracy = metrics.accuracy_score(y, predicted)
        
        print("Random Forest accuracy : %0.2f" % accuracy)       
        print()
        
        # Calculate feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Rearrange feature names so they match the sorted feature importances
        names = [df.columns[i] for i in indices]
        
        # print('importances: ', names)
          
        # Barplot: Add bars
        plt.bar(range(X.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(X.shape[1]), names, rotation=20, fontsize = 8)
        # Create plot title
        plt.title("Feature Importance")
        # Show plot
        plt.show()
    
        # Based on plot the important predictor variables
        # are Creit amount, Status, Age, Duration, Purpose, Credit History

