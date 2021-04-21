#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 24 10:36:08 2021

@author: Rahul
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# Imports the dataset and returns the dataframe object
class ImportData:

    fileName = 'CustomerData.xlsx'
    sheetName = 'Retrieve CustomerCreditRiskData'    
    df = None
    
    def __init__(self):
        self.df = pd.read_excel(self.fileName, sheet_name = self.sheetName)  
        self.transformingCategories()
        self.validatingDataQuality()
        
        
    def transformingCategories(self):
        LE = LabelEncoder()
        #Transforming predictor variables 
        self.df['status'] = LE.fit_transform(self.df['status'])
        self.df['foreignworker'] = LE.fit_transform(self.df['foreignworker'])
        self.df['purpose'] = LE.fit_transform(self.df['purpose'])
        self.df['gender'] = LE.fit_transform(self.df['gender'])
        self.df['job'] = LE.fit_transform(self.df['job'])
        self.df['employmentsince'] = LE.fit_transform(self.df['employmentsince'])
        self.df['otherdebtors'] = LE.fit_transform(self.df['otherdebtors'])
        self.df['otherinstallments'] = LE.fit_transform(self.df['otherinstallments'])
        self.df['property'] = LE.fit_transform(self.df['property'])
        self.df['savings'] = LE.fit_transform(self.df['savings'])
        self.df['credithistory'] = LE.fit_transform(self.df['credithistory']) 
        
        #Transforming target variable
        self.df['creditworthy'] = LE.fit_transform(self.df['creditworthy'])

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
        


