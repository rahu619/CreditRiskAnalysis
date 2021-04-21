#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 24 10:36:08 2021

@author: Rahul
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


# Imports the dataset and returns the dataframe object
# Data transform class
class ProcessData:

    fileName = 'CustomerData.xlsx'
    sheetName = 'Retrieve CustomerCreditRiskData'    
    df = None
    
    def __init__(self):
        self.df = pd.read_excel(self.fileName, sheet_name = self.sheetName)  
        self.printDataQuality()
        self.transformingCategories()  
        
    def transformingCategories(self):
        LE = LabelEncoder()
        #Transforming predictor variables 
        self.df['foreignworker'] = LE.fit_transform(self.df['foreignworker'])
        self.df['status'] = LE.fit_transform(self.df['status'])
        self.df['credithistory'] = LE.fit_transform(self.df['credithistory']) 
        self.df['purpose'] = LE.fit_transform(self.df['purpose'])
        self.df['savings'] = LE.fit_transform(self.df['savings'])
        self.df['employmentsince'] = LE.fit_transform(self.df['employmentsince'])
        self.df['otherdebtors'] = LE.fit_transform(self.df['otherdebtors'])
        self.df['property'] = LE.fit_transform(self.df['property'])
        self.df['otherinstallments'] = LE.fit_transform(self.df['otherinstallments'])
        self.df['housing'] = LE.fit_transform(self.df['housing'])
        self.df['job'] = LE.fit_transform(self.df['job'])
        self.df['phone'] = LE.fit_transform(self.df['phone'])
        self.df['gender'] = LE.fit_transform(self.df['gender'])
        
        #Transforming target variable
        self.df['creditworthy'] = LE.fit_transform(self.df['creditworthy'])

    # For verifying data quality
    def printDataQuality(self):
        print("--------------- Validating if there are NULL values ----------------\n")
        print(self.df.info()) # We could verify if there are any null values
        print('--------------- Dataframe Head Info ----------------\n')
        print(self.df.head())
        print()        
        print('--------------- Validating Multivariate outliers in n-dimensional space ----------------\ \n')
        print(self.df.describe())
        print()
        
        
    # Returns the filtered dataframe after removing outliers
    def removeOutliers(self):
        z_scores = stats.zscore(self.df)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        return self.df[filtered_entries]

    # For visualizing outliers in our top features
    # Helps us to understand the variability from lower and upper quartiles
    def visualizeOutliers(self, features): 
        print("--------------- Visualizing outliers ----------------\n")
        filtered_df = pd.DataFrame(data = np.random.random(size=(6,6)), columns = features)
        sns.boxplot(x="variable", y="value", data=pd.melt(filtered_df))
        plt.show()
        # seems like we have lot of points for credit amount outside the box of observation 
        
        
    # Getter for Dataframe  
    @property
    def getdataFrame(self):
        return self.df       
        


