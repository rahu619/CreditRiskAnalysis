# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:16:06 2021
Retrieves the X and y variables.

@author: Rahul
"""

    # Manually choosing features
def retrieveVariablesManually(data):

    #The independent / predictor variables
    X = data[[
        'foreignworker',
        'status',
        'credithistory',
        'savings',
        'employmentsince',
        'creditamount',
        'age'
        ]].values
    
    # The dependent / target variable
    y = data[['creditworthy']]

    return X, y


    