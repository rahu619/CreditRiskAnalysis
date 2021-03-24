#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Mon Mar 22 13:16:06 2021

@author: Rahul
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from persistence import Persistence

class Algorithm:

    # TODO: implement later
    usePersistedModel = False

    def __init__(self, usePersistedModel):
        self.usePersistedModel = usePersistedModel

    # using KNN algorithm
    # saving Model for this credit dataset might be an overkill;
    # but just adding it for the sake of it

    def KNNApproach(self, X, y):

        # create model
        # k = 25

        knn = neighbors.KNeighborsClassifier(n_neighbors=25,
                weights='uniform')

        (X_train, X_test, y_train, y_test) = train_test_split(X, y,
                test_size=0.2)

        knn.fit(X_train, np.ravel(y_train, order='C'))
        
        # if persistence is set to true
        if self.usePersistedModel:
            persistenceObj = Persistence('kNN');
            model = persistenceObj.LoadModel()
            
            #if model doesn't exist, saving it for later use
            if not model:
               persistenceObj.SaveModel(knn)
                
        prediction = knn.predict(X_test)

        return ([X_train, X_test, y_train, y_test], prediction)
