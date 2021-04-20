#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Mon Mar 22 13:16:06 2021

@author: Rahul
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors
from persistence import Persistence
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class Classifier:

    usePersistedModel = False
    df = None
    X = None
    y = None 
    
    def __init__(self, df, usePersistedModel):
        self.df = df
        self.usePersistedModel = usePersistedModel
        self.initializeVariables()
        
    
    # Intializing predictor and target variables
    # Features will chosen manually 
    # based on our analysis done earlier.
    def initializeVariables(self):   
        
       # The independent / predictor variables
        self.X = self.df[[
            'creditamount',
            'status',            
            'age',
            'duration',
            'purpose',
            'credithistory'
            ]].values


       # The dependent / target variable
        self.y = self.df[['creditworthy']]
                 
       # TODO: plot data?

       # Preprocessing step 
       # Transforming the textual data to numbers
       # as we can't the data directly into the algorithm
        Le = LabelEncoder()

       # transforming X column wise
        for i in range(len(self.X[0])):
            self.X[:, i] = Le.fit_transform(self.X[:, i])

       # transforming y column
        label_mapping = {'Not Worthy': 0, 'Worthy': 1}

        self.y = self.y.copy()
        self.y['creditworthy'] = self.y['creditworthy'].map(label_mapping)
        self.y = np.array(self.y)
        
        
        
    # using KNN algorithm
    # TODO: Dimensionalty Reduction
    def knnApproach(self):
        
        print("-------------KNN Algorithm-------------------")
        print()
        
        # Dimenionsality reduction
        # create model
        
        # The optimal 'K' value will be square root of N
        # where N is the no:of samples 
        # since we have 988 samples; the K value will be ~31

        # Nearest neighbour classifier
        knn = neighbors.KNeighborsClassifier(n_neighbors=31,
                weights='uniform')

        # Allocating 20% of the data sample as test data
        (X_train, X_test, y_train, y_test) = train_test_split(self.X, self.y, test_size=0.2)

        # Fitting model
        knn.fit(X_train, np.ravel(y_train, order='C'))
        
        self.PersistModel('kNN', knn) # persisting model
                
        prediction = knn.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, prediction)

        # print("KNN predictions :", prediction)
        print("KNN accuracy : %0.2f" % accuracy)       
        print()


    # Using Linear SVM 
    # Using Cross validation instead of train_test_split method
    def svmApproach(self):
        
        print("-------------SVM Algorithm-------------------")
        print()

        # Added hyperparameter c setting
        # yet to be optimized
        clf = svm.SVC(verbose=True, kernel='linear', C = 1)

        # Trying 5 fold cross validation
        predicted = cross_val_predict(clf, self.X, np.ravel(self.y, order='C'), cv = 5)
        
        accuracy = metrics.accuracy_score(self.y, predicted)

        print("\nSVM Accuracy: %0.2f" % accuracy)
        
        
    def randomForestApproach(self, X = None, y = None):
        if X is None or y is None:
            X = self.X
            y = self.y
            print('------------Random Forest Algorithm----------------')
            
        # Create decision tree classifer object
        clf = RandomForestClassifier(random_state=0, n_jobs=-1)
        
        # Train model
        model = clf.fit(X, np.ravel(y, order='C'))
        
        
        # Trying 5 fold cross validation
        predicted = cross_val_predict(clf, X, np.ravel(y, order='C'), cv=5)
        
        accuracy = metrics.accuracy_score(y, predicted)
        
        print("Random Forest accuracy : %0.2f" % accuracy)       
        print()
        
        return model
    
    # Using random forest to achieve this.
    # Considering all input variables so that we could create a bar chart to identify
    # the most important ones.
    def plotImportantFeatures(self):
        cols = list(self.df.columns.values) #Make a list of all of the columns in the df
        cols.pop(cols.index('creditworthy')) #Remove b from list
        self.df = self.df[cols+['creditworthy']] #Create new dataframe with columns in the order you want
        
        array = self.df.values

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

       # passing in predictor and target variables into our RFA method
        model = self.randomForestApproach(X, y)
        
        # Calculate feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Rearrange feature names so they match the sorted feature importances
        names = [self.df.columns[i] for i in indices]
        
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
    
        
    # Persisting the model so that we could use the
    # already saved model next time.     
    def PersistModel(self, classifierName , classifierObj):        
        # Only if usePersistedModel is set to true
        if self.usePersistedModel:
            persistenceObj = Persistence(classifierName);
            model = persistenceObj.LoadModel()
            
            #if model doesn't exist, saving it for later use
            if not model:
               persistenceObj.SaveModel(classifierObj)
