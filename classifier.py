#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Mon Apr 12 13:16:06 2021

@author: Rahul
"""
import numpy as np
from persistence import Persistence
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

class Classifier:

    usePersistedModel = False
    df = None
    X = None
    y = None 
    
    def __init__(self, df, usePersistedModel):
        self.df = df
        self.usePersistedModel = usePersistedModel
        
    # Intializing predictor and target variables
    # Features will chosen manually 
    # based on our analysis done using Feature importance plotting.
    def initializeVariables(self, features):           
       # The independent / predictor variables
        self.X = self.df[features].values
       # The dependent / target variable
        self.y = self.df[['creditworthy']]
                
        
    # using KNN algorithm
    # TODO: Dimensionalty Reduction
    def knnApproach(self):
        
        print("------------- KNN Algorithm -------------------\n") 
        # Dimenionsality reduction could also be introduced
        # The optimal 'K' value will be square root of N
        # where N is the no:of samples 
        # since we have 988 samples; the K value will be ~31

        # Nearest neighbour classifier
        clf = neighbors.KNeighborsClassifier(n_neighbors=31, weights='uniform')

        # Allocating 20% of the data sample as test data
        (X_train, X_test, y_train, y_test) = train_test_split(self.X, self.y, test_size=0.2)

        # Fitting model
        clf.fit(X_train, np.ravel(y_train, order='C'))
        
        self.PersistModel('kNN', clf) # persisting model
        prediction = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, prediction)
        print("KNN accuracy : %0.2f \n" % accuracy)       


    # Using Linear SVC instead of SVC class as it has a much slower performance
    def svmApproach(self):        
        print("------------- SVM Algorithm -------------------\n")
        
        clf = make_pipeline(StandardScaler(),
                            LinearSVC(random_state=0, tol=1e-5, dual=False))
        # Trying 5 fold cross validation instead of train_test_split method
        predicted = cross_val_predict(clf, self.X, np.ravel(self.y, order='C'), cv = 5)
        accuracy = metrics.accuracy_score(self.y, predicted)
        print("Linear SVM Accuracy: %0.2f \n" % accuracy)
            
        
    # Random Forest Decision Tree   
    def randomForestApproach(self, X = None, y = None):
        isPlot = (X is not None and y is not None)
        if not isPlot:
            X = self.X
            y = self.y 
            
        # Create decision tree classifer object
        # Tweaking hyper parameters for optimal prediction
        clf = RandomForestClassifier(n_estimators = 120, max_depth = 4, random_state = 0, n_jobs = -1)
        
        # Only performing 3-fold cross validation, if it's not to plot 
        # a feature importance graph 
        if not isPlot:
            predicted = cross_val_predict(clf, X, np.ravel(y, order='C'), cv = 5)            
            accuracy = metrics.accuracy_score(y, predicted)
            print('------------ Random Forest Algorithm ----------------\n')            
            print("Random Forest accuracy : %0.2f \n" % accuracy)
           
        return clf
    
    # Using random forest to achieve this.
    # Considering all input variables so that we could create a bar chart to identify
    # the most important ones.
    def plotImportantFeatures(self):
        print("--------------- Plotting important features ----------------\n")
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
      
        # passing in predictor and target variables into our RFA method
        clf = self.randomForestApproach(X, y) 
        model = clf.fit(X, np.ravel(y, order='C')) # Training model       
        importances = model.feature_importances_   # Calculating feature importances
        indices = np.argsort(importances)[::-1]    # Sorting feature importances in descending order
        
        # Rearrange feature names so they match the sorted feature importances
        names = [self.df.columns[i] for i in indices]        
        # print('importances: ', names)

        # Barplot
        plt.bar(range(X.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(X.shape[1]), names, rotation=20, fontsize = 8)
        # Create plot title
        plt.title("Feature Importance Chart")  
        
        # Based on plot the top predictor variables are
        # Credit amount per month, Status, Age, Duration, Purpose, Credit History
        
        # Show plot
        plt.show()
    
    
        
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
