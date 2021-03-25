# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:56:05 2021

@author: Rahul
"""
import seaborn as sns
import matplotlib.pyplot as plt

class plot:
    
    def Visualize(self, data):
        data.set(rc={'figure.figsize':(11.7,8.27)})
        # create a countplot
        data.countplot('Route To Market', data = data, hue = 'Opportunity Result')
        # Remove the top and down margin
        data.despine(offset = 10, trim = True)
        # display the plotplt.show()
        