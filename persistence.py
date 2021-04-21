#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed Apr 7 10:25:38 2021

@author: Rahul
"""

from joblib import dump, load
from pathlib import Path

class Persistence:

    basePath = './models'
    fileName = ''

    def __init__(self, fileName):
        self.fileName = fileName
                

    # getter for the filepath    
    @property    
    def filepath(self):
        if not self.fileName:
            raise ValueError('Please provide a valid filename!')
        return f'{self.basePath}/{self.fileName}.joblib'

    def ModelExists(self):
        modelfile = Path(self.filepath)
        return modelfile.is_file()

    def SaveModel(self, model):
        dump(model, self.filepath)

    def LoadModel(self):
        if self.ModelExists():
            return load(f'{self.filepath}')