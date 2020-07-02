# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:06:54 2020

@author: kit
"""


import pandas as pd
import kaggle
import os

os.listdir()

dat1 = pd.read_csv('train.csv')
testset1 = pd.read_csv('test.csv')
#Preliminary analysis#
#Data is clean with no missing value
#Structured with input and output variable
#Sex variable contain string variables
# 

#Methodology#
#proposed steps 1. Remove unwanted variables
#proposed steps 2. Perform preliminary ML
#proposed steps 3. analyse result, indentify implicit useless variables
#proposed steps 4. remove identified useless variable 1 by 1, loop and compare result
#proposed steps 5. finalize selected variable, output result for testing


#preliminary ML

del dat1['Name']







from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


X = dat1.copy()
del X['Survived']
y = dat1['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit()




#outputdat1 = dat1['Survived']
#inputdat1 = dat1
#del inputdat1['Survived']