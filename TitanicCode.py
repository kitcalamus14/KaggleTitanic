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

#Preliminary Analysis /EDA
##Descriptive Statistics

colname = list(dat1.columns.values)
colname
colname.insert(0,'Stat')
DescriptiveStat = pd.DataFrame(columns = colname)
missingval = dat1.isnull().sum(axis=0)
mean = dat1.mean()

#preliminary ML
#Data Cleansing and XY split
del dat1['Name']
count = dat1.iloc[:,4].isna().sum()
dat1 = dat1.dropna(subset=['Age'])
dat1 = dat1.dropna(subset=['Embarked'])
X = dat1.copy()
del X['Survived']
del X['Ticket']
del X['Cabin']
X['Sex'].replace('male',1,inplace=True)
X['Sex'].replace('female',2,inplace=True)
X['Embarked'].replace('C',1,inplace=True)
X['Embarked'].replace('Q',2,inplace=True)
X['Embarked'].replace('S',3,inplace=True)
y = dat1['Survived']



#NB MAIN CODE and Test/train split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

#NB EVALUATION
from sklearn import metrics
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))

#Preliminary study suggested a result with 0.778 accuracy, assuming random accuracy is .5 and 1 is perfect prediction, 0.778 is moderate.

#Issue encountered
#Age is continuous variables, which won't work well with NB
##has to convert into categorical variable by quartile differences
##Removed Variables may contained useful information. Find a way to use it
##NAN data, assuming its not random/random, perform missing value imputation and compare result

#Fixing Age Variable
dat1['Age'] = pd.qcut(dat1['Age'],8, labels=[1,2,3,4,5,6,7,8])
X = dat1.copy()
del X['Survived']
del X['Ticket']
del X['Cabin']
X['Sex'].replace('male',1,inplace=True)
X['Sex'].replace('female',2,inplace=True)
X['Embarked'].replace('C',1,inplace=True)
X['Embarked'].replace('Q',2,inplace=True)
X['Embarked'].replace('S',3,inplace=True)
y = dat1['Survived']

#2nd NB MAIN CODE and Test/train split 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

#2nd NB EVALUATION
from sklearn import metrics
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))

#Fixing NAN value
