# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:08:10 2018

@author: Chen
"""
import pandas
import numpy as np
from sklearn import tree
from sklearn.cross_validation import train_test_split
data = pandas.read_csv('BlackFriday.csv', sep=',')

#print(data)

X = data.iloc[:, 0:9]
y = data.iloc[:, 9]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.05)

clf = tree.DecisionTreeClassifier()
FM_clf = clf.fit(train_X, train_y)

# 預測
test_y_predicted = FM_clf.predict(test_X)
print(test_y_predicted)

# 標準答案
print(test_y.tolist())