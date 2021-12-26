#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Random forest.py
@Time    :   2021/12/26 20:45:59
@Author  :   Pei Kaiyu 
@Version :   1.0
@Contact :   spicap0103@outlook.com
@License :   none
@Desc    :   None
'''

# here put the import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn import datasets


def RandomForest():
    wine = datasets.load_wine()
    x = wine["data"]
    y = wine["target"]
    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=1, train_size=0.6,test_size=0.4)
    #print("\ntrain size：", x_train.shape)
    #print("train lable：", y_train.shape)
    #print("test size：", x_test.shape)
    #print("test label：", y_test.shape)
    accuracy = 0
    score = 0
    for i in range(500):
        clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test) + score
    accuracy = score/500
    print("avg_accuracy=", accuracy)
    plt.plot
    
    
    
def RandomForest_test():
    wine = datasets.load_wine()
    x = wine["data"]
    y = wine["target"]
    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=1, train_size=0.6,test_size=0.4)
    

if __name__ == "__main__":
    RandomForest()