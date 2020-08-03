#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:23:15 2019

@author: khan
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics

import numpy as np
import pandas as pd
import functions_

def r2(y_true, y_pred):
    return metrics.r2_score(y_true, y_pred)

def predict_visual(x_train, y_train, lr):
    plt.scatter(x_train, y_train, color = "red")
    plt.plot(x_train, lr.predict(x_train), color = "green")
    plt.show()

df = pd.read_excel("Final1.xlsx", index="")

#X, H, I, J = functions_.negative_tweet_porpostion_X(df) # negative proportion as X
#X, H, I, J = functions_.negative_tweet_porpostion_X_65_users(df)
#X, H, I, J = functions_.vocab_count_X(df) #specific vocabulary count as X
#X, H, I, J = functions_.vocab_count_X_65_users(df) #not a good idea to use
#X, H, I, J = functions_.tweet_count_X_65_users(df)
#X, H, I, J = functions_.gender_X(df)


reg_H = LinearRegression().fit(X, H)
reg_I = LinearRegression().fit(X, I)
reg_J = LinearRegression().fit(X, J)
Y_I = reg_I.predict(X)
Y_J = reg_J.predict(X)
print(r2(H, reg_H.predict(X)), r2(I, reg_I.predict(X)), r2(J, reg_J.predict(X))) 

