#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:23:15 2019

@author: khan
"""
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor

import numpy as np
import pandas as pd
import functions_

def r2(y_true, X,  model):
    return metrics.r2_score(y_true, model.predict(X))

def mse(y_true, X, model):
    return metrics.mean_squared_error(y_true, model.predict(X))

def predict_visual(x_train, y_train, lr):
    plt.scatter(x_train, y_train, color = "red")
    plt.plot(x_train, lr.predict(x_train), color = "green")
    plt.show()

def choose_col(X, col=[4,38,68,58,49,  98,5,37,89,53]):
    # These coloums have highest corelation with social skills (+++++ -----)
    new_X = []
    for i in X:
        new_X.append([i[4], i[38], i[68], i[58], i[49], i[98], i[5], i[37], i[89], i[53]])
    return new_X

def normalization(raw):
    return [float(i)/sum(raw) for i in raw]

def BaseLine(X, H, I, J):
    dummy_H = DummyRegressor(strategy="mean")
    dummy_I = DummyRegressor(strategy="mean")
    dummy_J = DummyRegressor(strategy="mean")
        
    H_results = cross_validate(dummy_H, X, H, cv=13, scoring=('r2', 'neg_mean_squared_error'))
    I_results = cross_validate(dummy_I, X, I, cv=13, scoring=('r2', 'neg_mean_squared_error'))
    J_results = cross_validate(dummy_J, X, J, cv=13, scoring=('r2', 'neg_mean_squared_error'))
    
    print("H_neg_MSE", np.mean(H_results['test_neg_mean_squared_error']))
    print("I_neg_MSE", np.mean(I_results['test_neg_mean_squared_error']))
    print("J_neg_MSE", np.mean(J_results['test_neg_mean_squared_error']))
    
df = pd.read_excel("Check.xlsx")
df_topics = pd.read_excel("topic_modeling/NLP_JP100_150_users_topic.xlsx", sheet_name="ThetaValues")
H_I_J_cols = ['Generalized trust', 'Social skill', 'Well-being']

#X, H, I, J = functions_.negative_tweet_porpostion_X(df, H_I_J_cols) # negative proportion as X
#X, H, I, J = functions_.negative_tweet_count_X(df, H_I_J_cols)
#X, H, I, J = functions_.negative_tweet_porpostion_X_150_users(df, H_I_J_cols)
#X, H, I, J = functions_.negative_tweet_count_X_150_users(df, H_I_J_cols)

#X, H, I, J = functions_.positive_tweet_porpostion_X(df, H_I_J_cols)
#X, H, I, J = functions_.positive_tweet_count_X(df, H_I_J_cols)
#X, H, I, J = functions_.positive_tweet_porpostion_X_150_users(df, H_I_J_cols)
#X, H, I, J = functions_.positive_tweet_count_X_150_users(df, H_I_J_cols)

#X, H, I, J = functions_.tweet_count_X_150_users(df, H_I_J_cols)

#X, H, I, J = functions_.gender_X(df)
#X, H, I, J = functions_.gender_X_150_users(df, H_I_J_cols)

#X, H, I, J = functions_.topic_proportion_X_150_users(df, df_topics, H_I_J_cols)

#X, H, I, J = functions_.positive_negative_tweet_topic_proportion_X_150_users(df, df_topics, H_I_J_cols)

#X = choose_col(X)

#H = normalization(H); I = normalization(I); J = normalization(J)
"""
reg_H = linear_model.LinearRegression().fit(X, H)
reg_I = linear_model.LinearRegression().fit(X, I)
reg_J = linear_model.LinearRegression().fit(X, J)
"""

#print(r2(H, reg_H.predict(X)), r2(I, reg_I.predict(X)), r2(J, reg_J.predict(X))) 

# Lasso is Linear Regression with Regularization parameter

reg_H = linear_model.Lasso(alpha=0.1).fit(X, H)
reg_I = linear_model.Lasso(alpha=0.1).fit(X, I)
reg_J = linear_model.Lasso(alpha=0.1).fit(X, J)

H_results = cross_validate(reg_H, X, H, cv=13, scoring=('r2', 'neg_mean_squared_error'))
I_results = cross_validate(reg_I, X, I, cv=13, scoring=('r2', 'neg_mean_squared_error'))
J_results = cross_validate(reg_H, X, J, cv=13, scoring=('r2', 'neg_mean_squared_error'))

print("H_neg_MSE", np.mean(H_results['test_neg_mean_squared_error']))
print("I_neg_MSE", np.mean(I_results['test_neg_mean_squared_error']))
print("J_neg_MSE", np.mean(J_results['test_neg_mean_squared_error']))



