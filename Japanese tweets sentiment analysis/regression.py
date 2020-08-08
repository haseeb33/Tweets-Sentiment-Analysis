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

df = pd.read_excel("Final1.xlsx")
df_topics = pd.read_excel("topic_modeling/Hashtag_NLP_JP100Topics65users.xlsx", sheet_name="ThetaValues")

#X, H, I, J = functions_.negative_tweet_porpostion_X(df) # negative proportion as X
#X, H, I, J = functions_.negative_tweet_porpostion_X_65_users(df)
#X, H, I, J = functions_.vocab_count_X(df) #specific vocabulary count as X
#X, H, I, J = functions_.vocab_count_X_65_users(df) #not a good idea to use
#X, H, I, J = functions_.tweet_count_X_65_users(df)
#X, H, I, J = functions_.gender_X(df)
#X, H, I, J = functions_.followers_X(df)
#X, H, I, J = functions_.following_X(df)
#X, H, I, J = functions_.followers_following_X(df)
#X, H, I, J = functions_.all_numerical_params_X(df)
X, H, I, J = functions_.topic_proportion_X_65_users(df, df_topics)
X = choose_col(X)

H = normalization(H); I = normalization(I); J = normalization(J)

#reg_H = linear_model.Lasso(alpha=0.1).fit(X, H)
#print(r2(H, reg_H.predict(X)), r2(I, reg_I.predict(X)), r2(J, reg_J.predict(X))) 

reg_H = linear_model.Lasso(alpha=0.1).fit(X, H)
reg_I = linear_model.Lasso(alpha=0.1).fit(X, I)
reg_J = linear_model.Lasso(alpha=0.1).fit(X, J)

"""
H_results = cross_validate(reg_H, X, H, cv=13, scoring=('r2', 'mean_squared_error'))
I_results = cross_validate(reg_I, X, I, cv=13, scoring=('r2', 'mean_squared_error'))
J_results = cross_validate(reg_H, X, J, cv=13, scoring=('r2', 'mean_squared_error'))

print(H_results['test_r2'])
print(I_results['test_r2'])
print(J_results['test_r2'])
"""


