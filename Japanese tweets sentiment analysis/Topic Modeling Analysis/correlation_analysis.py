#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:28:50 2019

@author: khan
"""
import pandas as pd

theta_df = pd.read_excel("NLP_JP100topic.xlsx", sheet_name = "ThetaValues", index_col=0)
original_df = pd.read_excel("Final1.xlsx", usecols = ["通し番号","一般的信頼合計", "社会的スキル合計", "心理的幸福感合計"])
imp_df = original_df.loc[original_df["通し番号"].isin(theta_df[300])]
H_col = imp_df["一般的信頼合計"].values.tolist()
I_col = imp_df["社会的スキル合計"].values.tolist()
J_col = imp_df["心理的幸福感合計"].values.tolist()
theta_df["H"] = H_col
theta_df["I"] = I_col
theta_df["J"] = J_col
corr = theta_df.corr()

new_df = pd.DataFrame([corr["H"], corr["I"], corr["J"]])
new_df = new_df.T
new_df = new_df.sort_values(by=['H','I','J'], ascending=False)
new_df.to_excel("CorrWithTopics.xlsx")
