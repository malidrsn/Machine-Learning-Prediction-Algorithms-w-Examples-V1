# residual = y-y_head
# square residual = residual^2
# sum square residual = sum((y-y_head)^2) = SSR
# y_avg=12000 olsun
# sum square total = sum((y-y_avg)^2) = SST
# R^2 = R square değerlendirme metodudur.
# R^2 = R square  = 1-(SSR/SST) eğer R^2 değeri 1'e ne kadar yakın ise o kadar iyidir.
# **********"""random forest'a uygulanması""""************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("random_forest.csv", sep=";", header=None)
x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

from sklearn.ensemble import RandomForestRegressor

# random_state ise bir önceki random kadar bölmesini sağlar
# aynı random değerlerinin seçilmesini sağlar
rf = RandomForestRegressor(n_estimators=100, random_state=42)  # n_est = kaç tane tree kullanacağımız anlamına gelir
rf.fit(x, y)

y_head = rf.predict(x)
# R Square
from sklearn.metrics import r2_score

print("R_Score", r2_score(y, y_head))
