import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("mlr_dataset.csv", sep=";")
# print(df)

x = df.iloc[:, [0, 2]].values
y = df.maas.values.reshape(-1, 1)

multiple_lr = LinearRegression()

multiple_lr.fit(x, y)

print("b0 : ", multiple_lr.intercept_)
print("b1 , b2 : ", multiple_lr.coef_)

tahminEt = multiple_lr.predict(np.array([[10, 35], [5, 35]]))
print(tahminEt)

