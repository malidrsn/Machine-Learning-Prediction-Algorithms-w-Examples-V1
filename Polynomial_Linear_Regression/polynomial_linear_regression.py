import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynominal_regression.csv", sep=";")

y = df.araba_max_hiz.values.reshape(-1, 1)
x = df.araba_fiyat.values.reshape(-1, 1)

plt.scatter(x, y)
plt.ylabel("araba max hız")
plt.xlabel("araba fiyat")

# linear regression y=b0+b1*x
# multiple linear regression y = b0 + b1*x1 + b2*x2

# linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x, y)

#predict
y_head = lr.predict(x)
print(y_head)

plt.plot(x, y_head, color="red")
plt.show()

abc = lr.predict([[10000]])
print("a = ", abc)

#polynomial linear regression y = b0 + b1*x + b2*x^2 + b3*x^3 + ..... + bn*x^n