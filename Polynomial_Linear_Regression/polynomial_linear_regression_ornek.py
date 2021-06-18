import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynominal_regression.csv", sep=";")

y = df.araba_max_hiz.values.reshape(-1, 1)
x = df.araba_fiyat.values.reshape(-1, 1)

plt.scatter(x, y)
plt.ylabel("araba max hız")
plt.xlabel("araba fiyat")

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x, y)

# predict
y_head = lr.predict(x)
print(y_head)

plt.plot(x, y_head, color="red", label="linear")

# polynomial regression

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree=4)  # degree artarsa mse (ortalama hata) azalır
x_polynomial = polynomial_regression.fit_transform(x)

# fit
linear_regression_2 = LinearRegression()
linear_regression_2.fit(x_polynomial, y)

y_head2 = linear_regression_2.predict(x_polynomial)

plt.plot(x, y_head2, color="green", label="poly")
plt.legend()
plt.show()
