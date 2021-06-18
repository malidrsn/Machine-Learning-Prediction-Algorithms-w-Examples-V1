# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import data
df = pd.read_csv("Linear_regresyon.csv", sep=",")

# plot data
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")

"""
sklearn kütüphanesini kullanacağız. Burada machine learning algoritmaları bulunmaktadır.
biz burada sklearn içerisinden LinearRegression modelini import edecez
"""
# sklearn(sci-kit learning library
from sklearn.linear_model import LinearRegression

# Linear regression model
linear_reg = LinearRegression()
# x = df.deneyim.values
# x = x.shape  # bu kısımda çıktımız (14, ) şeklindedir ama ikinci parametre olması gerekmektedir.
# bu yüzden reshape edilerek tekrar düzenlenmesi gerekmektedir. Çünkü sklearn okuyamaz bunu (14,1)

x = df.deneyim.values.reshape(-1, 1)
y = df.maas.values.reshape(-1, 1)

linear_reg.fit(x, y)

y_head = linear_reg.predict(x)
plt.plot(x, y_head, color="red")

# R_Square
from sklearn.metrics import r2_score

print("R_Square : ", r2_score(y, y_head))
