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

# prediction (tahmin)
b0 = linear_reg.predict([[0]])
print("b0", b0)
# b0 bulunmasını sağlıyor
b00 = linear_reg.intercept_  # y eksenini kestiği nokta
print("b00", b00)

b1 = linear_reg.coef_  # eğimi verir
print("b1", b1)

# maas = 1663 +1138*deneyim diyebiliriz
yeni_maas = b0 + b1 * 11
print("Yeni maaş", yeni_maas)
# yada
yeni_maas2 = linear_reg.predict([[11]])
print("Yeni Maaş 2 :", yeni_maas2)

array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape(-1, 1)
y_head = linear_reg.predict(array)  # maaş
plt.plot(array, y_head, color="red")
plt.show()

yuzyil_calisan = linear_reg.predict([[100]])
print("100 Yılı deneyimi olan ", yuzyil_calisan)
