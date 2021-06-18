# Rassal ağaçlar
# ensemble learning üyesidir.
# ensemble learning = aynı anda birden fazla algoritmayı kullanarak ortalamısını alan ve sonuç veren algoritma
# random forest = ağaçların toplamıdır. decision tree toplanıyor sonuç alınıyor
# data üzerinden n sayıda örnek alınması ve bu örneklerin farklı tree'ler ile eğitilmesi ve sonuç olarak tree'lerin sonuçlarının toplanması şeklindedir.
# recommendation sistemlerinde kullanılır. film izledikten sonra benzer tavsiyeler vermesi
# body part classification
# stock price prediction
# decision tree'ye göre daha iyi sonuçlar verir

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

print("7.8 seviyesinde fiyatın ne kadar olduğu : ", rf.predict([[7.8]]))

x_ = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
y_head = rf.predict(x_)

plt.scatter(x, y, color="red")
plt.plot(x_, y_head, color="blue")
plt.xlabel("tribün level")
plt.ylabel("fiyat")
plt.show()
