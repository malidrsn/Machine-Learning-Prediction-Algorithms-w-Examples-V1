# decision tree
# CART = Classification And Regression Tree
# Regression
# temelinde split'ler vardır.
# information entropi'ye göre data split ediliyor.
# x1<10 E? H? yes or no şeklinde
# x2<35 Yes ? No ? şeklinde ağaç yapılan regression modelidir.
# veriler splitlere ayrılır split1 split2 vb. her bölge terminal leaf olarak geçer

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset.csv", sep=";", header=None)

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

# decision tree regression
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x, y)

print(tree_reg.predict([[5.5]]))
x_ = np.arange(min(x), max(x), 0.01).reshape(-1, 1)

y_head = tree_reg.predict(x_)

# visualize
plt.scatter(x, y, color="red")
plt.plot(x_, y_head, color="blue")
plt.xlabel("Tribün level")
plt.ylabel("Fiyat")
plt.show()
