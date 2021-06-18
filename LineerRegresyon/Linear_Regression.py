import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("Linear_regresyon.csv",
                 sep=",")  # seperator ise ayırıcıdır. Csv dosyası hangi sep ile tutulmuşsa o yazılır

plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

# örnek bir Line çizeriz ve bu line matematiksel olarak y= b0 + b1*x
# b0 çizilen line'in y eksenini kestiği nokta
# b1 ise x ekseni ile arasında ki açı değeridir.
# b0 = constant(bias) ve b1 = coefficient(coeff) eğim = karşı / komşu
# residual = y- y_head (y değerinden tahmin edilen y değerinin çıkarılmasıdır) + ve - olabilir
# bunu residual'ın karesini alarak + hala getirebiliriz. residual^2
# daha sonraasında bunların hepsi toplanır ve toplam hatamızı görürüz error değeri elde edilir
# daha sonra ise bu değer toplam değer sayısına bölünür bu örnekte 14 sample vardır ve bize MSE'yi verir
# MSE = mean squarred error bulunması gerekir.
# burada en küçük hatayı bularak bu noktalar üzerindeki en uygun line çizmeliyiz.
