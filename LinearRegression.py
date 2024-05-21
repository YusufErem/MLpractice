import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

veriler = pd.read_csv('satislar.csv')

aylar  = veriler[['Aylar']]
satislar= veriler[['Satislar']]
# satislar = veriler.iloc[:,:1]
# aylar = veriler.iloc[:,1:2]#.values
#print(aylar)

x_train, x_test, y_train, y_test, = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

# sc = StandardScaler()
# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)
# Y_train = sc.fit_transform(y_train)
# Y_test = sc.fit_transform(y_test)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# print(y_test)
# print(tahmin)

x_train = x_train.sort_index()
y_train = y_train.sort_index()
x_test = x_test.sort_index()
y_test = y_test.sort_index()
tahmin = lr.predict(x_test)
# print(x_train)
# print(x_train,y_train,x_test,lr.predict(x_test))
plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)
plt.title("Aylara Gore Satis")
plt.xlabel("Aylar")
plt.ylabel("Satislar")
plt.show()
