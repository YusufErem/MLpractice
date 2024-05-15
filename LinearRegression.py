import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('satislar.csv')

# aylar  = veriler[['Aylar']]
# satislar= veriler[['Satislar']]
satislar = veriler.iloc[:,:1]
aylar = veriler.iloc[:,1:2]


print(aylar)
